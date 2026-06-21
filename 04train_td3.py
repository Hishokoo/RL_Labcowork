# 執行前請安裝：pip install "gymnasium[mujoco]" stable-baselines3 moviepy tqdm rich
# 規格來源：markdown/04_train_td3_spec.md（沿用 03，針對「步態品質指標」再優化）
#
# 與 03 的差異（目標：讓 diagonal_sync / 週期性 / jerk / 姿態 等指標真正進步）：
#   1. gait_mode="antiphase"  —— 步態 reward 改以「對角線反相」為主導。03 的 legacy 公式
#      讓站著不動的 r_gait = 1.6（比走路還高），本身就是個站著 attractor；antiphase 讓
#      靜態姿勢拿不到分，逼出真正的交替踏步。
#   2. forward_mode="progress" —— 速度 reward 改為 max(0, min(x_vel, target))。03 的
#      -|x_vel-target| 在站著時 = -1.0，正是 markdown/ant_v5_attractor_fix.md 實測「會逼出
#      快速摔倒擺爛」的 speed penalty；progress 讓站著 = 0、走路才是唯一正收益。
#   3. smooth_weight / tilt_weight —— 新增 jerk 與軀幹直立懲罰，直接壓「抽搐」、穩姿態。
#   4. 評估強化：每個 eval interval 跑 deterministic episode 算整段 scorecard（速度誤差、
#      CoT、jerk、anti_phase、週期性）寫進 TensorBoard；並存中間 checkpoint（03 只有 final）。
#
# 站著 vs 走路的每步 reward（約略）：03 legacy 站著 +1.6 > 走路 +0.8（站著贏，危險）；
# 04 站著 ≈ +0.8、走路 ≈ +1.7（走路明顯勝出，gap 由 -0.8 翻成 +0.9）。站著仍為正（非負），
# 故不會觸發 ant_v5_attractor_fix.md 記載的「負向懲罰 → 快速摔倒擺爛」失敗模式。
import os
from functools import partial

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from tools.gait_wrapper import RealisticGaitWrapper
from tools import gait_metrics

# ── Hyperparameters ───────────────────────────────────────────────────────────
ENV_NAME        = "Ant-v5"
SEED            = 0
# 平行環境數：N_ENVS=1 為單環境（與 03 可公平比較）。多環境時下方會把 train_freq/gradient_steps
# 設成維持 1:1 的梯度更新比例，總步數與更新數不變。
# 註：實測 8 環境只比單環境快 ~13%（CPU 仍只用 ~1.2 核、GPU 14%）——瓶頸在 SB3 單執行緒的
# 梯度更新/取樣迴圈，不在環境收集，加環境數幫助有限。故預設回 1。保留多環境路徑備用。
N_ENVS          = int(os.environ.get("N_ENVS", 1))
# 多數旋鈕可用環境變數覆寫，讓每個 reward 實驗只要換 env vars + OUTPUT_DIR，不必新增檔案。
# 例（05 修站著 attractor，先跑 300k 短驗證、關錄影）：
#   OUTPUT_DIR=output/05_gated GAIT_MODE=antiphase_gated FORWARD_WEIGHT=2 \
#   MAX_TIMESTEPS=300000 VIDEO_INTERVAL=9999999 MUJOCO_GL=egl python 04train_td3.py
MAX_TIMESTEPS   = int(os.environ.get("MAX_TIMESTEPS", 1_000_000))
LEARNING_STARTS = 10_000
EVAL_INTERVAL   = int(os.environ.get("EVAL_INTERVAL", 50_000))       # 數值 scorecard 間隔（便宜）
VIDEO_INTERVAL  = int(os.environ.get("VIDEO_INTERVAL", 200_000))     # 錄影間隔（moviepy 編碼貴，與 scorecard 解耦以加速）
CHECKPOINT_FREQ = int(os.environ.get("CHECKPOINT_FREQ", 100_000))    # 中途存檔間隔
OUTPUT_DIR      = os.environ.get("OUTPUT_DIR", "output/04train_td3")

# RealisticGaitWrapper 參數（reward 旋鈕可用 env var 覆寫以快速迭代）。
# 經三次失敗（站著 / 原地踏步 / 摔死）後回到 ant_v5_attractor_fix.md 已驗證會走的配方：
# forward 主導 + 溫和懲罰 + 小 alive 底分（防摔死）+ 「只有前進才吃得到」的步態 garnish（forward_gated）。
TARGET_SPEED    = 1.0
CTRL_WEIGHT     = float(os.environ.get("CTRL_WEIGHT", 0.5))    # 溫和（原 5.0 太重→逼出摔死）
GAIT_WEIGHT     = float(os.environ.get("GAIT_WEIGHT", 2.0))    # 步態品質乘子（只在前進時生效）
POSTURE_WEIGHT  = float(os.environ.get("POSTURE_WEIGHT", 0.5)) # 溫和
ALIVE_WEIGHT    = float(os.environ.get("ALIVE_WEIGHT", 0.5))   # 小底分：站著≈+0.4 非負→不摔死；但遠小於走路→不站著
GAIT_MODE       = os.environ.get("GAIT_MODE", "antiphase_gated")
FORWARD_MODE    = os.environ.get("FORWARD_MODE", "progress")
FORWARD_WEIGHT  = float(os.environ.get("FORWARD_WEIGHT", 1.0))
REWARD_STRUCTURE = os.environ.get("REWARD_STRUCTURE", "forward_gated")  # 步態 bonus 以前進為閘門：不前進拿不到
SMOOTH_WEIGHT   = float(os.environ.get("SMOOTH_WEIGHT", 0.02)) # 溫和
TILT_WEIGHT     = float(os.environ.get("TILT_WEIGHT", 0.2))    # 溫和
# ─────────────────────────────────────────────────────────────────────────────


def make_env(seed: int = 0, render_mode: str | None = None) -> gym.Env:
    env = gym.make(
        ENV_NAME,
        render_mode=render_mode,
        healthy_reward=1.0,
        forward_reward_weight=1.0,
        ctrl_cost_weight=0.5,
        contact_cost_weight=5e-4,
    )
    env = RealisticGaitWrapper(
        env,
        target_speed=TARGET_SPEED,
        ctrl_weight=CTRL_WEIGHT,
        gait_weight=GAIT_WEIGHT,
        posture_weight=POSTURE_WEIGHT,
        alive_weight=ALIVE_WEIGHT,
        gait_mode=GAIT_MODE,
        forward_mode=FORWARD_MODE,
        forward_weight=FORWARD_WEIGHT,
        smooth_weight=SMOOTH_WEIGHT,
        tilt_weight=TILT_WEIGHT,
        reward_structure=REWARD_STRUCTURE,
    )
    env.reset(seed=seed)
    return env


class GaitMonitorCallback(BaseCallback):
    """記錄 wrapper 的 reward 分解與步態指標到 TensorBoard（含 04 新增的 smooth/tilt/anti_phase）。"""

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "reward_components" in info:
                comp = info["reward_components"]
                self.logger.record_mean("gait/r_forward", comp["forward"])
                self.logger.record_mean("gait/r_alive", comp["alive"])
                self.logger.record_mean("gait/r_ctrl", comp["ctrl"])
                self.logger.record_mean("gait/r_gait", comp["gait"])
                self.logger.record_mean("gait/r_posture", comp["posture"])
                self.logger.record_mean("gait/r_smooth", comp["smooth"])
                self.logger.record_mean("gait/r_tilt", comp["tilt"])
                self.logger.record_mean("gait/x_velocity", comp["x_velocity"])
                self.logger.record_mean("gait/torso_z", comp["torso_z"])
                self.logger.record_mean("gait/uprightness", comp["uprightness"])
                self.logger.record_mean("gait/anti_phase", comp["anti_phase"])

            if "foot_contacts" in info:
                contacts = info["foot_contacts"]
                self.logger.record_mean("contacts/FL", float(contacts[0]))
                self.logger.record_mean("contacts/FR", float(contacts[1]))
                self.logger.record_mean("contacts/BL", float(contacts[2]))
                self.logger.record_mean("contacts/BR", float(contacts[3]))
                self.logger.record_mean("contacts/diagonal_sync",
                                        gait_metrics.diagonal_sync(contacts))
        return True


class EvalScorecardCallback(BaseCallback):
    """每隔 eval_interval steps 在獨立環境跑一個 deterministic episode，
    錄影並計算整段步態 scorecard（速度誤差、CoT、jerk、anti_phase、週期性）寫進 TensorBoard。

    這是「量化模型好壞」的乾淨訊號：deterministic（無探索噪聲）、per-episode，
    比訓練中帶 noise 的 record_mean 更能反映模型實際表現。
    """

    def __init__(self, eval_interval: int = 50_000, video_interval: int = 200_000,
                 video_root: str = f"{OUTPUT_DIR}/videos", verbose: int = 0):
        super().__init__(verbose)
        self.eval_interval = eval_interval
        self.video_interval = video_interval
        self.video_root = video_root
        self._last_eval_block = 0
        self._last_video_block = 0

    def _on_step(self) -> bool:
        # 以 num_timesteps 計（多環境時每步會跳 n_envs），確保每 eval_interval 步觸發一次。
        # 數值 scorecard 每 eval_interval 跑（便宜）；錄影只每 video_interval 跑（moviepy 編碼貴）。
        block = self.num_timesteps // self.eval_interval
        if block > self._last_eval_block:
            self._last_eval_block = block
            video_block = self.num_timesteps // self.video_interval
            record_video = video_block > self._last_video_block
            if record_video:
                self._last_video_block = video_block
            self._record_episode(step=self.num_timesteps, record_video=record_video)
        return True

    def _record_episode(self, step: int, record_video: bool = True) -> None:
        eval_env = make_env(seed=SEED + 1, render_mode="rgb_array" if record_video else None)
        if record_video:
            eval_env = RecordVideo(
                eval_env,
                video_folder=f"{self.video_root}/step_{step:07d}",
                name_prefix=f"eval_step_{step}",
                episode_trigger=lambda ep: True,
            )
        dt = eval_env.unwrapped.dt  # Ant-v5 ≈ 0.05s，用來把 x_velocity 積分成距離

        obs, _ = eval_env.reset()
        total_reward, ep_len = 0.0, 0
        actions, contacts_seq, x_vels, uprights = [], [], [], []
        done = False

        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            total_reward += reward
            ep_len += 1
            actions.append(np.asarray(action, dtype=np.float64))
            contacts_seq.append(info["foot_contacts"])
            x_vels.append(info["reward_components"]["x_velocity"])
            uprights.append(info["reward_components"]["uprightness"])
            done = terminated or truncated

        eval_env.close()

        actions = np.asarray(actions)
        contacts_seq = np.asarray(contacts_seq)
        x_vels = np.asarray(x_vels)
        distance = float(np.sum(x_vels) * dt)

        # ── scorecard：每一項都是「步態好壞」的正交量化軸 ──
        self.logger.record("eval/episode_return", total_reward)
        self.logger.record("eval/episode_length", ep_len)
        self.logger.record("eval/speed_error", float(np.mean(np.abs(x_vels - TARGET_SPEED))))
        self.logger.record("eval/distance", distance)
        self.logger.record("eval/action_jerk", gait_metrics.action_jerk(actions))
        self.logger.record("eval/transport_cost", gait_metrics.transport_cost(actions, distance))
        self.logger.record("eval/contact_regularity", gait_metrics.contact_regularity(contacts_seq))
        self.logger.record("eval/diagonal_sync",
                           float(np.mean([gait_metrics.diagonal_sync(c) for c in contacts_seq])))
        self.logger.record("eval/anti_phase",
                           float(np.mean([gait_metrics.anti_phase(c) for c in contacts_seq])))
        self.logger.record("eval/uprightness", float(np.mean(uprights)))

        if self.verbose:
            print(f"[Step {step}] return={total_reward:.1f} len={ep_len} "
                  f"speed_err={np.mean(np.abs(x_vels - TARGET_SPEED)):.3f} "
                  f"regularity={gait_metrics.contact_regularity(contacts_seq):.3f}")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 多環境用 SubprocVecEnv（各 env 跑在獨立 process、不受 GIL 限制，並行收集經驗）；
    # 單環境退回 DummyVecEnv，行為與 03 一致。
    if N_ENVS > 1:
        # partial（可 pickle）+ fork（此處尚未初始化 CUDA/GL，fork 安全；避免 spawn 重匯入 "04..." 檔名問題）
        train_env = SubprocVecEnv(
            [partial(make_env, seed=SEED + i) for i in range(N_ENVS)],
            start_method="fork",
        )
    else:
        train_env = DummyVecEnv([lambda: make_env(seed=SEED)])

    n_actions = train_env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    # 維持「每收集 1 筆 transition 做 1 次梯度更新」的 1:1 比例（與單環境 TD3 預設相同），
    # 確保總梯度更新數 ≈ 總步數 = MAX_TIMESTEPS，資料利用率不變。
    # 多環境必須用 step 為單位的 train_freq（SB3 不允許多環境 episodic 訓練）。
    if N_ENVS > 1:
        freq_kwargs = dict(train_freq=(1, "step"), gradient_steps=N_ENVS)
    else:
        freq_kwargs = dict()  # 沿用 TD3 預設 train_freq=(1,"episode"), gradient_steps=-1（亦為 1:1）

    model = TD3(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        policy_delay=2,
        target_policy_noise=0.2,
        target_noise_clip=0.5,
        action_noise=action_noise,
        learning_starts=LEARNING_STARTS,
        tensorboard_log=f"{OUTPUT_DIR}/tb",
        verbose=1,
        seed=SEED,
        **freq_kwargs,
    )

    # CheckpointCallback 的 save_freq 以「vec-step」計，多環境要除以 N_ENVS 才是每 CHECKPOINT_FREQ 步
    checkpoint_cb = CheckpointCallback(
        save_freq=max(CHECKPOINT_FREQ // N_ENVS, 1),
        save_path=f"{OUTPUT_DIR}/checkpoints",
        name_prefix="td3_gait",
    )

    model.learn(
        total_timesteps=MAX_TIMESTEPS,
        callback=[
            GaitMonitorCallback(),
            EvalScorecardCallback(eval_interval=EVAL_INTERVAL, video_interval=VIDEO_INTERVAL, verbose=1),
            checkpoint_cb,
        ],
        progress_bar=True,
    )

    model.save(f"{OUTPUT_DIR}/final_model")
    print("Training complete.")
