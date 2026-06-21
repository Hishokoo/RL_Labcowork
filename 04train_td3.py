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
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from tools.gait_wrapper import RealisticGaitWrapper
from tools import gait_metrics

# ── Hyperparameters ───────────────────────────────────────────────────────────
ENV_NAME        = "Ant-v5"
SEED            = 0
MAX_TIMESTEPS   = 1_000_000
LEARNING_STARTS = 10_000
EVAL_INTERVAL   = 50_000      # 中途錄影 + scorecard 間隔
CHECKPOINT_FREQ = 100_000     # 中途存檔間隔（03 沒有，這次補上以便看指標演進）
OUTPUT_DIR      = "output/04train_td3"

# RealisticGaitWrapper 參數（相對 03 的調整見檔頭）
TARGET_SPEED    = 1.0
CTRL_WEIGHT     = 5.0
GAIT_WEIGHT     = 2.0
POSTURE_WEIGHT  = 2.0
ALIVE_WEIGHT    = 0.0         # 03 為 1.0；歸零以徹底消除「站著的收入」（見 ant_v5_attractor_fix.md）。
                              # 摔倒由 terminate_when_unhealthy 結束 episode 提供「別摔」訊號，不需 alive 底分
GAIT_MODE       = "antiphase"
FORWARD_MODE    = "progress"
SMOOTH_WEIGHT   = 0.1         # jerk 懲罰起始值，視 gait/r_smooth 量級再調
TILT_WEIGHT     = 0.5         # 軀幹直立懲罰起始值
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
        smooth_weight=SMOOTH_WEIGHT,
        tilt_weight=TILT_WEIGHT,
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

    def __init__(self, eval_interval: int = 50_000, video_root: str = f"{OUTPUT_DIR}/videos", verbose: int = 0):
        super().__init__(verbose)
        self.eval_interval = eval_interval
        self.video_root = video_root

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_interval == 0:
            self._record_episode(step=self.num_timesteps)
        return True

    def _record_episode(self, step: int) -> None:
        eval_env = make_env(seed=SEED + 1, render_mode="rgb_array")
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

    train_env = DummyVecEnv([lambda: make_env(seed=SEED)])

    n_actions = train_env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

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
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=f"{OUTPUT_DIR}/checkpoints",
        name_prefix="td3_gait",
    )

    model.learn(
        total_timesteps=MAX_TIMESTEPS,
        callback=[
            GaitMonitorCallback(),
            EvalScorecardCallback(eval_interval=EVAL_INTERVAL, verbose=1),
            checkpoint_cb,
        ],
        progress_bar=True,
    )

    model.save(f"{OUTPUT_DIR}/final_model")
    print("Training complete.")
