"""共用的 TD3 步態訓練骨架。

各實驗（04/05…）只需定義自己的 reward 設定（wrap_kwargs）後呼叫 train()，
不必複製整支訓練程式（符合 CLAUDE.md：共用邏輯抽到 tools/）。
reward 旋鈕的定義在 tools/gait_wrapper_03.py；步態指標在 tools/gait_metrics.py。

執行控制（OUTPUT_DIR / MAX_TIMESTEPS / 各 interval / N_ENVS）由各實驗檔用 env var 傳入，
reward 設定則寫死在各實驗檔裡（每個實驗一份明確的 config）。
"""
import os
from functools import partial

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from tools.gait_wrapper_03 import RealisticGaitWrapper
from tools import gait_metrics

ENV_NAME = "Ant-v5"


def make_env(wrap_kwargs: dict, seed: int = 0, render_mode: str | None = None,
             wrapper_cls: type = RealisticGaitWrapper) -> gym.Env:
    env = gym.make(
        ENV_NAME, render_mode=render_mode, healthy_reward=1.0,
        forward_reward_weight=1.0, ctrl_cost_weight=0.5, contact_cost_weight=5e-4,
    )
    env = wrapper_cls(env, **wrap_kwargs)
    env.reset(seed=seed)
    return env


class GaitMonitorCallback(BaseCallback):
    """記錄 wrapper 的 reward 分解與步態指標到 TensorBoard。"""

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "reward_components" in info:
                c = info["reward_components"]
                for k in ("forward", "alive", "ctrl", "gait", "posture", "smooth", "tilt", "antiphase_bonus"):
                    if k in c:
                        self.logger.record_mean(f"gait/r_{k}", c[k])
                for k in ("x_velocity", "torso_z", "uprightness", "anti_phase"):
                    self.logger.record_mean(f"gait/{k}", c[k])
            if "foot_contacts" in info:
                ct = info["foot_contacts"]
                for i, name in enumerate(("FL", "FR", "BL", "BR")):
                    self.logger.record_mean(f"contacts/{name}", float(ct[i]))
                self.logger.record_mean("contacts/diagonal_sync", gait_metrics.diagonal_sync(ct))
        return True


class EvalScorecardCallback(BaseCallback):
    """每 eval_interval 步跑一個 deterministic episode 算整段步態 scorecard 寫進 TensorBoard；
    每 video_interval 步才額外錄影（moviepy 編碼貴，與數值 scorecard 解耦以加速）。"""

    def __init__(self, wrap_kwargs: dict, target_speed: float, output_dir: str,
                 eval_interval: int = 50_000, video_interval: int = 200_000,
                 seed: int = 0, verbose: int = 0, wrapper_cls: type = RealisticGaitWrapper):
        super().__init__(verbose)
        self.wrap_kwargs = wrap_kwargs
        self.target_speed = target_speed
        self.video_root = f"{output_dir}/videos"
        self.eval_interval = eval_interval
        self.video_interval = video_interval
        self.seed = seed
        self.wrapper_cls = wrapper_cls
        self._last_eval_block = 0
        self._last_video_block = 0

    def _on_step(self) -> bool:
        block = self.num_timesteps // self.eval_interval
        if block > self._last_eval_block:
            self._last_eval_block = block
            vblock = self.num_timesteps // self.video_interval
            record_video = vblock > self._last_video_block
            if record_video:
                self._last_video_block = vblock
            self._record_episode(self.num_timesteps, record_video)
        return True

    def _record_episode(self, step: int, record_video: bool) -> None:
        env = make_env(self.wrap_kwargs, seed=self.seed + 1,
                       render_mode="rgb_array" if record_video else None,
                       wrapper_cls=self.wrapper_cls)
        if record_video:
            env = RecordVideo(env, video_folder=f"{self.video_root}/step_{step:07d}",
                              name_prefix=f"eval_step_{step}", episode_trigger=lambda ep: True)
        dt = env.unwrapped.dt

        obs, _ = env.reset()
        total_reward, ep_len = 0.0, 0
        actions, contacts_seq, x_vels, uprights = [], [], [], []
        done = False
        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            ep_len += 1
            actions.append(np.asarray(action, dtype=np.float64))
            contacts_seq.append(info["foot_contacts"])
            x_vels.append(info["reward_components"]["x_velocity"])
            uprights.append(info["reward_components"]["uprightness"])
            done = terminated or truncated
        env.close()

        actions = np.asarray(actions)
        contacts_seq = np.asarray(contacts_seq)
        x_vels = np.asarray(x_vels)
        distance = float(np.sum(x_vels) * dt)

        rec = self.logger.record
        rec("eval/episode_return", total_reward)
        rec("eval/episode_length", ep_len)
        rec("eval/speed_error", float(np.mean(np.abs(x_vels - self.target_speed))))
        rec("eval/distance", distance)
        rec("eval/action_jerk", gait_metrics.action_jerk(actions))
        rec("eval/transport_cost", gait_metrics.transport_cost(actions, distance))
        rec("eval/contact_regularity", gait_metrics.contact_regularity(contacts_seq))
        rec("eval/diagonal_sync", float(np.mean([gait_metrics.diagonal_sync(c) for c in contacts_seq])))
        rec("eval/anti_phase", float(np.mean([gait_metrics.anti_phase(c) for c in contacts_seq])))
        rec("eval/uprightness", float(np.mean(uprights)))
        if self.verbose:
            print(f"[Step {step}] return={total_reward:.1f} len={ep_len} "
                  f"speed_err={np.mean(np.abs(x_vels - self.target_speed)):.3f} "
                  f"regularity={gait_metrics.contact_regularity(contacts_seq):.3f}")


def train(wrap_kwargs: dict, output_dir: str, *, target_speed: float = 1.0,
          max_timesteps: int = 1_000_000, learning_starts: int = 10_000,
          eval_interval: int = 50_000, video_interval: int = 200_000,
          checkpoint_freq: int = 100_000, n_envs: int = 1, seed: int = 0,
          wrapper_cls: type = RealisticGaitWrapper) -> None:
    os.makedirs(output_dir, exist_ok=True)

    if n_envs > 1:
        # 多環境用 SubprocVecEnv 並行收集；維持 1:1 梯度更新比例（總更新數不變）
        train_env = SubprocVecEnv(
            [partial(make_env, wrap_kwargs, seed=seed + i, wrapper_cls=wrapper_cls) for i in range(n_envs)],
            start_method="fork",
        )
        freq_kwargs = dict(train_freq=(1, "step"), gradient_steps=n_envs)
    else:
        train_env = DummyVecEnv([lambda: make_env(wrap_kwargs, seed=seed, wrapper_cls=wrapper_cls)])
        freq_kwargs = dict()  # 沿用 TD3 預設 (1,"episode"), -1

    n_actions = train_env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    model = TD3(
        "MlpPolicy", train_env, learning_rate=3e-4, buffer_size=1_000_000, batch_size=256,
        tau=0.005, gamma=0.99, policy_delay=2, target_policy_noise=0.2, target_noise_clip=0.5,
        action_noise=action_noise, learning_starts=learning_starts,
        tensorboard_log=f"{output_dir}/tb", verbose=1, seed=seed, **freq_kwargs,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=max(checkpoint_freq // n_envs, 1),
        save_path=f"{output_dir}/checkpoints", name_prefix="td3_gait",
    )

    model.learn(
        total_timesteps=max_timesteps,
        callback=[
            GaitMonitorCallback(),
            EvalScorecardCallback(wrap_kwargs, target_speed, output_dir,
                                  eval_interval, video_interval, seed, verbose=1,
                                  wrapper_cls=wrapper_cls),
            checkpoint_cb,
        ],
        progress_bar=True,
    )
    model.save(f"{output_dir}/final_model")
    model.save_replay_buffer(f"{output_dir}/replay_buffer")  # 存 buffer 以利日後續訓免重跑
    print("Training complete.")


def finetune(wrap_kwargs: dict, output_dir: str, init_model_path: str, *,
             target_speed: float = 1.0, max_timesteps: int = 120_000,
             learning_rate: float = 1e-4, action_noise_sigma: float = 0.03,
             eval_interval: int = 25_000, video_interval: int = 25_000,
             checkpoint_freq: int = 25_000, seed: int = 0,
             wrapper_cls: type = RealisticGaitWrapper) -> None:
    """從既有 checkpoint 接續微調（curriculum 用）：載入 actor/critic、換成新 reward、
    清空 replay buffer（不載入舊 buffer，避免混用新舊 reward）、不做 random warmup
    （learning_starts=0，第一個 episode 即用載入的策略收集），用較小 lr / action noise。

    與 train() 的差別僅在「初始化方式」——callback / 指標 / 錄影完全沿用，確保可比較。
    """
    os.makedirs(output_dir, exist_ok=True)
    train_env = DummyVecEnv([lambda: make_env(wrap_kwargs, seed=seed, wrapper_cls=wrapper_cls)])

    n_actions = train_env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions), sigma=action_noise_sigma * np.ones(n_actions))

    # 載入 03 的 actor/critic，覆寫 lr / action_noise / learning_starts；不載入舊 replay buffer
    # （TD3.load 預設不還原 buffer）→ buffer 全新清空，符合「避免混用新舊 reward」。
    model = TD3.load(
        init_model_path, env=train_env,
        custom_objects={
            "learning_rate": learning_rate,
            "action_noise": action_noise,
            "learning_starts": 0,          # 不做 random warmup：第一 episode 即用載入策略
        },
        tensorboard_log=f"{output_dir}/tb", verbose=1, seed=seed,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=max(checkpoint_freq, 1),
        save_path=f"{output_dir}/checkpoints", name_prefix="td3_gait_ft",
    )

    model.learn(
        total_timesteps=max_timesteps,
        callback=[
            GaitMonitorCallback(),
            EvalScorecardCallback(wrap_kwargs, target_speed, output_dir,
                                  eval_interval, video_interval, seed, verbose=1,
                                  wrapper_cls=wrapper_cls),
            checkpoint_cb,
        ],
        progress_bar=True,
        reset_num_timesteps=True,  # 微調的步數從 0 起算，eval interval 才對齊
    )
    model.save(f"{output_dir}/final_model")
    model.save_replay_buffer(f"{output_dir}/replay_buffer")  # 存 buffer 以利日後續訓免重跑
    print("Fine-tune complete.")


def resume(wrap_kwargs: dict, output_dir: str, init_model_path: str, replay_buffer_path: str, *,
           target_speed: float = 1.0, additional_timesteps: int = 600_000,
           learning_rate: float = 3e-4, eval_interval: int = 50_000,
           video_interval: int = 200_000, checkpoint_freq: int = 100_000, seed: int = 0) -> None:
    """載入既有 model + replay buffer，用**同一 reward**續訓更多步（免重跑）。

    與 finetune() 的差別：finetune 清空 buffer + 換 reward（curriculum）；
    resume 載入舊 buffer + 同 reward（單純訓更多步），等同「沒中斷地繼續訓練」。
    注意：若原訓練用 ctrl_schedule，resume 的 wrap_kwargs 應換成排程終點的固定 ctrl_weight
    （wrapper 的 _gstep 會從 0 起算，無法接續排程相位）。
    """
    os.makedirs(output_dir, exist_ok=True)
    train_env = DummyVecEnv([lambda: make_env(wrap_kwargs, seed=seed)])
    model = TD3.load(
        init_model_path, env=train_env,
        custom_objects={"learning_rate": learning_rate, "learning_starts": 0},
        tensorboard_log=f"{output_dir}/tb", verbose=1, seed=seed,
    )
    model.load_replay_buffer(replay_buffer_path)  # ★ 載入舊 buffer 接續，不重跑
    print(f"[resume] loaded replay buffer, size={model.replay_buffer.size()}")

    checkpoint_cb = CheckpointCallback(
        save_freq=max(checkpoint_freq, 1),
        save_path=f"{output_dir}/checkpoints", name_prefix="td3_gait",
    )
    model.learn(
        total_timesteps=additional_timesteps,
        callback=[
            GaitMonitorCallback(),
            EvalScorecardCallback(wrap_kwargs, target_speed, output_dir,
                                  eval_interval, video_interval, seed, verbose=1),
            checkpoint_cb,
        ],
        progress_bar=True,
        reset_num_timesteps=True,
    )
    model.save(f"{output_dir}/final_model")
    model.save_replay_buffer(f"{output_dir}/replay_buffer")
    print("Resume complete.")
