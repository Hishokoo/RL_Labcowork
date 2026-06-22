# 執行前請安裝：pip install "gymnasium[mujoco]" stable-baselines3 moviepy tqdm rich
# 規格來源：markdown/03_train_td3_spec.md
#
# 目標：訓練出接近真實動物的步態（四足交替著地、動作平滑），而非預設 reward 訓出的「慣性甩動」高速移動。
# 最終 reward 數值會比 benchmark（3000+）低，預期落在 800-1500 區間，這是設計上的取捨（見 spec 風險警示段落）。
import os
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from tools.gait_wrapper_03 import RealisticGaitWrapper

# ── Hyperparameters ───────────────────────────────────────────────────────────
ENV_NAME        = "Ant-v5"
SEED            = 0
MAX_TIMESTEPS   = 1_000_000
LEARNING_STARTS = 10_000
EVAL_INTERVAL   = 50_000     # 中途錄影間隔
OUTPUT_DIR      = "output/03train_td3"

# RealisticGaitWrapper 參數
TARGET_SPEED    = 1.0
CTRL_WEIGHT     = 5.0
GAIT_WEIGHT     = 2.0
POSTURE_WEIGHT  = 2.0
ALIVE_WEIGHT    = 1.0
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
    )
    env.reset(seed=seed)
    return env


class GaitMonitorCallback(BaseCallback):
    """記錄 wrapper 的 reward 分解與步態指標（對角線同步）到 TensorBoard。"""

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
                self.logger.record_mean("gait/x_velocity", comp["x_velocity"])
                self.logger.record_mean("gait/torso_z", comp["torso_z"])

            if "foot_contacts" in info:
                contacts = info["foot_contacts"]
                self.logger.record_mean("contacts/FL", float(contacts[0]))
                self.logger.record_mean("contacts/FR", float(contacts[1]))
                self.logger.record_mean("contacts/BL", float(contacts[2]))
                self.logger.record_mean("contacts/BR", float(contacts[3]))
                diag_sync = 1 - 0.5 * (
                    abs(contacts[0] - contacts[3]) + abs(contacts[1] - contacts[2])
                )
                self.logger.record_mean("contacts/diagonal_sync", float(diag_sync))
        return True


class VideoCallback(BaseCallback):
    """每隔 eval_interval steps 在獨立環境跑一個 evaluation episode 並錄影。"""

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

        obs, _ = eval_env.reset()
        total_reward, ep_len = 0.0, 0
        done = False

        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            total_reward += reward
            ep_len += 1
            done = terminated or truncated

        eval_env.close()
        self.logger.record("eval/episode_return", total_reward)
        self.logger.record("eval/episode_length", ep_len)
        if self.verbose:
            print(f"[Step {step}] Eval return: {total_reward:.1f}, length: {ep_len}")


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

    model.learn(
        total_timesteps=MAX_TIMESTEPS,
        callback=[GaitMonitorCallback(), VideoCallback(eval_interval=EVAL_INTERVAL)],
        progress_bar=True,
    )

    model.save(f"{OUTPUT_DIR}/final_model")
    print("Training complete.")
