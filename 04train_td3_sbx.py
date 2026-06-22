# SBX(SB3 + Jax)版的 04 訓練腳本 —— 測試 Jax 後端能否在 pc106(aarch64 + Blackwell GB10)跑、
# 以及是否比 SB3 快。SB3 的訓練迴圈是單執行緒 Python(這台卡在 ~200 fps);SBX 用同樣的 TD3
# 演算法、同樣的 API,但底層換成 Jax,會 JIT 整個訓練迴圈,理論上 GPU 上能快 5–10 倍。
#
# 與 04train_td3.py 共用同一個 RealisticGaitWrapper + gait_metrics(reward 與指標定義完全一致),
# 只把後端 stable_baselines3.TD3 → sbx.TD3。監控先精簡(訓練 + 結尾跑一次 deterministic scorecard);
# 若 SBX 證明值得,再把 04 的完整 callbacks 接上。
#
# 執行(先用 CPU jax 確認跑得起來；GPU jax 待 Blackwell 支援確認後再切)：
#   cd ~/RL_Labcowork && MUJOCO_GL=egl MAX_TIMESTEPS=20000 ~/rlap_env/bin/python 04train_td3_sbx.py
import os
import time

import numpy as np
import gymnasium as gym
from sbx import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv

from tools.gait_wrapper_03 import RealisticGaitWrapper
from tools import gait_metrics

ENV_NAME        = "Ant-v5"
SEED            = 0
MAX_TIMESTEPS   = int(os.environ.get("MAX_TIMESTEPS", 1_000_000))
LEARNING_STARTS = 10_000
OUTPUT_DIR      = "output/04train_td3_sbx"
TARGET_SPEED    = 1.0

# wrapper 設定與 04train_td3.py 完全相同（antiphase 步態 + progress 速度 + jerk/直立懲罰）
WRAP_KWARGS = dict(
    target_speed=1.0, ctrl_weight=5.0, gait_weight=2.0, posture_weight=2.0,
    alive_weight=0.0, gait_mode="antiphase", forward_mode="progress",
    smooth_weight=0.1, tilt_weight=0.5,
)


def make_env(seed: int = 0, render_mode: str | None = None) -> gym.Env:
    env = gym.make(
        ENV_NAME, render_mode=render_mode, healthy_reward=1.0,
        forward_reward_weight=1.0, ctrl_cost_weight=0.5, contact_cost_weight=5e-4,
    )
    env = RealisticGaitWrapper(env, **WRAP_KWARGS)
    env.reset(seed=seed)
    return env


def run_scorecard(model, n_episodes: int = 2) -> list[dict]:
    """跑 deterministic episode，用共用的 gait_metrics 算 scorecard（與 04 同定義）。"""
    env = make_env(seed=SEED + 1)
    dt = env.unwrapped.dt
    rows = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        actions, contacts, x_vels, uprights = [], [], [], []
        total_reward, done = 0.0, False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            actions.append(np.asarray(action, dtype=np.float64))
            contacts.append(info["foot_contacts"])
            x_vels.append(info["reward_components"]["x_velocity"])
            uprights.append(info["reward_components"]["uprightness"])
            done = terminated or truncated
        actions, contacts, x_vels = np.asarray(actions), np.asarray(contacts), np.asarray(x_vels)
        distance = float(np.sum(x_vels) * dt)
        rows.append(dict(
            ret=round(total_reward, 1),
            speed_err=round(float(np.mean(np.abs(x_vels - TARGET_SPEED))), 3),
            jerk=round(gait_metrics.action_jerk(actions), 4),
            cot=round(gait_metrics.transport_cost(actions, distance), 3),
            regularity=round(gait_metrics.contact_regularity(contacts), 3),
            anti_phase=round(float(np.mean([gait_metrics.anti_phase(c) for c in contacts])), 3),
            uprightness=round(float(np.mean(uprights)), 3),
        ))
    env.close()
    return rows


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

    t0 = time.time()
    model.learn(total_timesteps=MAX_TIMESTEPS, progress_bar=True)
    elapsed = time.time() - t0

    model.save(f"{OUTPUT_DIR}/final_model")
    print(f"[SBX] {MAX_TIMESTEPS} steps in {elapsed:.1f}s = {MAX_TIMESTEPS / elapsed:.1f} fps")
    for i, row in enumerate(run_scorecard(model), 1):
        print(f"[SBX] eval ep{i}: {row}")
    print("SBX_DONE")
