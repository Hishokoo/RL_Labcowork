"""載入訓練好的 SB3 模型，跑 N 個 deterministic episode，用同一套步態指標算 scorecard（mean±std）。

用途：客觀比較不同訓練（如 03 vs 04）的步態品質。指標由 contacts / velocity / action 計算，
與訓練時用哪種 reward 無關，所以可公平比較不同 reward 訓出的 policy。

用法：python -m tools.eval_scorecard <name> <model_path> [n_episodes]
輸出：一行 JSON {name: {metric: {mean, std}, ...}}
"""
import sys
import json

import numpy as np
import gymnasium as gym
from stable_baselines3 import TD3

from tools.gait_wrapper_03 import RealisticGaitWrapper
from tools import gait_metrics

TARGET_SPEED = 1.0


def make_eval_env(seed: int = 0):
    env = gym.make("Ant-v5", healthy_reward=1.0, forward_reward_weight=1.0,
                   ctrl_cost_weight=0.5, contact_cost_weight=5e-4)
    # wrapper 的 reward 設定不影響 scorecard 指標（指標只看 contacts/velocity）；
    # 這裡只是要 wrapper 把 foot_contacts / x_velocity / uprightness 放進 info。
    env = RealisticGaitWrapper(env, gait_mode="antiphase_gated", reward_structure="forward_gated")
    env.reset(seed=seed)
    return env


def evaluate(model_path: str, n_episodes: int = 10, seed: int = 0) -> dict:
    model = TD3.load(model_path, device="cpu")
    env = make_eval_env(seed=seed)
    dt = env.unwrapped.dt
    rows = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + 100 + ep)
        actions, contacts, x_vels, uprights = [], [], [], []
        ret, done = 0.0, False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ret += reward
            actions.append(np.asarray(action, dtype=np.float64))
            contacts.append(info["foot_contacts"])
            x_vels.append(info["reward_components"]["x_velocity"])
            uprights.append(info["reward_components"]["uprightness"])
            done = terminated or truncated
        actions = np.asarray(actions)
        contacts = np.asarray(contacts)
        x_vels = np.asarray(x_vels)
        dist = float(np.sum(x_vels) * dt)
        rows.append(dict(
            episode_length=float(len(actions)),
            mean_speed=float(np.mean(x_vels)),
            speed_error=float(np.mean(np.abs(x_vels - TARGET_SPEED))),
            distance=dist,
            contact_regularity=gait_metrics.contact_regularity(contacts),
            action_jerk=gait_metrics.action_jerk(actions),
            transport_cost=gait_metrics.transport_cost(actions, dist),
            anti_phase=float(np.mean([gait_metrics.anti_phase(c) for c in contacts])),
            diagonal_sync=float(np.mean([gait_metrics.diagonal_sync(c) for c in contacts])),
            uprightness=float(np.mean(uprights)),
            stationary_fraction=gait_metrics.stationary_fraction(x_vels),
        ))
    env.close()
    keys = list(rows[0].keys())
    return {k: dict(mean=float(np.mean([r[k] for r in rows])),
                    std=float(np.std([r[k] for r in rows]))) for k in keys}


if __name__ == "__main__":
    name, path = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    print(json.dumps({name: evaluate(path, n_episodes=n)}))
