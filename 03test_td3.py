# 執行：/opt/miniconda3/envs/rl_env02/bin/python 03test_td3.py
import argparse
import gymnasium as gym
from stable_baselines3 import TD3

from tools.gait_wrapper_03 import RealisticGaitWrapper

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str,
                    default="output/03train_td3/final_model",
                    help="載入的模型路徑（不含 .zip）")
parser.add_argument("--episodes", type=int, default=5, help="跑幾個 episode")
args = parser.parse_args()

env = gym.make(
    "Ant-v5",
    render_mode="human",
    healthy_reward=1.0,
    forward_reward_weight=1.0,
    ctrl_cost_weight=0.5,
    contact_cost_weight=5e-4,
)
env = RealisticGaitWrapper(env)

model = TD3.load(args.checkpoint)
print(f"載入模型：{args.checkpoint}")

for ep in range(1, args.episodes + 1):
    obs, _ = env.reset()
    done = False
    total_reward = 0.0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated
    print(f"Episode {ep}  reward={total_reward:.1f}")

env.close()
