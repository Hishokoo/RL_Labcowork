# 執行：/opt/miniconda3/envs/rl_env02/bin/python 02test_td3.py
import argparse
import torch
import gymnasium as gym
from tools.td3_agent import TD3Agent

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str,
                    default="output/02train_td3/td3_ant_step1000000.pt",
                    help="載入的模型路徑")
parser.add_argument("--episodes", type=int, default=5, help="跑幾個 episode")
args = parser.parse_args()

device = torch.device("cpu")

env = gym.make("Ant-v5", render_mode="human")
obs_dim    = env.observation_space.shape[0]
act_dim    = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

agent = TD3Agent(obs_dim, act_dim, max_action, device)
agent.load(args.checkpoint)
print(f"載入模型：{args.checkpoint}")

for ep in range(1, args.episodes + 1):
    obs, _ = env.reset()
    done = False
    total_reward = 0.0
    while not done:
        action = agent.select_action(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated
    print(f"Episode {ep}  reward={total_reward:.1f}")

env.close()
