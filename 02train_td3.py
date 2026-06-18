# 執行前請安裝：pip install "gymnasium[mujoco]" torch tensorboard
import os
import numpy as np
import torch
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter

from tools.replay_buffer import ReplayBuffer
from tools.td3_agent import TD3Agent

# ── Hyperparameters ───────────────────────────────────────────────────────────
ENV_NAME          = "Ant-v5"
SEED              = 42
MAX_TIMESTEPS     = 1_000_000
START_TIMESTEPS   = 25_000      # 隨機動作暖機步數
EVAL_FREQ         = 5_000       # 每隔幾步跑一次 evaluation
SAVE_FREQ         = 100_000     # 每隔幾步存一次 checkpoint
BATCH_SIZE        = 256
EXPLORATION_NOISE = 0.1         # rollout 時加入的 Gaussian noise 標準差（相對 max_action）
OUTPUT_DIR        = "output/td3_ant_v5"
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Environment ───────────────────────────────────────────────────────────────
env      = gym.make(ENV_NAME)
eval_env = gym.make(ENV_NAME)
env.reset(seed=SEED)
eval_env.reset(seed=SEED + 1)
torch.manual_seed(SEED)
np.random.seed(SEED)

obs_dim    = env.observation_space.shape[0]
act_dim    = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
print(f"Env: {ENV_NAME}  obs={obs_dim}  act={act_dim}  max_action={max_action}")

# ── Agent & Buffer ────────────────────────────────────────────────────────────
agent         = TD3Agent(obs_dim, act_dim, max_action, device)
replay_buffer = ReplayBuffer(obs_dim, act_dim)
writer        = SummaryWriter(log_dir=f"{OUTPUT_DIR}/tb")


def evaluate(n_episodes: int = 5) -> float:
    total = 0.0
    for _ in range(n_episodes):
        obs, _ = eval_env.reset()
        done = False
        while not done:
            action = agent.select_action(obs)
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            total += reward
            done = terminated or truncated
    return total / n_episodes


# ── Training loop ─────────────────────────────────────────────────────────────
obs, _           = env.reset()
episode_reward   = 0.0
episode_steps    = 0
episode_num      = 0

for t in range(1, MAX_TIMESTEPS + 1):
    episode_steps += 1

    # Collect action
    if t < START_TIMESTEPS:
        action = env.action_space.sample()
    else:
        noise  = np.random.normal(0, max_action * EXPLORATION_NOISE, size=act_dim)
        action = (agent.select_action(obs) + noise).clip(-max_action, max_action)

    next_obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

    # Store transition（done 用 terminated，不含 timeout truncation）
    replay_buffer.add(obs, action, next_obs, reward, float(terminated))
    obs            = next_obs
    episode_reward += reward

    # Train
    if t >= START_TIMESTEPS:
        critic_loss, actor_loss = agent.train(replay_buffer, BATCH_SIZE)
        writer.add_scalar("Loss/critic", critic_loss, t)
        if actor_loss is not None:
            writer.add_scalar("Loss/actor", actor_loss, t)

    # Episode end
    if done:
        print(f"[T={t:>7}] ep={episode_num+1:>4}  steps={episode_steps:>4}  reward={episode_reward:.1f}")
        writer.add_scalar("Train/reward", episode_reward, t)
        obs, _         = env.reset()
        episode_reward = 0.0
        episode_steps  = 0
        episode_num   += 1

    # Evaluation
    if t % EVAL_FREQ == 0:
        eval_reward = evaluate()
        print(f"  >>> Eval T={t}  avg_reward={eval_reward:.1f}")
        writer.add_scalar("Eval/reward", eval_reward, t)

    # Checkpoint
    if t % SAVE_FREQ == 0:
        ckpt_path = f"{OUTPUT_DIR}/td3_ant_step{t}.pt"
        agent.save(ckpt_path)
        print(f"  >>> Checkpoint saved: {ckpt_path}")

env.close()
eval_env.close()
writer.close()
print("Training complete.")
