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
OUTPUT_DIR        = "output/02train_td3"

# Reward shaping（詳見 markdown/ant_v5_attractor_fix.md）
# 根本問題：Ant-v5 預設 healthy_reward=1.0 讓「站著不動」變成零風險的 attractor，
# 不管怎麼調 forward_reward_weight / stillness penalty 都會被吸進去（5 次實驗驗證）。
# 解法：拿掉 healthy_reward（站著不再有收入）+ 提高 contact_cost_weight（直接懲罰重踩，改善跑姿）+
# 還原 forward_reward_weight=1.0（不再用降低速度誘因的方式抑制衝刺）。
FORWARD_REWARD_WEIGHT = 1.0     # 還原 Ant-v5 預設值，不再壓速度誘因
HEALTHY_REWARD        = 0.1     # 留一點安全網（0.0 時 agent 學成「衝一下就摔倒重來」，episode 卡在 8~12 步，見 markdown/ant_v5_attractor_fix.md 風險1）
CONTACT_COST_WEIGHT   = 5e-3    # 預設 5e-4 太小幾乎沒作用，提高 10 倍直接懲罰重踩蹬地
SOFT_RADIUS           = 6.0     # 超過此距離開始給漸增邊界懲罰（事前梯度，避免到了才知道）
MAX_RADIUS            = 8.0     # 離原點距離超過此值強制截斷 episode
BOUNDARY_PENALTY      = 5.0     # 邊界懲罰係數（相對 SOFT_RADIUS 漸增）
ACTION_PENALTY_WEIGHT = 0.5     # 動作幅度懲罰係數（避免暴力蹬地）
ACTION_DIFF_PENALTY_WEIGHT = 0.1  # 相鄰動作差異懲罰係數（避免抖動）
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Environment ───────────────────────────────────────────────────────────────
env      = gym.make(ENV_NAME, forward_reward_weight=FORWARD_REWARD_WEIGHT,
                     healthy_reward=HEALTHY_REWARD, contact_cost_weight=CONTACT_COST_WEIGHT)
eval_env = gym.make(ENV_NAME, forward_reward_weight=FORWARD_REWARD_WEIGHT,
                     healthy_reward=HEALTHY_REWARD, contact_cost_weight=CONTACT_COST_WEIGHT)
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
prev_action      = np.zeros(act_dim)

for t in range(1, MAX_TIMESTEPS + 1):
    episode_steps += 1

    # Collect action
    if t < START_TIMESTEPS:
        action = env.action_space.sample()
    else:
        noise  = np.random.normal(0, max_action * EXPLORATION_NOISE, size=act_dim)
        action = (agent.select_action(obs) + noise).clip(-max_action, max_action)

    next_obs, reward, terminated, truncated, info = env.step(action)

    # 邊界位置懲罰：超過 SOFT_RADIUS 後漸增懲罰（事前梯度），超過 MAX_RADIUS 強制截斷
    dist = info["distance_from_origin"]
    if dist > SOFT_RADIUS:
        reward -= BOUNDARY_PENALTY * (dist - SOFT_RADIUS)
    boundary_violation = dist > MAX_RADIUS
    if boundary_violation:
        truncated = True

    # 動作正則化：幅度懲罰 + 相鄰動作差異懲罰（contact_cost 已內建於 Ant-v5 reward，不重複懲罰）
    reward -= ACTION_PENALTY_WEIGHT * np.sum(np.square(action))
    reward -= ACTION_DIFF_PENALTY_WEIGHT * np.sum(np.square(action - prev_action))
    prev_action = action.copy()

    done = terminated or truncated

    # Store transition：出界視為 terminal（同 env 原生 terminated），讓 critic 學到「出界=沒有未來」
    # 一般 timeout truncation 才不算 terminal，這裡是我們自訂的懲罰性截斷，必須讓 bootstrap 停止
    replay_buffer.add(obs, action, next_obs, reward, float(terminated or boundary_violation))
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
        prev_action    = np.zeros(act_dim)

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
