# RL 期末專題：TD3 + MuJoCo Ant-v4

## 專題概述

本專題使用 **TD3（Twin Delayed DDPG）** 演算法，訓練 **Gymnasium MuJoCo Ant-v4** 環境中的四足機器人完成連續控制任務。目標是產出穩定的訓練實作與分析報告。

- **Deadline：** 6 月 23 日（距今 5 天）
- **必交項目：** Demo 影片（6–8 分鐘）、六段書面報告、GitHub Repo、Google 表單

---

## 5 天時程規劃

| 天 | 重點工作 |
|----|---------|
| Day 1 | 環境安裝、用 SB3 跑通 TD3 baseline、確認訓練能啟動 |
| Day 2 | 正式訓練（1M steps 背景跑），邊跑邊記 Bug Log |
| Day 3 | 訓練收尾、截圖 reward 曲線、錄製 Ant 行走影片片段 |
| Day 4 | 寫六段書面報告、整理 GitHub README |
| Day 5 | 剪輯 7 分鐘 Demo 影片、填 Google 表單、交齊 |

> ⚠️ Day 1 就要讓訓練跑起來，1M steps 需要數小時，要盡早讓它在背景執行。

---

## 環境規格：Ant-v4

- **平台：** Gymnasium（OpenAI Gym 繼任者）
- **任務：** `Ant-v4`（MuJoCo 物理模擬）
- **觀測空間：** 27 維連續向量（關節角度、速度、接觸力等）
- **動作空間：** 8 維連續動作（各關節力矩，範圍 `[-1, 1]`）
- **Reward 組成：** `forward_reward + survive_reward − ctrl_cost − contact_cost`

### `info` 字典可取出的欄位

```python
info["reward_forward"]   # 前進速度獎勵
info["reward_survive"]   # 存活獎勵
info["reward_ctrl"]      # 控制懲罰（負值）
info["reward_contact"]   # 接觸懲罰（負值）
info["x_velocity"]       # 實際前進速度
```

### `done` 處理注意事項

Ant-v4 有兩種結束：
- `terminated=True`：真的倒下，bootstrap 應斷掉（`done=True`）
- `truncated=True`：跑滿步數上限，**不應**設 `done=True`，否則會低估 episode 末尾的 Q 值

```python
done = terminated  # 不是 terminated or truncated
```

---

## 演算法：TD3

TD3 是為連續動作空間設計的 off-policy actor-critic 演算法，在 DDPG 基礎上用三個機制解決 Q 值過估計問題。

### 核心機制

#### 1. Twin Critics（雙 Q-network）

同時訓練兩個獨立 critic，計算 target Q 時取最小值：

```python
# DDPG 做法
target_q = reward + gamma * critic_target(next_state, next_action)

# TD3 做法
target_q1 = critic1_target(next_state, next_action)
target_q2 = critic2_target(next_state, next_action)
target_q = reward + gamma * min(target_q1, target_q2)  # 取最小
```

Actor 只用 `critic1` 的輸出來更新梯度，不參與 min 運算。

#### 2. Delayed Policy Update（延遲更新）

Critic 每步更新，Actor 每 `policy_delay` 步才更新一次，讓 critic 先收斂：

```python
for step in range(total_steps):
    update_critics()                        # 每步都跑

    if step % policy_delay == 0:
        update_actor()
        update_target_networks()            # soft update 也在這裡
```

#### 3. Target Policy Smoothing（目標動作加噪）

計算 target Q 時在 next_action 上加 clipped noise，防止 Q-function 對尖峰動作過擬合：

```python
noise = torch.clamp(
    torch.randn_like(action) * noise_std,
    -noise_clip, noise_clip                 # std=0.2, clip=0.5
)
next_action = torch.clamp(
    actor_target(next_state) + noise,
    -1, 1                                   # Ant 動作範圍
)
```

### 主要 Hyperparameters（建議起點）

```python
actor_lr          = 3e-4
critic_lr         = 3e-4
batch_size        = 256
replay_buffer_size = 1_000_000
gamma             = 0.99
tau               = 0.005        # soft update 係數
policy_delay      = 2
noise_std         = 0.2          # target policy smoothing
noise_clip        = 0.5
exploration_noise = 0.1          # rollout 時加在 action 上的 Gaussian noise
warmup_steps      = 10_000       # 純隨機探索的前置步數
```

---

## 完整 Training Loop

```python
for step in range(total_steps):

    # ── 1. 收集資料 ──────────────────────────────────────
    if step < warmup_steps:
        action = env.action_space.sample()
    else:
        action = actor(state)
        action += np.random.normal(0, exploration_noise, size=action_dim)
        action = np.clip(action, -1, 1)

    next_state, reward, terminated, truncated, info = env.step(action)
    done = terminated          # 注意：truncated 不設 done=True
    replay_buffer.add(state, action, reward, next_state, done)
    state = next_state if not done else env.reset()[0]

    if step < warmup_steps:
        continue

    # ── 2. 從 buffer 採樣 ────────────────────────────────
    s, a, r, s_, d = replay_buffer.sample(batch_size)

    # ── 3. 計算 target Q（含 smoothing noise）────────────
    with torch.no_grad():
        noise = (torch.randn_like(a) * 0.2).clamp(-0.5, 0.5)
        a_ = (actor_target(s_) + noise).clamp(-1, 1)
        q1_t = critic1_target(s_, a_)
        q2_t = critic2_target(s_, a_)
        target_q = r + (1 - d) * gamma * torch.min(q1_t, q2_t)

    # ── 4. 更新兩個 Critic ───────────────────────────────
    q1 = critic1(s, a)
    q2 = critic2(s, a)
    loss_c1 = F.mse_loss(q1, target_q)
    loss_c2 = F.mse_loss(q2, target_q)
    critic1_optimizer.zero_grad(); loss_c1.backward(); critic1_optimizer.step()
    critic2_optimizer.zero_grad(); loss_c2.backward(); critic2_optimizer.step()

    # ── 5. 延遲更新 Actor ────────────────────────────────
    if step % policy_delay == 0:
        actor_loss = -critic1(s, actor(s)).mean()
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        # Soft update all target networks
        for p, p_t in zip(critic1.parameters(), critic1_target.parameters()):
            p_t.data.copy_(tau * p.data + (1 - tau) * p_t.data)
        for p, p_t in zip(critic2.parameters(), critic2_target.parameters()):
            p_t.data.copy_(tau * p.data + (1 - tau) * p_t.data)
        for p, p_t in zip(actor.parameters(), actor_target.parameters()):
            p_t.data.copy_(tau * p.data + (1 - tau) * p_t.data)

    # ── 6. TensorBoard 記錄（見下方章節）────────────────
```

---

## Replay Buffer 注意事項

- **大小建議 1M**：Ant 這種任務 early experience 很重要，buffer 太小會把好的經驗洗掉
- **obs normalization**：Ant 的 27 維 obs 各維度範圍差異大，建議做 running mean/std normalization（SB3 預設有開）

---

## TensorBoard 監控指標

### 訓練啟動

```python
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("runs/td3_ant")
```

### TD3 演算法內部

```python
# Critic
writer.add_scalar("train/q1_loss",      loss_c1.item(),          step)
writer.add_scalar("train/q2_loss",      loss_c2.item(),          step)
writer.add_scalar("train/q1_mean",      q1.mean().item(),        step)
writer.add_scalar("train/q2_mean",      q2.mean().item(),        step)
writer.add_scalar("train/q1_q2_diff",   (q1 - q2).abs().mean().item(), step)
writer.add_scalar("train/target_q_mean", target_q.mean().item(), step)

# Actor（只在 policy_delay 步更新時記錄）
writer.add_scalar("train/actor_loss",   actor_loss.item(),       step)
```

| 指標 | 健康的樣子 |
|------|----------|
| `q1_loss` / `q2_loss` | 隨時間下降並趨穩 |
| `q1_mean` / `q2_mean` | 穩定上升；暴漲代表過估計 |
| `q1_q2_diff` | 收斂後應該縮小 |
| `actor_loss` | 持續下降（負值越小越好） |

### 探索行為

```python
writer.add_scalar("explore/action_std",      actions_batch.std().item(),      step)
writer.add_scalar("explore/action_mean_abs", actions_batch.abs().mean().item(), step)
```

- `action_std` 太小 → actor 過度收斂，不再探索
- `action_std` 太大 → 訓練還很早期或發散中

### Ant 專屬 reward 分解

```python
writer.add_scalar("ant/forward_reward", info["reward_forward"], step)
writer.add_scalar("ant/survive_reward", info["reward_survive"], step)
writer.add_scalar("ant/ctrl_cost",      info["reward_ctrl"],    step)  # 負值
writer.add_scalar("ant/contact_cost",   info["reward_contact"], step)  # 負值
writer.add_scalar("ant/x_velocity",     info["x_velocity"],     step)
```

> 💡 **報告寫作提示：** 若 episode return 停滯但 `forward_reward` 上升、`ctrl_cost` 同時變大，代表 Ant 在用粗暴方式前進，尚未學到省力步態。這種細節能直接支撐 results 章節的分析。

### Replay Buffer 健康度

```python
writer.add_scalar("buffer/size",        len(replay_buffer),                step)
writer.add_scalar("buffer/reward_mean", replay_buffer.recent_reward_mean(), step)
```

### Gradient 監控（debug 用，選擇性）

```python
# 在 actor 更新後加入
actor_grad_norm = sum(
    p.grad.norm().item() ** 2
    for p in actor.parameters() if p.grad is not None
) ** 0.5
writer.add_scalar("grad/actor_norm", actor_grad_norm, step)
```

gradient norm 突然爆炸 → 學習率太大或 Q 值過估計的早期訊號。

---

## 訓練過程影片截取

每隔固定 steps 跑一次獨立 evaluation 並錄影，不干擾訓練環境：

```python
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

def evaluate_and_record(actor, step, writer, video_dir="videos/"):
    eval_env = gym.make("Ant-v4", render_mode="rgb_array")
    eval_env = RecordVideo(eval_env, video_folder=f"{video_dir}/step_{step}")

    obs, _ = eval_env.reset()
    total_reward, ep_len = 0, 0
    done = False

    while not done:
        with torch.no_grad():
            action = actor(torch.FloatTensor(obs)).numpy()
        obs, reward, terminated, truncated, _ = eval_env.step(action)
        total_reward += reward
        ep_len += 1
        done = terminated or truncated

    eval_env.close()
    writer.add_scalar("eval/episode_return", total_reward, step)
    writer.add_scalar("eval/episode_length", ep_len,       step)
    print(f"[Step {step}] Eval Return: {total_reward:.1f} | Length: {ep_len}")
    return total_reward

# 在訓練 loop 裡
eval_interval = 50_000
if step % eval_interval == 0:
    evaluate_and_record(actor, step, writer)
```

### 影片資料夾結構（預期輸出）

```
videos/
├── step_0/         # 剛開始，Ant 隨機亂動
├── step_50000/     # 開始有一點方向感
├── step_200000/    # 能走但步態不穩
├── step_500000/    # 步態成形
└── step_1000000/   # 收斂後的樣子
```

---

## 訓練規模參考

| 指標 | 數值 |
|------|------|
| 建議總步數 | 1M～3M steps |
| GPU 訓練時間（1M steps） | 約 2～4 小時 |
| 收斂 reward（Ant-v4） | 3000～6000+（視實作品質） |

---

## 工具與套件

```
gymnasium[mujoco]       # 環境
stable-baselines3       # TD3 參考實作（快速驗證用）
torch                   # 自行實作
tensorboard             # 訓練曲線記錄
```

> 建議先用 Stable-Baselines3 跑通 baseline 確認環境正常，再進行自行實作。

---

## 書面報告結構（六段）

1. **摘要 Abstract**（200 字）
2. **背景說明**：動機、DDPG 的問題、TD3 如何解決
3. **研究方法 Methodology**：三個核心機制詳細說明
4. **實驗結果 Results**：reward 曲線、reward 分解圖、影片截圖
5. **Bug Log**：記錄踩過的坑（安裝問題、超參數崩潰等）
6. **結論 Conclude**：步態演化分析、未來方向

### 報告可用的分析角度

- Ablation study：拿掉 twin critics → 觀察過估計；關閉 delayed update → 比較穩定性
- `forward_reward` vs `ctrl_cost` 的消長：說明步態從粗暴到省力的演化過程
- `q1_q2_diff` 隨時間縮小：直接佐證 twin critics 的收斂效果

---

## 參考資料

- Fujimoto et al., 2018 — [*Addressing Function Approximation Error in Actor-Critic Methods*（TD3 原始論文）](https://arxiv.org/abs/1802.09477)
- [Gymnasium MuJoCo Ant-v4 文件](https://gymnasium.farama.org/environments/mujoco/ant/)
- [Stable-Baselines3 TD3](https://stable-baselines3.readthedocs.io/en/master/modules/td3.html)
