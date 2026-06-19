# Ant-v4 Reward 修改任務

## 目標

在現有的 TD3 訓練程式碼中加入兩項修改：
1. **邊界位置懲罰**：防止 ant 跑出可觀察範圍
2. **動作正則化**：讓跑姿更穩健、不像慣性蹬地

---

## 修改一：邊界位置懲罰

在 `env.step()` 之後，從 observation 取出 ant 的 x, y 座標，計算距原點距離，超過閾值時給予線性懲罰。

**邏輯說明：**
- Ant-v4 的 observation 前兩個維度是 `x`, `y` 座標
- 在半徑內不影響原本 reward，只在超出後才懲罰
- 使用 `truncated=True` 而非 `terminated=True`，避免影響 Q 值 bootstrap

```python
# 在 env.step(action) 之後加入

x, y = next_obs[0], next_obs[1]
dist = np.sqrt(x**2 + y**2)

MAX_RADIUS = 8.0          # 可調整，對應棋盤格格數
BOUNDARY_PENALTY = 1.0    # 懲罰強度，可調整

if dist > MAX_RADIUS:
    boundary_penalty = -BOUNDARY_PENALTY * (dist - MAX_RADIUS)
    reward += boundary_penalty
    truncated = True      # 超出邊界直接截斷 episode
```

---

## 修改二：動作正則化

在每次 `env.step()` 之後，根據當前 action 計算三項懲罰並加入 reward。

**需要額外追蹤 `prev_action`（上一步的動作），在 episode reset 時一起清零。**

```python
# 在訓練 loop 的 reset 處初始化
prev_action = np.zeros(env.action_space.shape)

# 在 env.step(action) 之後加入

# 1. 動作幅度懲罰（避免暴力蹬地）
action_penalty = -0.5 * np.sum(np.square(action))

# 2. 相鄰動作差異懲罰（避免抖動）
action_diff_penalty = -0.1 * np.sum(np.square(action - prev_action))

# 3. 接觸力懲罰（讓腳輕落地）
#    Ant-v4 的 info dict 裡有 'contact_cost'，直接取用
contact_penalty = -0.5 * info.get("contact_cost", 0.0)

reward += action_penalty + action_diff_penalty + contact_penalty

# 更新 prev_action
prev_action = action.copy()
```

> **注意**：若已在 `gym.make` 設定了 `ctrl_cost_weight` 或 `contact_cost_weight`，
> 請確認不要重複懲罰同一項目，避免過度抑制動作。

---

## 可選：直接在 gym.make 調整環境參數

如果不想動 reward 計算邏輯，也可以直接在環境初始化時加大懲罰係數：

```python
env = gymnasium.make(
    "Ant-v4",
    ctrl_cost_weight=1.0,       # 預設 0.5，調高抑制暴力動作
    contact_cost_weight=1e-3,   # 預設 5e-4，調高懲罰撞擊力
    healthy_reward=1.0,         # 保持站立的持續獎勵
    render_mode="rgb_array",    # 依需求調整
)
```

這個方法與修改二擇一即可，不需要同時使用。

---

## 修改位置總結

| 修改項目 | 在哪裡改 |
|---|---|
| 邊界懲罰 + truncated | `env.step()` 之後、存入 replay buffer 之前 |
| `prev_action` 初始化 | episode `reset()` 之後 |
| 動作正則化三項懲罰 | `env.step()` 之後、存入 replay buffer 之前 |
| `prev_action` 更新 | 動作正則化計算之後 |

---

## 建議係數調整順序

先用預設係數跑幾個 episode 觀察行為，再依以下方向調整：

- ant 還是蹬地飛出去 → 調高 `action_penalty` 係數（0.5 → 1.0）
- ant 動作變得太僵、幾乎不動 → 調低 `action_penalty` 係數（0.5 → 0.2）
- ant 還是跑出邊界 → 縮小 `MAX_RADIUS` 或調高 `BOUNDARY_PENALTY`
- episode 太短、還沒學到東西就截斷 → 調大 `MAX_RADIUS`
