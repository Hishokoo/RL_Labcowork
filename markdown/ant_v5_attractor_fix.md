# Ant-v5 TD3 訓練：跳出「站著不動」局部解 + 改善跑姿

## 問題診斷

過去 5 次實驗失敗的共同原因：**Ant-v5 的 `healthy_reward=1.0`（預設值）讓「站著不動」成為一個 risk-free 的 attractor basin**。

當降低 `forward_reward_weight`（從 1.0 → 0.6 / 0.8）時，發生的事是：

| 行為 | 預期每步 reward | 風險 |
|---|---|---|
| 站著不動 | `healthy(1.0) - ctrl(~0) - contact(~0)` ≈ **+1.0** | 零風險 |
| 慢走 vx≈1.0 | `0.6×1.0 + 1.0 - ctrl - action_penalty` ≈ **+1.2** | 可能摔倒丟掉整段未來 |
| 摔倒 | 0 + episode 終止 | — |

走路的邊際收益（+0.2/step）對 TD3 的 deterministic policy + 小幅探索噪聲來說太低，承擔不了摔倒風險，所以 agent 收斂到「站著」。

過去嘗試的 stillness penalty 失敗，是因為它提高了「站著」的成本，但沒提高「走路」的可達性——結果變成「摔倒擺爛」優於兩者。

---

## 解方總覽

**雙管齊下：**
1. **方向 A**：移除 `healthy_reward`，徹底消除「站著」的收入來源
2. **方向 B**：加重 `contact_cost_weight`，直接懲罰造成「不科學跑姿」的根因（蹬地力過大）

兩個方向互補，建議合併使用。

---

## 修改一：環境初始化

```python
env = gymnasium.make(
    "Ant-v5",
    forward_reward_weight=1.0,    # 還原預設值，不再壓速度
    healthy_reward=0.0,           # ★ 關鍵：拿掉「活著就有分」
    ctrl_cost_weight=0.5,         # 預設值，保留
    contact_cost_weight=5e-3,     # ★ 從 5e-4 提高 10 倍，直接懲罰重踩
    terminate_when_unhealthy=True,  # 預設，保留（摔倒結束 episode）
    render_mode=...,
)
```

**參數說明：**

- `healthy_reward=0.0`：站著不動不再有任何收入，agent 唯一拿分的方式是 `forward_reward`。`terminate_when_unhealthy=True` 仍會在摔倒時結束 episode，提供「不要摔倒」的負面訊號，不需要額外懲罰。

- `contact_cost_weight=5e-3`：Ant-v5 預設值 `5e-4` 太小幾乎沒作用。提高 10 倍後，「重踩」會直接被罰，agent 必須學會輕落地。如果跑姿改善仍不夠明顯，可以再調到 `1e-2`（但太高會抑制移動）。

---

## 修改二：保留現有的邊界機制（不動）

邊界機制已在實驗中驗證有效，保持不變：

```python
SOFT_RADIUS           = 6.0
MAX_RADIUS            = 8.0
BOUNDARY_PENALTY      = 5.0
ACTION_PENALTY_WEIGHT = 0.5
ACTION_DIFF_PENALTY_WEIGHT = 0.1

# 邊界懲罰
dist = info["distance_from_origin"]
if dist > SOFT_RADIUS:
    reward -= BOUNDARY_PENALTY * (dist - SOFT_RADIUS)
boundary_violation = dist > MAX_RADIUS
if boundary_violation:
    truncated = True

# 動作正則化
reward -= ACTION_PENALTY_WEIGHT * np.sum(np.square(action))
reward -= ACTION_DIFF_PENALTY_WEIGHT * np.sum(np.square(action - prev_action))

# Bootstrap bug fix
replay_buffer.add(obs, action, next_obs, reward,
                  float(terminated or boundary_violation))
```

---

## 修改三：移除 `forward_reward_weight` 的客製化邏輯

如果現有程式碼中有覆寫 `forward_reward_weight` 的地方（之前實驗設成 0.6/0.8），請刪除——這次要還原使用 Ant-v5 原生的 1.0。

---

## 預期行為與訓練監控

**訓練前期（0–100k steps）**

- Episode 長度可能會比過去短，因為沒有 healthy_reward 撐著，摔倒時收益歸零
- Reward 可能整體偏低甚至為負，這是正常的（少了 `+1000` healthy bonus per episode）
- **關鍵觀察點：avg_vx 應該 > 0.1**，代表 agent 開始嘗試移動

**訓練中期（100k–500k steps）**

- Episode 長度應該逐漸拉長（學會邊走邊平衡）
- avg_vx 應該穩定在 0.5–1.5 之間
- max_dist 應該偶爾觸及 SOFT_RADIUS（6m）但很少超過 MAX_RADIUS

**訓練後期（500k–1M steps）**

- Episode 穩定接近 1000 步
- 跑姿應該明顯比過去自然（contact_cost 在懲罰重踩）

**如果 200k steps 之後 avg_vx 仍 < 0.1**，代表 agent 還是學不會走路，可能需要：
- 把 `forward_reward_weight` 提高到 2.0（強化移動誘因）
- 或縮短 episode 長度（從 1000 改成 500），縮短 reward 信號的延遲

---

## 風險與備案

**風險 1：移除 healthy_reward 後 agent 學成「快速摔倒結束 episode」**

機率不高，因為 `terminate_when_unhealthy=True` 會讓摔倒丟掉所有未來 forward_reward 機會，這是強烈的負面訊號。但如果發生（觀察到 episode 長度持續 < 50），可以加回一點點 healthy_reward：

```python
healthy_reward=0.1,  # 改成原本的 1/10
```

**風險 2：contact_cost 太強，agent 為避免重踩而選擇不動**

如果觀察到 ant 又退化成「站著但抖動」（avg_vx 很低但動作大），把 `contact_cost_weight` 從 `5e-3` 調回 `2e-3`。

**保底方案**

如果 300k steps 內仍未看到改善跡象，回到原始 report.html 中的「選項 C」：保留 `forward_reward_weight=1.0` 跟邊界機制，跑姿問題寫成 future work。在 deadline 壓力下，「會走會守邊界」比「跑姿自然但不會走」更適合做 demo。

---

## 實作 checklist

- [ ] 確認 `gymnasium.make("Ant-v5", ...)` 加上 `healthy_reward=0.0` 與 `contact_cost_weight=5e-3`
- [ ] 確認 `forward_reward_weight` 還原成 1.0（移除過往的 0.6/0.8 覆寫）
- [ ] 保留邊界機制（SOFT_RADIUS / MAX_RADIUS / BOUNDARY_PENALTY）與 bootstrap bug 修正
- [ ] 保留 ACTION_PENALTY_WEIGHT=0.5 與 ACTION_DIFF_PENALTY_WEIGHT=0.1
- [ ] **移除所有 stillness penalty 邏輯**（這次靠 healthy_reward=0 解決）
- [ ] **移除所有 MIN_SPEED / TARGET_SPEED / SPEED_PENALTY 邏輯**（已驗證無效）
- [ ] 訓練至少跑 200k steps 再評估，不要過早判斷
- [ ] TensorBoard 監控指標：episode_length, avg_vx, contact_cost（如可記錄）
