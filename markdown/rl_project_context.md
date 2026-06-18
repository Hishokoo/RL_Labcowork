# RL 期末專題：TD3 + MuJoCo Ant-v4

## 專題概述

本專題使用 **TD3（Twin Delayed DDPG）** 演算法，訓練 **Gymnasium MuJoCo Ant-v4** 環境中的四足機器人完成連續控制任務。目標是產出穩定的訓練實作與分析報告。

---

## 環境

- **平台：** Gymnasium（OpenAI Gym 繼任者）
- **任務：** `Ant-v4`（MuJoCo 物理模擬）
- **觀測空間：** 27 維連續向量（關節角度、速度、接觸力等）
- **動作空間：** 8 維連續動作（各關節力矩，範圍 \[-1, 1\]）
- **Reward：** 前進速度 + 存活獎勵 − 控制懲罰 − 接觸懲罰

---

## 演算法：TD3

TD3 是為 **連續動作空間** 設計的 off-policy actor-critic 演算法，解決 DDPG 的過估計問題。

### 核心機制

| 機制 | 說明 |
|------|------|
| **Twin Critics** | 兩個獨立 Q-network，取最小值減少過估計偏差 |
| **Delayed Policy Update** | Actor 每 2 步才更新一次，提升穩定性 |
| **Target Policy Smoothing** | 在 target action 加入 clipped noise，防止 Q-function 對尖峰動作過擬合 |

### 主要 Hyperparameters（建議起點）

```python
actor_lr        = 3e-4
critic_lr       = 3e-4
batch_size      = 256
replay_buffer   = 1_000_000
gamma           = 0.99
tau             = 0.005          # soft update 係數
policy_delay    = 2
noise_std       = 0.2
noise_clip      = 0.5
exploration_noise = 0.1          # Gaussian noise added to actions during rollout
```

---

## 演算法比較（MuJoCo 連續控制）

| 演算法 | Ant 適合度 | 複雜度 | 備註 |
|--------|-----------|--------|------|
| **TD3** | ⭐⭐⭐⭐⭐ | 中 | 本專題選用，穩定且好解釋 |
| **SAC** | ⭐⭐⭐⭐⭐ | 中 | 效果略優於 TD3，entropy 自動調節 |
| **PPO** | ⭐⭐⭐⭐ | 低 | on-policy，sample efficiency 較差 |
| **DDPG** | ⭐⭐⭐ | 低 | TD3 前身，訓練較不穩定 |
| **TQC** | ⭐⭐⭐⭐ | 高 | 效果最佳但複雜度高 |

---

## 工具與套件

```
gymnasium[mujoco]       # 環境
stable-baselines3       # TD3/SAC 參考實作（快速驗證用）
torch                   # 自行實作用
tensorboard / wandb     # 訓練曲線記錄
```

> 建議先用 Stable-Baselines3 跑通 baseline，再視需求自行實作。

---

## 訓練規模參考

| 指標 | 數值 |
|------|------|
| 建議總步數 | 1M～3M steps |
| GPU 訓練時間（1M steps）| 約 2～4 小時 |
| 收斂 reward（Ant-v4） | 3000～6000+（視實作品質） |

---

## 報告方向

- 訓練 reward 曲線（episode return vs. timestep）
- Ablation study 建議：
  - 拿掉 twin critics → 觀察過估計現象
  - 關閉 delayed policy update → 比較穩定性
  - TD3 vs SAC 同環境對比

---

## 參考資料

- Fujimoto et al., 2018 — [*Addressing Function Approximation Error in Actor-Critic Methods* (TD3 原始論文)](https://arxiv.org/abs/1802.09477)
- [Gymnasium MuJoCo 文件](https://gymnasium.farama.org/environments/mujoco/ant/)
- [Stable-Baselines3 TD3](https://stable-baselines3.readthedocs.io/en/master/modules/td3.html)
