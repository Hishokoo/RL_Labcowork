# v12a Spec — Anti-Phase Bonus + Smooth/Tilt 對齊微調

> **給 Claude Code (cc) 的完整實作提示詞**
> 產出兩個新檔案：`tools/gait_wrapper_12a.py` 與 `12a_train_td3.py`，並驗證程式碼可載入 03 checkpoint 開始 finetune。

---

## 1. 任務目標

在 12 (legacy + speed_gate=0.3 + 從 03 finetune) 已經跑出「25k 步達到峰值、之後逐步退化」的觀察基礎上，做出 v12a 變體：**在 wrapper reward 中補上三個對齊 scorecard 指標的訊號，避免 actor 在 25k 之後越優化 wrapper、越偏離 scorecard 弱維度**。

v12a 是「對齊缺口的最小侵入修補」，不改變主訓練流程（仍從 03 final_model 接續微調、仍用 legacy gait 公式作為主項），只新增三個對齊欄位 + 加密 eval。

---

## 2. 背景與診斷

### 2.1 現況

- **03** 是已成功的基底：1M 步訓練、x_vel ≈ 0.94 m/s、diagonal_sync = 0.712（scorecard 第一）、jerk = 0.028（第一）、CoT = 1.02（第一）。**弱點**：anti_phase = 0.209、contact_regularity = 0.239（皆全版本最低）。
- **12** = 從 03 接續 finetune，引入 `gait_speed_gate=0.3` 試圖在站著時關掉 r_gait 來推 agent 走得更乾淨。實測在 25k 達峰、之後分道揚鑣。

### 2.2 對齊缺口分析

`gait_mode="legacy"` 的 r_gait 公式：
```
r_gait = gait_weight · (0.4·diag1_sync + 0.4·diag2_sync + 0.2·cross_pattern)
```
其中 `diag1_sync = 1−|FL−BR|`、`diag2_sync = 1−|FR−BL|`、`cross_pattern = |FL−FR|`。

對照 scorecard 量測：

| Scorecard 指標 | wrapper 是否直接獎勵 | 缺口 |
|---|---|---|
| mean_speed / speed_error | ✓ via `forward_mode="deviation"` | 對齊 |
| **anti_phase** | ✗ 只有 cross_pattern（只看前腳）weak proxy | **大** |
| diagonal_sync | ✓ via diag1/diag2_sync | 對齊（03 已 0.712，不應再加重） |
| **contact_regularity** | ✗ 無時間項 | **大**（全版本最弱維度） |
| action_jerk | ✗ `smooth_weight=0` | **中**（03 強項沒被保護） |
| transport_cost | ✓ ctrl_weight 間接 | 對齊 |
| uprightness | ✗ `tilt_weight=0` | 小 |

### 2.3 25k 後退化的可能機制

1. **legacy 不直接獎 anti_phase，但 gate 強迫 x_vel ≥ 0.3**：agent 學到「維持速度 + 維持 diagonal_sync」就拿滿 r_gait——「四腳近同步起落」反而比真 trot 在這個 reward 下划算。03 本來 anti_phase=0.209 就低，finetune 把它推**更**低。
2. **gate 用瞬時 x_vel**：真步態 swing 相位讓 x_vel 暫時下探，r_gait 被切斷。agent 學到「拖著腳走」（不抬腳→速度不波動）保 gate。**注意**：v12a 暫不修這條，留給 v12b。

---

## 3. v12a 設計理念

### 3.1 三項對齊訊號

| 對齊項 | scorecard 目標 | wrapper 改動 | 風險 |
|---|---|---|---|
| anti_phase bonus | anti_phase | **新增** `antiphase_bonus_weight` 參數 | 低（疊加項，不動 legacy 主公式） |
| smooth penalty | action_jerk | 啟用既有 `smooth_weight` 欄位 | 極低（保護 03 強項） |
| tilt penalty | uprightness | 啟用既有 `tilt_weight` 欄位 | 極低（各版本本來就高） |

### 3.2 為何用「疊加 bonus」而不切到 `antiphase_gated` 模式

切到 `antiphase_gated` 會把整個 r_gait 乘上 anti_phase 當 gate。但 03 訓練時 anti_phase ≈ 0.2，切換後 r_gait 會從 ~1.6 跌到 ~0.32，actor/critic value function 受到大 shock，等於浪費 03 checkpoint 的 prior。

**疊加做法**：legacy 主公式不動，額外加一個 `antiphase_bonus_weight · anti_phase(contacts)` 的疊加項。bonus 套用在 speed gate 之後（讓 gate 同時保護 bonus），靜止時雙重歸零（anti_phase 本來就≈0、gate 也會切）。

### 3.3 為何不在 v12a 做 EMA gate / regularity proxy

- **EMA gate** 會改變 reward 的 timescale 動力學，與 anti_phase bonus 共同跑會把變因混在一起讀不出來。留給 v12b 單獨驗證。
- **regularity proxy** 需要在 wrapper 維護 contact 歷史窗口，是新狀態，留給 v13。

v12a 只做「不改變動力學、純疊加對齊項」的增量改動。

---

## 4. 檔案產出規格

### 4.1 `tools/gait_wrapper_12a.py`（新檔案）

基於 `tools/gait_wrapper_03.py` 複製，**唯一結構性變更**是：

1. `__init__` 新增參數 `antiphase_bonus_weight: float = 0.0`（預設 0 = 完全相容 03 行為）
2. `__init__` 內加上參數驗證 `if antiphase_bonus_weight < 0.0: raise ValueError(...)`
3. `_compute_reward` 內，在「4b. 柔和速度 gate」段落調整為先把 gate 算進獨立變數 `gate`、再分別套用到 `r_gait` 與新增的 anti_phase bonus
4. 在 components dict 新增 `"antiphase_bonus"` 鍵以利 TB logging
5. docstring 頭部更新為 v12a 設計說明

**完整檔案內容見第 5.1 節**。

### 4.2 `12a_train_td3.py`（新檔案）

基於 `12train_td3.py` 複製，變更：

1. import 改為 `from tools.gait_wrapper_12a import RealisticGaitWrapper`（如果 `gait_train` 已抽離 wrapper 注入點，可能要同時改 `tools/gait_train.py` 中對 wrapper 的引用——**請 cc 先檢查**，必要時新增環境變數或注入參數讓 `12a_train_td3.py` 可以指定使用哪個 wrapper）
2. `WRAP_KWARGS` 新增三個欄位：`smooth_weight=0.5`、`tilt_weight=1.0`、`antiphase_bonus_weight=0.5`
3. 訓練超參調整：`max_timesteps=80_000`、`learning_rate=5e-5`、`action_noise_sigma=0.02`、`eval_interval=10_000`、`video_interval=10_000`、`checkpoint_freq=10_000`
4. `output_dir` 預設改為 `output/12a_train_td3`
5. header 註解更新為 v12a 設計說明

**完整檔案內容見第 5.2 節**。

### 4.3 `tools/gait_train.py` 注入點檢查（cc 必做）

`12a_train_td3.py` 必須能讓 `finetune()` 使用 `gait_wrapper_12a.RealisticGaitWrapper` 而非 `gait_wrapper_03.RealisticGaitWrapper`。**請 cc 先打開 `tools/gait_train.py` 確認 wrapper 是怎麼接的**：

- 若 `gait_train.py` 直接 `from tools.gait_wrapper_03 import RealisticGaitWrapper`：
  改成接受一個可選參數 `wrapper_cls` 或環境變數讓呼叫端指定，**保留向後相容**（預設仍為 03）。
- 若 wrapper 已經是動態注入或從外部傳入：直接在 `12a_train_td3.py` 傳新 wrapper class。

切勿直接覆寫 `gait_wrapper_03.py` 或永久改掉 `gait_train.py` 的 import，會破壞 03/05/06/07/08/11/12 的可重現性。

---

## 5. 完整程式碼

### 5.1 `tools/gait_wrapper_12a.py`

```python
import gymnasium as gym
import numpy as np

from tools import gait_metrics

# Ant-v5 的四隻腳在這個 MuJoCo 模型裡沒有命名（body name 是空字串），
# 用 print(model.body(i).name for i in range(model.nbody)) 配合 geom bodyid 反查得到的對應關係：
#   body 4  (left_ankle_geom)   = front_left  (FL)
#   body 7  (right_ankle_geom)  = front_right (FR)
#   body 10 (third_ankle_geom)  = back_left   (BL)
#   body 13 (fourth_ankle_geom) = back_right  (BR)
FOOT_BODY_IDS = [4, 7, 10, 13]  # 順序：[FL, FR, BL, BR]


class RealisticGaitWrapper(gym.Wrapper):
    """
    v12a：在 03 wrapper 基礎上新增「antiphase_bonus_weight」做對齊微調。

    動機 ─ 12 (legacy + speed_gate=0.3, 從 03 finetune) 在 25k 達峰、之後退化。根因：
    wrapper 的 r_gait 公式 (0.4·diag1 + 0.4·diag2 + 0.2·cross_pattern) 不直接獎勵
    scorecard 量測的 anti_phase。actor 25k 後越優化 wrapper、anti_phase 反而被推更低。

    v12a 補三道對齊訊號 (與 03 完全向後相容)：
      1. antiphase_bonus_weight (新增): r_gait 之外疊加 w·anti_phase(contacts)，受同一
         speed gate 保護。直接對齊 scorecard 的 anti_phase 指標，而非透過 cross_pattern
         的 weak proxy。預設 0 = 不啟用。
      2. smooth_weight (沿用 03 已實作欄位, 03/12 預設 0): 直接對齊 scorecard 的
         action_jerk，保護 03 的 0.028 強項在 finetune 期間漂掉。
      3. tilt_weight   (沿用 03 已實作欄位, 03/12 預設 0): 對齊 scorecard 的
         uprightness。各版本本來就高，貢獻量小，純粹當對齊保險。

    為何不切到 antiphase_gated 模式：那會把整個 r_gait 乘上 anti_phase。03 訓練時
    anti_phase ≈ 0.2，切換後 r_gait 從 ~1.6 跌到 ~0.32，actor/critic value function
    受大 shock，浪費 03 checkpoint 的 prior。疊加 bonus 是更穩的增量改動。

    其餘行為（gait_mode 三種模式、forward_mode 兩種、reward_structure 兩種、
    ctrl_schedule、gait_speed_gate、intra_weight 等）與 gait_wrapper_03.py 完全一致。
    """

    def __init__(
        self,
        env,
        target_speed: float = 1.0,
        ctrl_weight: float = 5.0,
        gait_weight: float = 2.0,
        posture_weight: float = 2.0,
        alive_weight: float = 1.0,
        contact_threshold: float = 1.0,
        gait_mode: str = "legacy",
        forward_mode: str = "deviation",
        forward_weight: float = 1.0,
        smooth_weight: float = 0.0,
        tilt_weight: float = 0.0,
        reward_structure: str = "additive",
        forward_gate_shape: str = "cap",
        intra_weight: float = 0.25,
        gait_speed_gate: float = 0.0,
        ctrl_schedule: tuple | None = None,
        antiphase_bonus_weight: float = 0.0,  # v12a 新增
    ):
        super().__init__(env)
        self.target_speed = target_speed
        self.ctrl_weight = ctrl_weight
        self.gait_weight = gait_weight
        self.posture_weight = posture_weight
        self.alive_weight = alive_weight
        self.contact_threshold = contact_threshold
        if gait_mode not in ("legacy", "antiphase", "antiphase_gated"):
            raise ValueError(f"未知的 gait_mode：{gait_mode}（可用 'legacy' / 'antiphase' / 'antiphase_gated'）")
        if forward_mode not in ("deviation", "progress"):
            raise ValueError(f"未知的 forward_mode：{forward_mode}（可用 'deviation' / 'progress'）")
        if reward_structure not in ("additive", "forward_gated"):
            raise ValueError(f"未知的 reward_structure：{reward_structure}（可用 'additive' / 'forward_gated'）")
        if forward_gate_shape not in ("cap", "tent"):
            raise ValueError(f"未知的 forward_gate_shape：{forward_gate_shape}（可用 'cap' / 'tent'）")
        self.gait_mode = gait_mode
        self.forward_mode = forward_mode
        self.forward_weight = forward_weight
        self.smooth_weight = smooth_weight
        self.tilt_weight = tilt_weight
        self.forward_gate_shape = forward_gate_shape
        self.reward_structure = reward_structure
        if not 0.0 <= intra_weight <= 0.5:
            raise ValueError(f"intra_weight 須在 [0, 0.5]（兩個 intra 項合計 ≤ 1）：{intra_weight}")
        self.intra_weight = intra_weight
        if gait_speed_gate < 0.0:
            raise ValueError(f"gait_speed_gate 須 ≥ 0（0=關閉、>0=啟用的速度門檻 m/s）：{gait_speed_gate}")
        self.gait_speed_gate = gait_speed_gate
        if antiphase_bonus_weight < 0.0:
            raise ValueError(f"antiphase_bonus_weight 須 ≥ 0：{antiphase_bonus_weight}")
        self.antiphase_bonus_weight = antiphase_bonus_weight
        # ctrl 排程（None=關閉，用固定 ctrl_weight）。(t0, t1, c0, c1)：step≤t0→c0、
        # t0..t1 線性 c0→c1、>t1→c1。
        if ctrl_schedule is not None:
            t0, t1, c0, c1 = ctrl_schedule
            if not (0 <= t0 <= t1):
                raise ValueError(f"ctrl_schedule 須 0≤t0≤t1：{ctrl_schedule}")
        self.ctrl_schedule = ctrl_schedule
        self._gstep = 0
        self.prev_contacts = np.zeros(4)
        self.prev_action = np.zeros(env.action_space.shape[0], dtype=np.float64)
        self.foot_body_ids = FOOT_BODY_IDS

    def _effective_ctrl_weight(self) -> float:
        if self.ctrl_schedule is None:
            return self.ctrl_weight
        t0, t1, c0, c1 = self.ctrl_schedule
        if self._gstep <= t0:
            return c0
        if self._gstep >= t1:
            return c1
        return c0 + (c1 - c0) * (self._gstep - t0) / (t1 - t0)

    def reset(self, **kwargs):
        self.prev_contacts = np.zeros(4)
        self.prev_action = np.zeros(self.env.action_space.shape[0], dtype=np.float64)
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, _orig_reward, terminated, truncated, info = self.env.step(action)

        contacts = self._get_foot_contacts()
        is_healthy = not terminated
        reward, components = self._compute_reward(action, info, is_healthy, contacts)

        info["foot_contacts"] = contacts
        info["reward_components"] = components
        info["original_reward"] = _orig_reward
        self.prev_contacts = contacts
        self.prev_action = np.asarray(action, dtype=np.float64)
        self._gstep += 1

        return obs, reward, terminated, truncated, info

    def _get_foot_contacts(self) -> np.ndarray:
        cfrc = self.env.unwrapped.data.cfrc_ext[self.foot_body_ids]
        contact_magnitudes = np.linalg.norm(cfrc, axis=1)
        return (contact_magnitudes > self.contact_threshold).astype(np.float32)

    def _compute_reward(self, action, info, is_healthy, contacts):
        # 1. 速度 reward
        x_vel = info.get("x_velocity", 0.0)
        if self.forward_mode == "progress":
            r_forward = self.forward_weight * max(0.0, min(x_vel, self.target_speed))
        else:  # "deviation"
            r_forward = -self.forward_weight * abs(x_vel - self.target_speed)

        # 2. 存活基本分
        r_alive = self.alive_weight if is_healthy else 0.0

        # 3. 控制懲罰
        r_ctrl = -self._effective_ctrl_weight() * float(np.sum(np.square(action)))

        # 4. 步態 reward 主項（與 03 完全一致）
        if self.gait_mode == "antiphase_gated":
            intra1 = 1.0 - abs(contacts[0] - contacts[3])
            intra2 = 1.0 - abs(contacts[1] - contacts[2])
            anti = gait_metrics.anti_phase(contacts)
            base = 1.0 - 2.0 * self.intra_weight
            r_gait = self.gait_weight * anti * (base + self.intra_weight * intra1 + self.intra_weight * intra2)
        elif self.gait_mode == "antiphase":
            intra1 = 1.0 - abs(contacts[0] - contacts[3])
            intra2 = 1.0 - abs(contacts[1] - contacts[2])
            anti = gait_metrics.anti_phase(contacts)
            r_gait = self.gait_weight * (0.2 * intra1 + 0.2 * intra2 + 0.6 * anti)
        else:  # "legacy"
            diag1_sync = 1.0 - abs(contacts[0] - contacts[3])
            diag2_sync = 1.0 - abs(contacts[1] - contacts[2])
            cross_pattern = abs(contacts[0] - contacts[1])
            r_gait = self.gait_weight * (0.4 * diag1_sync + 0.4 * diag2_sync + 0.2 * cross_pattern)

        # 4b. 柔和速度 gate（v12a：抽出 gate 變數，讓 anti_phase bonus 共用同一個 gate）
        if self.gait_speed_gate > 0.0:
            p = float(np.clip(max(x_vel, 0.0) / self.gait_speed_gate, 0.0, 1.0))
            gate = p * p * (3.0 - 2.0 * p)
        else:
            gate = 1.0
        r_gait *= gate

        # 4c. v12a 新增：anti_phase 對齊獎金（受同一 speed gate 保護）
        # 設計意圖：legacy 主公式只用 cross_pattern 弱代理 anti_phase，這裡直接疊加
        # 完整 anti_phase(contacts) 訊號當 bonus。站著時 anti_phase ≈ 0，gate 也 ≈ 0，
        # 雙重歸零；真 trot 單腳支撐瞬間 anti_phase = 1，bonus 最大 = antiphase_bonus_weight。
        antiphase_value = float(gait_metrics.anti_phase(contacts))
        r_antiphase_bonus = self.antiphase_bonus_weight * antiphase_value * gate
        r_gait += r_antiphase_bonus

        # 5. 姿態 reward（軀幹高度）
        torso_z = self.env.unwrapped.data.qpos[2]
        r_posture = -self.posture_weight * abs(torso_z - 0.6)

        # 6. 動作平滑懲罰（jerk）
        action_arr = np.asarray(action, dtype=np.float64)
        jerk = float(np.sum(np.square(action_arr - self.prev_action)))
        r_smooth = -self.smooth_weight * jerk

        # 7. 軀幹直立懲罰
        qpos = self.env.unwrapped.data.qpos
        upright = gait_metrics.uprightness(qpos)
        r_tilt = -self.tilt_weight * (1.0 - upright)

        if self.reward_structure == "forward_gated":
            if self.forward_gate_shape == "tent":
                forward_progress = self.target_speed * max(
                    0.0, 1.0 - abs(x_vel - self.target_speed) / self.target_speed)
            else:
                forward_progress = max(0.0, min(x_vel, self.target_speed))
            gait_contrib = forward_progress * r_gait
            total = forward_progress + gait_contrib + r_alive + r_ctrl + r_smooth + r_tilt + r_posture
            r_forward = forward_progress
            r_gait = gait_contrib
        else:
            total = r_forward + r_alive + r_ctrl + r_gait + r_posture + r_smooth + r_tilt

        components = {
            "forward": r_forward,
            "alive": r_alive,
            "ctrl": r_ctrl,
            "gait": r_gait,
            "antiphase_bonus": r_antiphase_bonus,  # v12a 新增 logging 欄位
            "posture": r_posture,
            "smooth": r_smooth,
            "tilt": r_tilt,
            "x_velocity": x_vel,
            "torso_z": torso_z,
            "uprightness": upright,
            "anti_phase": antiphase_value,
        }
        return float(total), components
```

### 5.2 `12a_train_td3.py`

```python
# 執行前請安裝：pip install "gymnasium[mujoco]" stable-baselines3 moviepy tqdm rich
# 12a：★ scorecard-aligned finetune ★ —— v12a 把對齊缺口補上的最小侵入修補。
#
# 12 (legacy + speed_gate=0.3, 從 03 finetune) 在 25k 達峰、之後分道揚鑣。Claude 診斷：
# wrapper r_gait (0.4·diag1 + 0.4·diag2 + 0.2·cross) 不直接獎勵 scorecard 的 anti_phase。
# 25k 後 actor 越優化 wrapper、anti_phase 反而被推更低，「四腳近同步起落」比真 trot 划算。
#
# 12a 在 03 路徑上補三道對齊訊號（皆為「不改動力學、純疊加」的增量改動）：
#   1) antiphase_bonus_weight=0.5：r_gait 之外疊加 anti_phase 直接訊號（受同 speed gate 保護）
#   2) smooth_weight=0.5：保護 03 的 jerk=0.028 強項不漂掉
#   3) tilt_weight=1.0：對齊 uprightness（小貢獻、純保險）
#
# 為何不切 antiphase_gated 模式：那會把整個 r_gait 乘上 anti_phase。03 訓練時 anti_phase ≈ 0.2，
# 切換後 r_gait 從 ~1.6 跌到 ~0.32，actor/critic value 受大 shock，浪費 03 prior。
#
# 為何不在 12a 改 EMA gate / regularity proxy：兩者會改變 reward 動力學的 timescale，
# 與 anti_phase bonus 共跑會把變因混在一起。EMA gate 留 v12b、regularity proxy 留 v13。
#
# 訓練超參相對 12 的調整：
#   - max_timesteps 120k → 80k（既然峰值在 25k 附近，沒必要跑那麼長）
#   - learning_rate 1e-4 → 5e-5（finetune 不需要這麼大 step）
#   - action_noise 0.03 → 0.02（finetune 不需要這麼大探索）
#   - eval/video/ckpt 25k → 10k（加密 eval 是防回頭找不到峰的最便宜保險）
#
# 成功標準（看影片 + scorecard）：episode_length=1000、speed≥0.9、anti_phase≥0.27、
# diagonal_sync 不退化（≥0.65）、jerk≤0.05、CoT≤1.5、觀感不退化。
#
#   cd ~/RL_Labcowork && MUJOCO_GL=egl python 12a_train_td3.py
import os

from tools.gait_train import finetune
from tools.gait_wrapper_12a import RealisticGaitWrapper as Wrapper12a

# ── reward 設定（= 12 + 三道對齊訊號）─────────────────────────────────────────
WRAP_KWARGS = dict(
    target_speed=1.0,
    ctrl_weight=5.0, gait_weight=2.0, posture_weight=2.0, alive_weight=1.0,
    contact_threshold=1.0,
    gait_mode="legacy", forward_mode="deviation", forward_weight=1.0,
    smooth_weight=0.5,             # v12a 新增：對齊 action_jerk（保護 03 強項）
    tilt_weight=1.0,               # v12a 新增：對齊 uprightness
    reward_structure="additive",
    gait_speed_gate=0.3,
    antiphase_bonus_weight=0.5,    # v12a 新增：直接對齊 scorecard 的 anti_phase
)
# ─────────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    finetune(
        WRAP_KWARGS,
        output_dir=os.environ.get("OUTPUT_DIR", "output/12a_train_td3"),
        init_model_path=os.environ.get("INIT_MODEL", "output/03train_td3/final_model.zip"),
        target_speed=WRAP_KWARGS["target_speed"],
        max_timesteps=int(os.environ.get("MAX_TIMESTEPS", 80_000)),
        learning_rate=float(os.environ.get("LEARNING_RATE", 5e-5)),
        action_noise_sigma=float(os.environ.get("ACTION_NOISE", 0.02)),
        eval_interval=int(os.environ.get("EVAL_INTERVAL", 10_000)),
        video_interval=int(os.environ.get("VIDEO_INTERVAL", 10_000)),
        checkpoint_freq=int(os.environ.get("CHECKPOINT_FREQ", 10_000)),
        wrapper_cls=Wrapper12a,        # 若 finetune() 已支援；否則見第 4.3 節
    )
```

---

## 6. 驗收標準

### 6.1 程式可正確啟動

cc 完成後必須驗證：

1. `python -c "from tools.gait_wrapper_12a import RealisticGaitWrapper"` 不報錯
2. `python 12a_train_td3.py` 啟動後能：
   - 成功載入 `output/03train_td3/final_model.zip`
   - 不執行 random warmup（learning_starts=0，第一個 episode 即用載入策略）
   - 第 1000 步左右輸出 reward components 含 `antiphase_bonus` 鍵
   - TensorBoard log 含 `antiphase_bonus`、`smooth`、`tilt` 三個新欄位
3. 03 既有檔案（`gait_wrapper_03.py`、`12train_td3.py`）未被改動
4. 若 `tools/gait_train.py` 必須改：保持向後相容（預設仍用 03 wrapper），既有 `03train_td3.py` / `12train_td3.py` 跑起來行為不變

### 6.2 訓練後 scorecard 對比目標

在 v12a 80k 訓練完成、跑 `tools/eval_scorecard.py` 後：

| 指標 | 03 基準 | 12 峰值 (25k) | v12a 目標 | 必達 / 期望 |
|---|---|---|---|---|
| episode_length | 1000 | 1000 | 1000 | **必達** |
| mean_speed | 0.94 | ~0.94 | 0.90–1.05 | **必達** |
| **anti_phase** | 0.209 | ? | **≥ 0.27** | **必達**（v12a 主目標） |
| diagonal_sync | 0.712 | ? | ≥ 0.60 | 期望（不可大退化） |
| action_jerk | 0.028 | ? | ≤ 0.05 | 期望 |
| transport_cost | 1.02 | ? | ≤ 1.5 | 期望 |
| uprightness | 0.986 | ? | ≥ 0.98 | 期望 |
| contact_regularity | 0.239 | ? | （不期望提升，留給 v13） | — |

### 6.3 失敗模式診斷指引（cc 跑完後若不達標可給 user 參考）

- **anti_phase 沒升 / diagonal_sync 大跌**：bonus weight 太小被 legacy 主項蓋過，或 gate 0.3 把 bonus 也濾掉太多。下一步嘗試 `antiphase_bonus_weight=1.0` 或 `gait_speed_gate=0.15`。
- **jerk 上升 / 觀感變差**：`smooth_weight=0.5` 不足；下一步嘗試 `smooth_weight=1.0`。
- **早期 episode 直接摔（< 500 步）**：三個對齊項合計 reward 太負壓垮 03 prior；先把 `smooth_weight` 與 `tilt_weight` 都降一半重跑。
- **峰值還是落在 < 20k**：把 `learning_rate` 再降到 3e-5、`action_noise` 降到 0.01；或檢查 03 checkpoint 是否真的有完全載入（replay buffer 是否被清空、learning_starts 是否真為 0）。

---

## 7. 訓練指令

```bash
cd ~/RL_Labcowork && MUJOCO_GL=egl python 12a_train_td3.py
```

可選環境變數覆寫：
```bash
MAX_TIMESTEPS=60000 LEARNING_RATE=3e-5 ACTION_NOISE=0.015 \
  MUJOCO_GL=egl python 12a_train_td3.py
```

訓練後評估：
```bash
python tools/eval_scorecard.py --model output/12a_train_td3/final_model.zip
# 或對每個 ckpt 跑 scorecard 找峰
for ckpt in output/12a_train_td3/checkpoints/*.zip; do
  python tools/eval_scorecard.py --model "$ckpt"
done
```

---

## 8. 後續實驗藍圖（與 v12a 無關，供 cc 知道整體計畫，不要在這次實作做）

- **v12b** = v12a + EMA gate（gait_speed_gate 改用 0.5 秒 EMA 平均速度而非瞬時 x_vel），目標解掉 swing 相位被切斷的問題
- **v13** = v12b + contact_regularity per-step 代理（最後 N frame 接觸視窗 + 最小 dwell time 的轉換獎勵），目標推 regularity 從 0.239 到 0.32+

v12a 是這條 roadmap 的第一步，不需要在這次實作中觸碰 v12b/v13 的部分。

---

## 9. cc 工作清單總結

- [ ] 開 `tools/gait_train.py` 檢查 wrapper 注入點，若必要：新增 `wrapper_cls` 參數並保持向後相容
- [ ] 新建 `tools/gait_wrapper_12a.py`（內容見 5.1）
- [ ] 新建 `12a_train_td3.py`（內容見 5.2）
- [ ] 不要動：`tools/gait_wrapper_03.py`、`12train_td3.py`、任何其他既有檔案的功能性程式碼
- [ ] 跑 `python -c "from tools.gait_wrapper_12a import RealisticGaitWrapper"` 驗證 import
- [ ] 跑 `python 12a_train_td3.py` 確認能載入 03 checkpoint 並開始訓練（看到第一個 episode 結束、reward components 含 antiphase_bonus 鍵後 Ctrl-C 結束即可，不需要跑完 80k）
- [ ] 回報 `output/12a_train_td3/` 的目錄結構與第一個 eval 點的 reward components（讓 user 確認對齊訊號真的在跑）
