
# Claude Code 任務規格：TD3 + Ant-v5 真實步態訓練

## 任務總覽

實作一份 TD3 訓練程式，在 **Gymnasium MuJoCo `Ant-v5`** 環境上訓練四足機器人，目標是訓練出**接近真實動物的步態**（四足交替著地、動作平滑），而非預設 reward 訓出的「慣性甩動」高速移動行為。

最終 reward 數值會比 benchmark（3000+）低，預期落在 **800–1500** 區間，這是設計上的取捨。

---

## 檔案位置與命名約定（重要：先觀察）

> ⚠️ **第一步：先 `ls` 列出專案根目錄結構，觀察既有檔案命名規律後再決定輸出位置。**

預期命名邏輯：
- 訓練檔輸出為 **`03_train_td3.py`**（沿用既有 `NN_xxx.py` 的數字前綴規律；若既有檔案是 `01_xxx.py`、`02_xxx.py` 則延續）
- 訓練輸出目錄（例如 `runs/`、`logs/`、`videos/`）也請沿用既有結構，不要自行新增同義目錄
- 若有 `utils/`、`wrappers/` 等模組化資料夾，請把 `RealisticGaitWrapper` 放進對應位置；若沒有就直接放在 `03_train_td3.py` 同檔內

請在開始實作前，先在訊息中回報你觀察到的命名規律與計劃的輸出位置。

---

## 環境：Ant-v5（注意不是 v4）

Ant-v5 與 v4 的差異：
- `Ant-v5` 預設已包含 contact forces 在觀測中
- reward 參數命名一致：`healthy_reward`、`forward_reward_weight`、`ctrl_cost_weight`、`contact_cost_weight`
- 動作空間：8 維 `[-1, 1]`
- 觀測空間：約 105 維（含 contact forces）

建立環境時應該明確指定 reward weights：

```python
env = gym.make(
    "Ant-v5",
    healthy_reward=1.0,
    forward_reward_weight=1.0,
    ctrl_cost_weight=0.5,
    contact_cost_weight=5e-4,
)
```

---

## 核心改動：RealisticGaitWrapper

為了訓出真實步態，包一層 `gym.Wrapper` 改寫 reward。**不要去改 Gymnasium 原始碼**。

### Wrapper 規格

```python
import gymnasium as gym
import numpy as np

class RealisticGaitWrapper(gym.Wrapper):
    """
    將 Ant-v5 的 reward 改寫為步態導向：
    - 速度有上限（不是越快越好）
    - 強懲罰大幅動作（鼓勵省力）
    - 獎勵四足交替著地（trot 步態）
    - 維持軀幹姿態穩定
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
    ):
        super().__init__(env)
        self.target_speed = target_speed
        self.ctrl_weight = ctrl_weight
        self.gait_weight = gait_weight
        self.posture_weight = posture_weight
        self.alive_weight = alive_weight
        self.contact_threshold = contact_threshold
        self.prev_contacts = np.zeros(4)

        # 取得四隻腳的 body id（Ant-v5 的腳名稱）
        # 順序：[front_left, front_right, back_left, back_right]
        foot_names = [
            "front_left_foot",
            "front_right_foot",
            "back_left_foot",
            "back_right_foot",
        ]
        self.foot_body_ids = []
        for name in foot_names:
            try:
                bid = self.env.unwrapped.model.body(name).id
                self.foot_body_ids.append(bid)
            except KeyError:
                # 若名稱不符，請印出所有 body 名稱檢查
                all_bodies = [
                    self.env.unwrapped.model.body(i).name
                    for i in range(self.env.unwrapped.model.nbody)
                ]
                raise KeyError(f"找不到 {name}，全部 body：{all_bodies}")

    def reset(self, **kwargs):
        self.prev_contacts = np.zeros(4)
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, _orig_reward, terminated, truncated, info = self.env.step(action)

        contacts = self._get_foot_contacts()
        reward, components = self._compute_reward(action, info, contacts)

        info["foot_contacts"] = contacts
        info["reward_components"] = components
        info["original_reward"] = _orig_reward
        self.prev_contacts = contacts

        return obs, reward, terminated, truncated, info

    def _get_foot_contacts(self) -> np.ndarray:
        """回傳四隻腳的接觸狀態（0 或 1）。"""
        cfrc = self.env.unwrapped.data.cfrc_ext[self.foot_body_ids]
        contact_magnitudes = np.linalg.norm(cfrc, axis=1)
        return (contact_magnitudes > self.contact_threshold).astype(np.float32)

    def _compute_reward(self, action, info, contacts):
        # 1. 速度向目標靠近（不是越快越好）
        x_vel = info.get("x_velocity", 0.0)
        r_forward = -abs(x_vel - self.target_speed)

        # 2. 存活基本分（由 wrapper 內判斷，沿用原 env 的 healthy 判定）
        is_healthy = info.get("is_healthy", True)
        r_alive = self.alive_weight if is_healthy else 0.0

        # 3. 控制懲罰（加重版）
        r_ctrl = -self.ctrl_weight * float(np.sum(np.square(action)))

        # 4. 步態 reward（核心）：trot 對角線同步
        diag1_sync = 1.0 - abs(contacts[0] - contacts[3])  # FL vs BR
        diag2_sync = 1.0 - abs(contacts[1] - contacts[2])  # FR vs BL
        cross_pattern = abs(contacts[0] - contacts[1])     # FL vs FR 應反相
        r_gait = self.gait_weight * (0.4 * diag1_sync + 0.4 * diag2_sync + 0.2 * cross_pattern)

        # 5. 姿態 reward（軀幹高度）
        torso_z = self.env.unwrapped.data.qpos[2]
        r_posture = -self.posture_weight * abs(torso_z - 0.6)

        total = r_forward + r_alive + r_ctrl + r_gait + r_posture
        components = {
            "forward": r_forward,
            "alive": r_alive,
            "ctrl": r_ctrl,
            "gait": r_gait,
            "posture": r_posture,
            "x_velocity": x_vel,
            "torso_z": torso_z,
        }
        return float(total), components
```

---

## 訓練程式結構（`03_train_td3.py`）

### 推薦做法：使用 Stable-Baselines3 TD3

考量到時程（剩 5 天），優先使用 SB3 的 TD3 實作，搭配自訂 wrapper 與 callback。

```python
import gymnasium as gym
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np

# === 環境構建 ===
def make_env(seed=0, record_video=False, video_folder=None):
    env = gym.make("Ant-v5", render_mode="rgb_array" if record_video else None)
    env = RealisticGaitWrapper(
        env,
        target_speed=1.0,
        ctrl_weight=5.0,
        gait_weight=2.0,
        posture_weight=2.0,
    )
    env.reset(seed=seed)
    return env

train_env = DummyVecEnv([lambda: make_env(seed=0)])

# === TD3 模型 ===
n_actions = train_env.action_space.shape[-1]
action_noise = NormalActionNoise(
    mean=np.zeros(n_actions),
    sigma=0.1 * np.ones(n_actions),
)

model = TD3(
    "MlpPolicy",
    train_env,
    learning_rate=3e-4,
    buffer_size=1_000_000,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    policy_delay=2,
    target_policy_noise=0.2,
    target_noise_clip=0.5,
    action_noise=action_noise,
    learning_starts=10_000,
    tensorboard_log="./runs/td3_ant_gait/",   # ← 路徑請依觀察到的目錄結構調整
    verbose=1,
    seed=0,
)

# === 訓練 ===
model.learn(
    total_timesteps=1_000_000,
    callback=[GaitMonitorCallback(), VideoCallback(eval_interval=50_000)],
    progress_bar=True,
)

model.save("./runs/td3_ant_gait/final_model")
```

---

## TensorBoard 自訂 Callback

SB3 的 TD3 已內建記錄 critic loss、actor loss、Q-value 等基本指標。需要額外加上 wrapper 的 reward components 與 gait 指標：

```python
class GaitMonitorCallback(BaseCallback):
    """記錄 reward 分解、步態指標到 TensorBoard。"""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_components = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "reward_components" in info:
                comp = info["reward_components"]
                self.logger.record_mean("gait/r_forward",   comp["forward"])
                self.logger.record_mean("gait/r_alive",     comp["alive"])
                self.logger.record_mean("gait/r_ctrl",      comp["ctrl"])
                self.logger.record_mean("gait/r_gait",      comp["gait"])
                self.logger.record_mean("gait/r_posture",   comp["posture"])
                self.logger.record_mean("gait/x_velocity",  comp["x_velocity"])
                self.logger.record_mean("gait/torso_z",     comp["torso_z"])

            if "foot_contacts" in info:
                contacts = info["foot_contacts"]
                self.logger.record_mean("contacts/FL", float(contacts[0]))
                self.logger.record_mean("contacts/FR", float(contacts[1]))
                self.logger.record_mean("contacts/BL", float(contacts[2]))
                self.logger.record_mean("contacts/BR", float(contacts[3]))
                # 對角線同步指標
                diag_sync = 1 - 0.5 * (
                    abs(contacts[0] - contacts[3]) + abs(contacts[1] - contacts[2])
                )
                self.logger.record_mean("contacts/diagonal_sync", float(diag_sync))
        return True
```

### 預期會看到的 TensorBoard 曲線

| 指標 | 健康的樣子 |
|------|----------|
| `gait/x_velocity` | 收斂到 ~1.0（目標速度） |
| `gait/r_ctrl` | 數值絕對值由大變小（動作變省力） |
| `gait/r_gait` | 持續上升並趨穩 |
| `gait/torso_z` | 接近 0.6 並穩定 |
| `contacts/diagonal_sync` | 收斂到接近 1.0（對角線同步） |
| `contacts/FL`, `FR`, `BL`, `BR` | 在 step 內呈規律振盪（方波） |

---

## 中途錄影 Callback

每隔固定 steps 在獨立環境跑一個 evaluation episode 並錄影：

```python
from gymnasium.wrappers import RecordVideo

class VideoCallback(BaseCallback):
    def __init__(self, eval_interval: int = 50_000, video_root="./videos/"):
        super().__init__()
        self.eval_interval = eval_interval
        self.video_root = video_root

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_interval == 0:
            self._record_episode(step=self.num_timesteps)
        return True

    def _record_episode(self, step: int):
        eval_env = gym.make("Ant-v5", render_mode="rgb_array")
        eval_env = RealisticGaitWrapper(eval_env)
        eval_env = RecordVideo(
            eval_env,
            video_folder=f"{self.video_root}/step_{step:07d}",
            name_prefix=f"eval_step_{step}",
            episode_trigger=lambda ep: True,
        )

        obs, _ = eval_env.reset()
        total_reward, ep_len = 0.0, 0
        done = False

        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            total_reward += reward
            ep_len += 1
            done = terminated or truncated

        eval_env.close()
        self.logger.record("eval/episode_return", total_reward)
        self.logger.record("eval/episode_length", ep_len)
        if self.verbose:
            print(f"[Step {step}] Eval return: {total_reward:.1f}, length: {ep_len}")
```

### 預期影片資料夾結構

```
videos/
├── step_0000000/      # 訓練初期，隨機動作
├── step_0050000/      # 開始有方向感
├── step_0200000/      # 步態雛形浮現
├── step_0500000/      # 步態成形
└── step_1000000/      # 收斂後的步態
```

---

## ⚠️ 風險警示：Attractor Basin 失敗模式

這個 reward shaping 設計有一個已知失敗模式：**「靜止比走路更賺」的吸引盆**。

當 `forward_reward` 是負的（懲罰偏離目標速度），而 `alive_reward` 是正的常數時，agent 可能會發現：
- 站著不動 → `x_velocity ≈ 0` → `r_forward = -1.0`，但 `r_alive = +1.0`，`r_ctrl ≈ 0`
- 走起來 → `r_forward ≈ 0`，但 `r_ctrl` 變很負（動作大），`r_gait` 在學會之前是 ~0

這會讓 agent 收斂到「站著不動」的 local optimum，而不是學步態。

### 偵測訊號

訓練到 200k steps 時，若 TensorBoard 顯示：
- `gait/x_velocity` 收斂到接近 0
- `contacts/diagonal_sync` 沒有規律振盪
- 整體 reward 卡在某個高原不動

→ 觸發 fallback 策略。

### Fallback 策略（按優先順序嘗試）

**Fallback 1：降低 alive_reward 到 0.0**
```python
RealisticGaitWrapper(env, alive_weight=0.0, ...)
```
讓站著不動失去誘因。

**Fallback 2：減弱 ctrl_weight 從 5.0 → 2.0**
讓 agent 在學習初期敢於用力探索。

**Fallback 3：採用「正向 forward_reward」設計**

把懲罰式速度 reward 換成獎勵式：
```python
# 原本：r_forward = -abs(x_vel - target_speed)
# 改為：在 [0, target_speed] 區間獎勵速度，超過 target_speed 不再加分
r_forward = 2.0 * min(x_vel, self.target_speed)
```
這樣靜止時 `r_forward = 0` 而非負值，破壞 attractor basin。

**Fallback 4：完全放棄 wrapper，回到原始 reward 但加重 contact_cost**
```python
env = gym.make("Ant-v5",
    healthy_reward=0.0,
    forward_reward_weight=1.0,
    ctrl_cost_weight=0.5,
    contact_cost_weight=5e-3,  # ← 從 5e-4 提高 10 倍
)
```
這是最保守的選項，仍能降低慣性甩動但不改變 reward 結構。

---

## 訓練監控檢查點

| Step | 應該看到 | 不對勁的訊號 |
|------|---------|------------|
| 50k  | reward 開始上升、agent 不再隨機抽搐 | 完全靜止 → 立刻 fallback |
| 200k | x_velocity 開始接近 target_speed | diagonal_sync 沒有振盪 → fallback |
| 500k | 步態大致成形，影片可看出走路 | reward 突然崩潰 → 檢查 Q-value 是否爆炸 |
| 1M   | 步態穩定、reward 趨穩 | 仍在抽搐 → 加重 ctrl_weight |

---

## 最終輸出清單

`03_train_td3.py` 跑完後應該產出：

1. **TensorBoard logs**：`./runs/td3_ant_gait/`（路徑依目錄結構調整）
2. **影片資料夾**：`./videos/step_NNNNNNN/`，至少 5 個時間點的 eval 影片
3. **模型檔**：`./runs/td3_ant_gait/final_model.zip`
4. **訓練終端輸出**：能完整看到 eval episode return 的進展

---

## 實作順序建議

1. **先觀察專案目錄結構**，回報觀察結果與計劃的輸出位置
2. **驗證 Ant-v5 的腳 body 名稱**：寫一個小 script `print(env.unwrapped.model.body(i).name for i in range(model.nbody))` 確認名稱對得上
3. **單獨測試 RealisticGaitWrapper**：跑 100 steps 隨機動作，印出 reward components 確認數值合理
4. **跑 50k steps 短訓練**：先確認沒有 NaN、訓練曲線正常上升
5. **正式跑 1M steps 訓練**，背景執行，邊跑邊看 TensorBoard
6. **若 200k 仍卡住** → 啟動 fallback 策略

---

## 注意事項

- 使用 Ant-**v5**，不是 v4
- 訓練時 `learning_starts=10_000`，給 buffer 累積基本經驗
- `policy_delay=2`、`target_policy_noise=0.2`、`target_noise_clip=0.5` 是 TD3 原始論文設定，不要改
- 不要關掉 `progress_bar`，方便監控
- GPU 訓練 1M steps 約 2–4 小時，請預留足夠時間
