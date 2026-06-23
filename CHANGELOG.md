# Changelog

所有重要的專案變更都會記錄在此檔案。

格式參考 [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)。
每次 commit 或完成一個實驗階段後更新。

---

## [2026-06-23]

### 新增
- `tools/gait_wrapper_12a.py` + `12a_train_td3.py`：依 `markdown/12a_train_td3_spec.md` 實作 v12a，修補 12 在 25k 步後退化的對齊缺口——新增 `antiphase_bonus_weight`（疊加 anti_phase 直接訊號，受 speed gate 保護）、啟用既有 `smooth_weight`/`tilt_weight`（保護 03 的 jerk/uprightness 強項）。從 03 final_model 接續 finetune。smoke test（5k 步）已驗證可載入 03 checkpoint、reward components 含 `antiphase_bonus`、TensorBoard 含 `gait/r_antiphase_bonus`/`r_smooth`/`r_tilt` 三個新欄位
- `output/02train_td3_reward_formula.html`：整理 `02train_td3.py` reward 修正的兩階段目標（邊界懲罰+動作正則化、修正站著不動 attractor）與完整計算公式匯出報告
- `output/03_vs_12at25k_raw_episodes.csv`：03 與 12@25k 各 10 個 deterministic episode 的逐筆原始指標（非彙整 mean/std），供外部分析軟體使用

### 變更
- `tools/gait_train.py`：`make_env` / `train` / `finetune` / `EvalScorecardCallback` 新增可選參數 `wrapper_cls`（預設仍為 `gait_wrapper_03.RealisticGaitWrapper`，向後相容），讓 12a 可指定載入 `gait_wrapper_12a`；`GaitMonitorCallback` 新增對 `antiphase_bonus` 鍵的條件式 TB logging（無此鍵的舊 wrapper 不受影響）
- `.gitignore`：修正 `output/**/replay_buffer*` 規則 —— 原本寫成行內註解（`pattern   # 說明`），但 gitignore 不支援行內註解，整行被當成一個 pattern 字面值，導致規則完全沒生效。改成獨立註解行 + 獨立 pattern 行，已用 `git check-ignore` 驗證生效（本機 12a 訓練產生的 1.6GB `replay_buffer.pkl` 確認被正確排除）

### 實驗結論：v12a 50k 步驗證（本機跑，未達標）
- 在本機（CPU）跑了三組 12a 設定各 50k 步，皆未達 spec 的 anti_phase≥0.27 必達門檻：
  1. `antiphase_bonus_weight=0.5, gait_speed_gate=0.3`（spec 原始配方）：anti_phase 峰值 **0.247 @40k**，三組最佳
  2. `antiphase_bonus_weight=1.0, gait_speed_gate=0.3`（spec 失敗診斷建議一）：anti_phase 峰值降到 0.231——**加重 bonus 反而更差**，因為 legacy 主項的 `diag1/diag2_sync` 對「四腳同步」有結構性偏好（站著也偏高），加重 bonus 後 diagonal_sync 被推更高（0.665→0.70+）、anti_phase 反被擠壓
  3. `antiphase_bonus_weight=0.5, gait_speed_gate=0.15`（spec 失敗診斷建議二）：anti_phase 峰值僅 0.230，比設定 1 更差
- 拿設定 1（三組最佳）與 12@25k 做統計對照（10-episode mean±std）：anti_phase 0.247±0.010 vs 12@25k 的 0.243±0.007，**差異在 1 個標準差內、不具統計意義**；而 12@25k 在 mean_speed（0.980 vs 0.971）、speed_error（0.115 vs 0.122）、diagonal_sync（0.678 vs 0.665）、transport_cost（0.966 vs 0.989）都顯著贏過 v12a。
- **結論：v12a（在 legacy 主公式上疊加 anti_phase bonus）未能勝過 12@25k，12@25k 仍是目前最佳模型**。這不是參數沒調好的問題——spec 自己列的兩個失敗診斷方向都驗證為更差，顯示「疊加 bonus」這個設計本身在這個權重/gate 範圍內打不穿 anti_phase 天花板。下一步應考慮 spec 第 8 節規劃的 v12b（EMA gate）或重新設計 reward 結構，而非繼續在這個方向微調。
- 三組訓練的 checkpoint 皆未優於現有最佳模型，不納入版控（同 13 的失敗 artifact 慣例）；完整數據見 `output/03_vs_12at25k_raw_episodes.csv`。

## [2026-06-22] — 同步遠端並統一 gait_wrapper 命名

### 變更
- `tools/gait_wrapper.py` 重新命名為 `tools/gait_wrapper_03.py`(Python 模組名不可以數字開頭 import,故不採 `03gait_wrapper.py`),同步修正 `03train_td3.py`、`03test_td3.py`、`04train_td3_sbx.py`、`tools/gait_train.py` 的 import 路徑

## [2026-06-21]

### 新增
- `03test_td3.py`：對應 `03train_td3.py` 的推論/模擬腳本，載入 SB3 `TD3.load()` 模型並套用相同的 `RealisticGaitWrapper` 與環境參數，於 MuJoCo human 模式下視覺化步態，支援 `--checkpoint` 與 `--episodes` 參數
- `03train_td3.py`：以 Stable-Baselines3 TD3 + `tools/gait_wrapper_03.py`(`RealisticGaitWrapper`)訓練步態導向的 Ant-v5,獎勵四足交替著地(trot)、限制速度上限、強懲罰大幅動作,取代預設 reward 訓出的「慣性甩動」高速移動,規格見 `markdown/03_train_td3_spec.md`

## [2026-06-23] — multi-seed 穩健性：progress 線（07）跨 seed 穩定

### 新增
- `14`（= 07 加 `SEED` env override，多 seed 驗證 progress 線）：seed 1 @1M **複現 seed 0**——ep_len 1000、mean_speed 0.947（seed0=0.958）、jerk 0.143、CoT 3.01、diagonal_sync 0.588、stationary 0.003，幾乎逐項一致。**對比 03 seed1 是 15 步 fast-fall → progress 線跨 seed 穩定（初步坐實，seed 2 待補）。** 結果拉回 `output/14multiseed/seed_1/`、影片 `gait_videos/14_seed1/`
- `15`（小修版 03：ctrl 漸進排程）：seed 1/2 @400k **fast-fall 跨 seed 一致修好**——ep_len 1000（原版 03=15 步秒摔）、stationary 0.03–0.07，但速度被 ctrl=5 壓在 ~0.58；seed1 續 600k speed 仍 0.61 且變不穩 → 加 forward 拉力
- `15-A`（15 + `FORWARD_WEIGHT` env）：forward_weight 1.2 → seed1 1M speed 0.764（補洞中），再 resume → **forward_weight 1.8 @eff-1.4M 解決速度**：**mean_speed 0.913 / speed_error 0.122（勝 03 的 0.140）/ jerk 0.052 / CoT 1.62 / diagonal_sync 0.744（勝 03）/ upright 0.991 / ep_len 1000 / stationary 0.002**。加速拉力反而讓 jerk 0.081→0.052、CoT 2.19→1.62（走更順更省力）。**deviation 線目前最佳、保留 03 自然感基因，下一步跑 seed2 驗證跨 seed 穩健性**
- `tools/gait_train.resume()`：載入 model + replay buffer 續訓同 reward（免重跑）。15-A 多次續訓（400k→1M→1.4M）全靠此機制，buffer 從 39 萬/99 萬/100 萬次更新接續

---

## [2026-06-22] — 步態量化評估管線 + 04→05→06→07 reward 迭代

### 新增
- `tools/gait_train.py`：抽出共用訓練骨架（`make_env` / `GaitMonitorCallback` / `EvalScorecardCallback` / `train`），各版改成只放 `WRAP_KWARGS` 的薄設定檔。影片每 200k 錄一支、TB 永遠開（四/五方比較需各版一致）
- `tools/eval_scorecard.py`：載入 SB3 模型跑 N 個 deterministic episode，用 `gait_metrics` 算 scorecard 平均±標準差（與訓練 reward 無關，可公平比較）。用法 `python -m tools.eval_scorecard <name> <model_path> [n_episodes]`
- `05train_td3.py`：融合版（加重 `gait_weight=3.0`/`posture=1.5`/`tilt=0.8`/`smooth=0.25`），10-ep eval：anti_phase 0.326、regularity 0.355 為各版最佳，但仍抖（jerk 0.158）且超速（x_vel 1.37）
- `06train_td3.py`：`forward_gate_shape="tent"` 修超速 + `smooth=0.40`/`ctrl=1.5` 加重平滑。10-ep eval：mean_speed 0.83（修好超速）、jerk 0.087（04 的 1/4）、uprightness 0.989 反超 03，最接近 03 的 shaped 版；代價是 anti_phase 0.212、diagonal_sync 0.552 被磨柔
- `07train_td3.py`：在 06 基礎上「把對角踏步銳利度拉回」——`gait_weight 3.0→4.5`（anti_phase 是乘法 gate，加重直接放大對角交替）、`smooth 0.40→0.30`、`ctrl 1.5→1.2`，其餘沿用 06。10-ep eval：**speed_error 0.100（全場最佳，連 03 都贏）、mean_speed 0.958（最準）**、anti_phase 0.212→0.267（回升超過 03），代價是 jerk 0.087→0.120、diagonal_sync 仍卡 0.544。各版量化見 `output/03_vs_04_comparison.html`（已更新為五方）
- `08train_td3.py`：在 07 基礎上 `intra_weight 0.25→0.35`（加重 antiphase_gated 的「同對角同步」拉 diagonal_sync）+ `smooth 0.30→0.35`。10-ep eval：**regularity 0.380 / anti_phase 0.358 / uprightness 0.993 三項總冠軍**（最像教科書 trot），但**關鍵發現：加重 intra 把 anti_phase 大幅拉起卻對 diagonal_sync 幾乎無效（0.544→0.575）**，且踏步更賣力 → speed_error 0.100→0.168、jerk/CoT 退一階
- `09train_td3.py`：現有設計極限測試（smooth 0.50 / ctrl 2.0 / intra 0.40）。未跑完即中止——改走結構性路線（10）。保留設定檔作軌跡
- `10train_td3.py`：★ 結構性換設計 ★ `gait_mode="legacy"` × `forward_gated`，假設「03 觀感來自 legacy 步態公式」。10-ep eval **反證此假設**：jerk 0.216 / CoT 4.91 / regularity 0.128 皆 shaped 版最差，diagonal_sync 僅 0.606。**真正教訓：03 的平滑/省力來自 `ctrl_weight=5.0` + additive，不是步態公式**——下一步主攻 ctrl_weight
- `output/03_vs_04_comparison.html`：03–10 七方量化比較報告（含 10 反證實驗與「下一步主攻 ctrl」的新方向）
- `11train_td3.py`：03 + 唯一變因 gait_speed_gate=0.3，從零訓練。300k 探針**秒摔（ep_len 13）**——證明站著 gait 分數是 03 的學習鷹架（非單純缺點）：gate 後站著從 +1.6 高峰變平地 0，配 ctrl=5/deviation 致 cold-start 塌進「快速摔倒」attractor
- `12train_td3.py` + `tools/gait_train.py` 新增 `finetune()`：checkpoint curriculum 探針。載入 03 final_model→換 gate reward→清 buffer→learning_starts=0→lr 1e-4/noise 0.03→微調 120k。10-ep eval **保住 03 觀感**：ep_len 1000、jerk 0.042（≤0.05✅）、CoT 1.09（≤1.5✅）、diagonal_sync 0.689/upright 0.985≈03，僅速度 0.94→0.88（差 0.9 一點）。**結論：鷹架只在學會走前需要，兩階段 curriculum 可行**。**★ 12@25k checkpoint 是目前最佳模型 ★**：10-ep eval ep_len 1000、mean_speed 0.983（>03 的 0.94）、jerk 0.037、CoT 0.960（<03 的 1.02）、diagonal_sync 0.678、anti_phase 0.245、upright 0.990——五項約束（=1000 / ≥0.9 / ≤0.05 / ≤1.5 / ≥0.65）全達標，且 speed/CoT/anti_phase/upright 都贏 03，站著 attractor 已用 gate 拿掉。final(120k) 速度漂到 0.88 是策略漂移，25k 為漂移前甜蜜點。**配方定案：03 → curriculum fine-tune（gate 0.3 / lr 1e-4 / noise 0.03）取 ~25k 早停**，下一步多 seed 驗證穩健性
- `13multiseed_td3.py` + `tools/gait_metrics.stationary_fraction`：multi-seed 穩健性驗證（seed 0 既有 + 新 seed 1/2，各跑 stage1 原版 03 1M → stage2 gate curriculum 50k，預先固定 stage2@25k 為主要結果）。**🚨 決定性結果：03 成功率僅 1/3** —— seed 0 ep_len 1000/x_vel 0.94 會走，但 **seed 1（14.7）、seed 2（16.2）原版 03 同設定卻塌進「快速摔倒」（死 ~15 步）**；超參數經逐項比對與 `03train_td3.py` 完全相同，故為真實高變異而非設定走樣。**結論：03 的會走是 seed-0 運氣，非穩定配方；12@25k 冠軍建在此脆弱起點上（seed 1/2 curriculum 因載入會摔的 03 而同樣崩）。根因＝03 的 `forward_mode="deviation"`（站著罰 −1）正是 `ant_v5_attractor_fix.md` 的 fast-fall attractor 來源。下一步應改用穩定的 `progress` 線並對其做 multi-seed 驗證，而非建在脆弱的 03 上。**（seed 1/2 模型留在 pc106，失敗 artifact 不入版控）
- `gait_videos/`（RLAP 根目錄，非版控）：整理好的各版 eval 影片，按版本分子資料夾、改檔名 `<版>_step_<step>.mp4`

### 變更
- `tools/gait_wrapper.py`：`RealisticGaitWrapper` 新增 `forward_gate_shape`（`cap` 到目標即滿分／`tent` 太快也遞減壓超速）與 `intra_weight`（antiphase_gated 的同對角同步權重，base 自動補成滿分=1）。兩者預設（`cap` / 0.25）維持既有行為
- `04train_td3.py`：收斂為「會走」的溫和 `forward_gated` 設定，改成薄設定檔（套用 `tools/gait_train.train`）

---

## [2026-06-22] — 04 reward 迭代到「會走」+ pc106(GB10)訓練環境 + SBX 探索

### 變更
- `04train_td3.py`：改成 **env-var 驅動的 runner**（reward 旋鈕 / `OUTPUT_DIR` / `MAX_TIMESTEPS` / `VIDEO_INTERVAL` 等皆可用環境變數覆寫），每個 reward 調參用同一支腳本跑、不另立編號檔。經三次失敗模式後（站著不動 / 原地踏步 / 快速摔倒），**預設調到會走的配方**：`reward_structure="forward_gated"` + `gait_mode="antiphase_gated"` + 溫和懲罰（`ctrl_weight=0.5`、posture 0.5、smooth 0.02、tilt 0.2）+ 小 `alive_weight=0.5` 底分。300k 驗證：speed_err 0.94→0.22、x_velocity→~1.2、episode 全程 1000 不摔、return~1600。錄影與數值 scorecard 解耦（`VIDEO_INTERVAL`）以加速迭代；保留多環境（SubprocVecEnv）路徑但預設單環境（實測多環境只快 ~13%，瓶頸在 SB3 單執行緒迴圈）
- `tools/gait_wrapper.py`：`RealisticGaitWrapper` 新增 `gait_mode="antiphase_gated"`（步態 reward 用 anti_phase 當乘法 gate，靜態拿不到分）、`reward_structure="forward_gated"`（步態 bonus 以前進為閘門，不前進整個正向 reward 歸零）、`forward_weight` 參數。**預設值維持 03 行為，向後相容**

### 新增
- `04train_td3_sbx.py`：SBX(SB3 + Jax)後端版,測試 Jax 能否加速。結論:**單一 CPU MuJoCo 環境下不會更快**(SBX-GPU 185 fps < SB3 211 < SBX-CPU 261;瓶頸是環境步進與單執行緒迴圈,GPU 全程 ~12%)。真正的 5-10x 需 GPU 向量化環境(MJX/Brax),非換後端可得

### 備註
- 訓練機 **pc106**(`aitopatom-186a`,aarch64 + NVIDIA GB10 Blackwell):venv `~/rlap_env`(torch 2.12.1+cu130 / sb3 2.9.0 / gymnasium 1.3.0 / mujoco 3.9.0 / jax 0.10.2 cuda13);離屏錄影需 `MUJOCO_GL=egl`。GPU 在 GB10 上可運算(CUDA 13 支援 Blackwell),但對 TD3+MlpPolicy 加速有限

---

## [2026-06-21] — 04 步態品質指標優化（尚未訓練）

### 新增
- `03test_td3.py`：對應 `03train_td3.py` 的推論/模擬腳本，載入 SB3 `TD3.load()` 模型並套用相同的 `RealisticGaitWrapper` 與環境參數，於 MuJoCo human 模式下視覺化步態，支援 `--checkpoint` 與 `--episodes` 參數
- `tools/gait_metrics.py`：步態品質量化指標共用模組（`anti_phase`、`diagonal_sync`、`uprightness` per-step；`action_jerk`、`transport_cost`(CoT 代理)、`contact_regularity`(自相關週期性 0..1) per-episode），供 wrapper / 訓練 callback / 未來 eval 腳本共用，已用純 numpy 驗證數值
- `04train_td3.py`：接續 03 的步態品質優化實驗。改用 `gait_mode="antiphase"`（步態 reward 以對角線反相為主導，修正 03 legacy 公式「站著 r_gait=1.6 反而高於走路」的隱性站著 attractor）、`forward_mode="progress"`（速度 reward 改 `max(0,min(x_vel,target))`，站著=0 而非 -1.0，符合 `ant_v5_attractor_fix.md` 結論）、新增 jerk 與軀幹直立懲罰；callback 擴充為每個 eval interval 跑 deterministic episode 算整段 scorecard 寫入 TensorBoard，並存中間 checkpoint（03 只有 final_model）。規格見 `markdown/04_train_td3_spec.md`
- `markdown/04_train_td3_spec.md`：04 規格與 03 指標卡關的診斷（含「站著加分」bug 的驗證數據）

### 變更
- `tools/gait_wrapper.py`：`RealisticGaitWrapper` 新增 `gait_mode`、`forward_mode`、`smooth_weight`、`tilt_weight` 四個參數，並開始記錄 `smooth`/`tilt`/`uprightness`/`anti_phase` reward 分量。**全部預設值維持 03 行為**，03 可完全重現；僅 `04train_td3.py` 啟用新設定

### 待辦
- 與隊友對齊「站著扣分」機制的整合方式（該機制不在版控裡），再跑 1M 正式訓練

### 新增（03，續）
- `03train_td3.py`：以 Stable-Baselines3 TD3 + `tools/gait_wrapper_03.py`(`RealisticGaitWrapper`)訓練步態導向的 Ant-v5,獎勵四足交替著地(trot)、限制速度上限、強懲罰大幅動作,取代預設 reward 訓出的「慣性甩動」高速移動,規格見 `markdown/03_train_td3_spec.md`
- `01test_td3.py`:對應 `01train_td3.py` baseline 模型的推論腳本
- `markdown/ant_v5_attractor_fix.md`:記錄 `02train_td3.py` reward shaping 的除錯過程 —— Ant-v5 預設 `healthy_reward=1.0` 讓「站著不動」變成零風險 attractor,5 次實驗驗證後改用拿掉 healthy_reward + 提高 contact_cost_weight + 還原 forward_reward_weight 解決
- `.gitignore`:排除 `.DS_Store`、`__pycache__/`、`output/**/videos/`(評估錄影檔案過大,不納入版控)

### 變更
- `02train_td3.py`:依 `markdown/ant_v5_attractor_fix.md` 調整 reward shaping —— `FORWARD_REWARD_WEIGHT` 還原為 1.0、`HEALTHY_REWARD` 降為 0.1、`CONTACT_COST_WEIGHT` 提高 10 倍至 5e-3;邊界懲罰改為 `SOFT_RADIUS`(漸增懲罰)+ `MAX_RADIUS`(強制截斷)兩段式;出界視為 terminal 存入 replay buffer,讓 critic 學到「出界=沒有未來」
- `02test_td3.py`:預設 checkpoint 路徑修正為 `output/02train_td3/`(原本指向已不存在的 `output/td3_ant_v5/`)

---

## [2026-06-19 17:20]

### 變更
- 整理 `output/` 目錄結構，統一依腳本編號命名：`output/01train_td3/`（對應 `01train_td3.py` baseline）、`output/02train_td3/`（對應 `02train_td3.py` reward shaping）
- `01train_td3.py`、`02train_td3.py`：`OUTPUT_DIR` 同步改為對應的新路徑

### 移除
- `output/td3_ant/`：早期 Ant-v4 孤兒訓練資料，與現行 01/02 腳本不對應，已刪除

---

## [2026-06-19 16:10]

### 新增
- `01train_td3.py`：保留 reward shaping 修改前的 `02train_td3.py` 版本（對應 commit `2e3433a`），作為 baseline 備份

---

## [2026-06-19 15:52]

### 新增
- `markdown/ant_reward_modifications.md`：Ant reward shaping 修改規格文件（邊界懲罰、動作正則化）

### 變更
- `02train_td3.py`：依規格文件加入 reward shaping，並調整為 Ant-v5 寫法
  - 邊界位置懲罰：用 `info["distance_from_origin"]` 取代 v4 寫法的 `next_obs[0:2]`（v5 預設不把 x,y 放進 observation），超過 `MAX_RADIUS=8.0` 線性懲罰並設 `truncated=True`
  - 動作正則化：新增動作幅度懲罰（`ACTION_PENALTY_WEIGHT=0.5`）與相鄰動作差異懲罰（`ACTION_DIFF_PENALTY_WEIGHT=0.1`），新增 `prev_action` 追蹤
  - 略過額外 contact 懲罰：Ant-v5 預設已將 `contact_cost` 內建於 reward，不重複懲罰

---

## [2026-06-18 01:00]

### 新增
- `02test_td3.py`：載入訓練好的 TD3 checkpoint 並開啟 MuJoCo 視覺化動畫，支援 `--checkpoint` 與 `--episodes` 參數
- 完成 1M steps 訓練，最終 eval reward 達 **4090~5183**（目標範圍 3000~6000 ✅）

### 變更
- `02train_td3.py`：環境升級 `Ant-v4` → `Ant-v5`（obs 27 → 105 維）；output 路徑改為 `output/td3_ant_v5/`

---

## [2026-06-18 00:00]

### 新增
- `02train_td3.py`：TD3 主訓練腳本（MuJoCo Ant-v4），含 random warmup、訓練迴圈、eval、TensorBoard logging、checkpoint 存至 `output/td3_ant/`
- `tools/__init__.py`：將 tools/ 設為 Python package
- `tools/replay_buffer.py`：ReplayBuffer，支援 add() 與 sample()（回傳 GPU tensor）
- `tools/networks.py`：Actor（tanh 輸出 × max_action）與 Twin Critic（雙 Q-network）
- `tools/td3_agent.py`：TD3Agent，實作 Twin Critics、Delayed Policy Update、Target Policy Smoothing、soft target update、save/load

---

## [2026-05-22 13:12]

### 新增
- `CHANGELOG.md`：工作日誌，依 Keep a Changelog 格式記錄所有變更

### 變更
- `README.md`：專案結構補充 CHANGELOG.md 說明；協作注意事項新增更新日誌提醒
- `CLAUDE.md`：MANDATORY REQUIREMENTS 新增「每次 commit 後必須更新 CHANGELOG」規則；專案結構補充 CHANGELOG.md

---

## [2026-05-22 13:06]

### 新增
- `README.md`：專案門面說明，包含快速開始、實驗目錄、命名慣例與協作注意事項

---

## [2026-05-19 15:14]

### 新增
- `01RL_Lab.py`：CartPole-v1 隨機動作入門，示範 MDP 基本迴圈（observation / action / reward / terminated）
- `CLAUDE.md`：AI agent 協作指引，定義絕對禁止事項、命名規則、資料夾結構
- `tools/`：共用工具資料夾（初始化）
- `output/`：實驗輸出資料夾（初始化）
- `.vscode/settings.json`：conda 環境設定

---

<!-- 新增記錄時複製以下模板到最上方 -->
<!--
## [YYYY-MM-DD HH:MM]

### 新增
- 新功能、新腳本、新實驗

### 變更
- 修改既有腳本的邏輯或參數

### 修正
- 修正 bug 或錯誤

### 移除
- 刪除的檔案或功能
-->
