# Changelog

所有重要的專案變更都會記錄在此檔案。

格式參考 [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)。
每次 commit 或完成一個實驗階段後更新。

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
- `03train_td3.py`：以 Stable-Baselines3 TD3 + `tools/gait_wrapper.py`(`RealisticGaitWrapper`)訓練步態導向的 Ant-v5,獎勵四足交替著地(trot)、限制速度上限、強懲罰大幅動作,取代預設 reward 訓出的「慣性甩動」高速移動,規格見 `markdown/03_train_td3_spec.md`
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
