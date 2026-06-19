# Changelog

所有重要的專案變更都會記錄在此檔案。

格式參考 [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)。
每次 commit 或完成一個實驗階段後更新。

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
