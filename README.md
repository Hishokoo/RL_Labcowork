# RL_Lab — 強化學習入門實驗室

以 [Gymnasium](https://gymnasium.farama.org/) 為基礎，從零開始學習強化學習（RL）。  
實驗從隨機動作出發，逐步進階到 DQN、Policy Gradient 等演算法。

---

## 學習路線

```
隨機動作 (01) → Q-Learning → DQN → Policy Gradient
```

---

## 快速開始

```bash
# 1. 安裝依賴
pip install "gymnasium[classic-control]"

# 2. 執行第一個實驗
python 01RL_Lab.py
```

> 執行後會彈出視窗，顯示 CartPole 台車即時動畫。

---

## 實驗目錄

| 腳本 | 說明 |
|------|------|
| `01RL_Lab.py` | CartPole 隨機動作入門 — 理解 MDP 基本迴圈（observation / action / reward） |

---

## 專案結構

```
RL_Lab/
├── CLAUDE.md          # AI agent 協作指引（Claude Code 專用）
├── tools/             # 共用工具腳本（各實驗可複用的函式）
├── output/            # 所有輸出統一放這（模型權重、訓練圖表）
│
├── 01RL_Lab.py        # 實驗 01
└── README.md
```

**規則：**
- 模型、圖表等輸出一律放 `output/`，不要放根目錄
- 共用邏輯提取到 `tools/`，避免各腳本間重複程式碼

---

## 腳本命名慣例

新增實驗時請遵循以下格式，方便協作者理解每支腳本的用途：

```
[序號][類型]_[描述].py

範例：
  02train_dqn.py      ← 訓練腳本
  02test_dqn.py       ← 測試 / 推論腳本
  03lab_pg.py         ← 實驗性探索
```

| 前綴 | 意義 |
|------|------|
| `XXtrain_` | 主訓練腳本 |
| `XXtest_`  | 測試 / 推論腳本 |
| `XXlab_`   | 實驗性探索 |

---

## 環境說明

| 項目 | 內容 |
|------|------|
| Framework | Gymnasium |
| 環境 | Classic Control（CartPole-v1 等） |
| Python 管理 | conda |
| IDE | VS Code |

---

## 給協作者的注意事項

- 新增實驗前，先閱讀現有腳本，避免重複造輪子
- 不要建立 `_v2`、`enhanced_`、`new_` 等命名的重複腳本，直接擴充原腳本
- 有共用邏輯請放進 `tools/`
- AI agent 協作請參閱 `CLAUDE.md`
