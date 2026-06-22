# Claude Code 任務規格：TD3 + Ant-v5 步態「品質指標」優化（接續 03）

## 任務總覽

03 已能訓出會走的 Ant，但用一組量化指標檢視時，步態品質卡住：
`diagonal_sync ≈ 0.72`、四腳著地比例低且左右不對稱、`r_gait` 只到滿分的 65%。
04 的目標**不是再壓 reward 數字**，而是讓「衡量步態好壞的指標」真正進步：
對角線交替的規律性、動作平滑度、姿態穩定度、能耗效率。

> 沿用 03 的環境、TD3 超參數、檔案/輸出命名規律，不重寫架構。
> 輸出位置：`output/04train_td3/`。共用指標邏輯放 `tools/gait_metrics.py`。

---

## 為什麼 03 的指標卡住：診斷

### 1. 步態 reward 是「逐幀靜態」的，獎勵站著不動

03 的 `r_gait = 2.0·(0.4·diag1_sync + 0.4·diag2_sync + 0.2·cross)`，其中
`diagN_sync = 1 - |腳A - 腳B|`。四腳全部踩死時 `diag1_sync = diag2_sync = 1`，
所以**站著的 r_gait = 1.6，比實際走路學到的 1.30 還高**（已用程式驗證）。

| 行為 | 03 legacy 每步 reward（約略） |
|---|---|
| 站著不動 | forward(−1.0) + alive(+1.0) + gait(**+1.6**) ≈ **+1.6** |
| 走路(trot) | forward(0) + alive(+1.0) + gait(+1.3) + ctrl(−0.7) ≈ **+0.8** |

→ **站著每步比走路還賺**。03 沒崩成站著，只是因為「完美靜止」在 Ant 裡不好維持
（探索噪聲一直擾動），很僥倖。這與 `markdown/ant_v5_attractor_fix.md` 的血淚教訓衝突。

### 2. `prev_contacts` 存了卻沒用——沒有任何「時間上交替」的訊號

步態的本質是週期性交替，但 03 的 reward 只看單一時間點，無法區分「規律踏步」與「四腳亂抖」。

### 3. 速度 reward 是 speed penalty

`r_forward = −|x_vel − target|` 站著時 = −1.0。`ant_v5_attractor_fix.md` 實測這類
speed/stillness penalty「會逼出快速摔倒擺爛」，結論是要移除、改靠「站著沒收入」。

---

## 改動：對應每一個要進步的指標

| 指標 | 改動 | 原理 |
|---|---|---|
| diagonal_sync / 週期性 | `gait_mode="antiphase"`：`r_gait = 2.0·(0.2·intra1 + 0.2·intra2 + 0.6·anti_phase)`，`anti_phase = \|mean(FL,BR) − mean(FR,BL)\|` | 反相為主導，靜態姿勢 = 0，逼出真正交替；站著的 r_gait 由 1.6 → 0.8 |
| 站著 attractor | `forward_mode="progress"`：`r_forward = max(0, min(x_vel, target))` | 站著 = 0（非負），走路才是唯一正收益；符合 02 已驗證的 `healthy_reward=0` 思路 |
| 動作平滑 / jerk | `smooth_weight=0.1`：懲罰 `‖aₜ − aₜ₋₁‖²` | 直接壓「抽搐 / 慣性甩動」 |
| 姿態穩定 | `tilt_weight=0.5`：懲罰軀幹偏離垂直（四元數算直立度） | 03 只管高度 z，沒管翻不翻 |

搭配 `alive_weight=0.0`（歸零，徹底消除站著收入，照 02 已驗證的 `healthy_reward=0` 做法）後，
每步 reward：**站著 ≈ +0.8、走路 ≈ +1.7**（gap 由 03 的 −0.8 翻成 +0.9，走路明顯勝出；
站著仍為正、非負，不會觸發「快速摔倒擺爛」）。

> ⚠️ 向後相容：以上全是 `RealisticGaitWrapper` 的新參數，預設值維持 03 行為，
> 03 仍可完全重現。只有 `04train_td3.py` 打開新設定。

---

## 評估：把「指標進步」看出來

03 只存 `final_model`、且只在 TB 記訓練中帶噪聲的滑動平均，看不到乾淨的演進。04 補上：

1. `EvalScorecardCallback`：每 `EVAL_INTERVAL` 跑一個 **deterministic** episode，
   錄影並把整段 scorecard 寫進 TB —— `eval/speed_error`、`eval/transport_cost`(CoT 代理)、
   `eval/action_jerk`、`eval/contact_regularity`(自相關週期性 0..1)、`eval/diagonal_sync`、
   `eval/anti_phase`、`eval/uprightness`、`eval/episode_return`。
2. `CheckpointCallback`：每 `CHECKPOINT_FREQ` 存中間模型，可回頭對每個 checkpoint 重算 scorecard、畫演進曲線。

### scorecard 各軸健康的樣子

| 指標 | 健康值 | 抓什麼 |
|---|---|---|
| `eval/anti_phase` | 隨訓練上升、趨穩(~0.5–0.7) | 有沒有真的交替踏步 |
| `eval/contact_regularity` | → 接近 1.0 | 步態週期性 |
| `eval/diagonal_sync` | 高且穩 | 對角協調（需搭配 anti_phase 才有意義） |
| `eval/action_jerk` | 由大變小 | 動作平滑 |
| `eval/transport_cost` | 由大變小 | 每公尺省力程度 |
| `eval/speed_error` | → 接近 0 | 速度追蹤準確度 |
| `eval/uprightness` | → 接近 1.0 | 姿態不歪 |

---

## 未決事項（需與隊友確認）

- 隊友提到的「站著扣分」機制不在版控裡。04 目前以 `forward_mode="progress"` +
  `antiphase` 從「移除站著收入」的角度處理，**刻意不疊加額外的負向 stillness penalty**，
  以免重蹈 `ant_v5_attractor_fix.md` 記載的「雙重懲罰 → 快速摔倒擺爛」失敗。
  若要整合隊友的機制，需先確認其公式與位置，並對應調掉 `progress` 避免雙重懲罰。

---

## 實作 checklist

- [x] `tools/gait_metrics.py`：anti_phase / diagonal_sync / uprightness / action_jerk / transport_cost / contact_regularity（已用純 numpy 驗證數值）
- [x] `tools/gait_wrapper.py`：新增 `gait_mode` / `forward_mode` / `smooth_weight` / `tilt_weight`，預設維持 03 行為
- [x] `04train_td3.py`：套用 04 設定 + GaitMonitorCallback 擴充 + EvalScorecardCallback + CheckpointCallback
- [ ] 環境就緒後先跑 50k 短訓練，確認無 NaN、scorecard 數值合理
- [ ] 與隊友對齊「站著扣分」整合方式後再跑 1M 正式訓練
