"""步態品質量化指標（per-step 與 per-episode 共用工具）。

設計原則：reward 數值本身在步態導向訓練裡是被刻意壓低的（見 03 spec），
不能拿來衡量「步態好不好」。因此把真正反映品質的量化邏輯集中在這裡，
讓 RealisticGaitWrapper（reward 計算）、訓練 callback（TensorBoard 監控）、
未來的獨立 eval 腳本都共用同一套定義，避免重複實作導致定義不一致。

座標慣例：四足接觸向量順序固定為 [FL, FR, BL, BR]。
"""
from __future__ import annotations

import numpy as np

# 對角線配對：trot 步態裡 (FL, BR) 同相、(FR, BL) 同相，兩條對角線彼此反相。
DIAG_1 = (0, 3)  # FL, BR
DIAG_2 = (1, 2)  # FR, BL


# ── per-step 指標（單一時間點，wrapper 與 callback 共用）────────────────────────
def diagonal_sync(contacts: np.ndarray) -> float:
    """對角線同步度 0..1：兩條對角線各自「兩腳狀態一致」的平均。

    與 03 callback 既有定義一致，保留作為跨實驗可比的監控指標。
    注意：站著不動（四腳全踩地）時此值 = 1.0，所以它**不能單獨**代表步態好壞，
    必須搭配 anti_phase 一起看。
    """
    s1 = 1.0 - abs(contacts[DIAG_1[0]] - contacts[DIAG_1[1]])
    s2 = 1.0 - abs(contacts[DIAG_2[0]] - contacts[DIAG_2[1]])
    return float(0.5 * (s1 + s2))


def anti_phase(contacts: np.ndarray) -> float:
    """對角線反相度 0..1：兩條對角線「一抬一踏」的程度。

    = |mean(FL,BR) - mean(FR,BL)|。
    - 站著不動 / 四腳同時離地（任何靜態姿勢）→ 0（拿不到分）
    - 真實 trot 的單腳支撐瞬間 → 1
    這是「有沒有真的在交替踏步」的核心訊號，靜態姿勢無法作弊。
    """
    diag1 = 0.5 * (contacts[DIAG_1[0]] + contacts[DIAG_1[1]])
    diag2 = 0.5 * (contacts[DIAG_2[0]] + contacts[DIAG_2[1]])
    return float(abs(diag1 - diag2))


def uprightness(qpos: np.ndarray) -> float:
    """軀幹直立度 0..1：軀幹本體 z 軸在世界座標的垂直分量。

    qpos[3:7] 是 MuJoCo 的軀幹四元數 (w, x, y, z)；
    本體 z 軸的世界 z 分量 = 1 - 2(x² + y²)，完全直立 = 1，翻倒 → 0 或負。
    """
    _, qx, qy, _ = qpos[3], qpos[4], qpos[5], qpos[6]
    return float(1.0 - 2.0 * (qx * qx + qy * qy))


# ── per-episode 指標（整段序列，eval / scorecard 用）──────────────────────────
def action_jerk(actions: np.ndarray) -> float:
    """動作平滑度（jerk）：相鄰動作差的平方和平均，越小越平滑。

    直接量化「抽搐 / 慣性甩動」。actions 形狀 (T, act_dim)。
    """
    actions = np.asarray(actions, dtype=np.float64)
    if len(actions) < 2:
        return 0.0
    diffs = np.diff(actions, axis=0)
    return float(np.mean(np.sum(diffs * diffs, axis=1)))


def transport_cost(actions: np.ndarray, distance: float, eps: float = 1e-3) -> float:
    """控制能耗 / 前進距離（Cost of Transport 代理量），越小越省力。

    無真實質量/重力項，以 Σ‖action‖²（控制努力）對前進距離正規化，
    用途是跨 checkpoint / 跨速度公平比較「每走一公尺花多少力」。
    """
    actions = np.asarray(actions, dtype=np.float64)
    effort = float(np.sum(np.square(actions)))
    return effort / max(abs(distance), eps)


def stationary_fraction(x_vels: np.ndarray, thresh: float = 0.1) -> float:
    """站著比例 0..1：前進速度低於 thresh（m/s）的時間步佔比，越低代表越沒在「站著/原地」。

    用來確認步態不是靠站著/原地踏步達標（與站著 attractor 直接對應）。
    """
    x = np.asarray(x_vels, dtype=np.float64)
    if len(x) == 0:
        return 0.0
    return float(np.mean(np.abs(x) < thresh))


def contact_regularity(contact_seq: np.ndarray, min_lag: int = 5,
                       max_lag: int | None = None) -> float:
    """步態週期性 0..1：四腳接觸序列的自相關主峰平均。

    真實步態是週期性的（接觸序列呈規律方波），亂踏 / 站死則沒有週期。
    對每隻會動的腳，取 lag ∈ [min_lag, max_lag] 範圍內的最大正規化自相關值，
    再對「有在動的腳」取平均。完全靜止（沒有任何踏步）回傳 0。

    contact_seq 形狀 (T, 4)，值為 0/1。
    """
    seq = np.asarray(contact_seq, dtype=np.float64)
    if seq.ndim != 2 or len(seq) < 2 * min_lag:
        return 0.0
    n = len(seq)
    hi = (n // 2) if max_lag is None else min(max_lag, n // 2)
    if hi <= min_lag:
        return 0.0

    peaks = []
    for foot in range(seq.shape[1]):
        x = seq[:, foot] - seq[:, foot].mean()
        var = float(np.dot(x, x))
        if var < 1e-8:
            continue  # 這隻腳整段沒變化（踩死或全程離地）→ 不算步態
        best = 0.0
        for lag in range(min_lag, hi):
            c = float(np.dot(x[:-lag], x[lag:])) / var
            if c > best:
                best = c
        peaks.append(best)

    if not peaks:
        return 0.0
    return float(np.clip(np.mean(peaks), 0.0, 1.0))
