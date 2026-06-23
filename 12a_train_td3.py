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
#   - eval/ckpt 25k → 10k（加密 eval 是防回頭找不到峰的最便宜保險）
#   - video 25k → 100k（錄影/編碼成本高，跟 eval 數值頻率脫鉤，每 100k 步留一支代表性影片即可）
#
# 成功標準（看影片 + scorecard）：episode_length=1000、speed≥0.9、anti_phase≥0.27、
# diagonal_sync 不退化（≥0.65）、jerk≤0.05、CoT≤1.5、觀感不退化。
#
# ★ 50k 驗證結論（本機跑了三組設定，皆未達標，已比對 12@25k）★
#   1) antiphase_bonus_weight=0.5, gate=0.3（本檔現有設定）：anti_phase 峰值 0.247 @40k，三組最佳
#   2) antiphase_bonus_weight=1.0, gate=0.3：anti_phase 峰值降到 0.231（加重 bonus 反而更差——
#      legacy 主項的 diag1/diag2_sync 對「四腳同步」有結構性偏好，bonus 加倍時 diagonal_sync
#      被推更高、anti_phase 反被擠壓）
#   3) antiphase_bonus_weight=0.5, gate=0.15：anti_phase 峰值僅 0.230，比設定 1) 更差
#   與 12@25k（anti_phase=0.243±0.007）對照：設定 1) 的 0.247±0.010 落在 1 個標準差內，
#   不是統計上有意義的提升；而 12@25k 在 mean_speed / speed_error / diagonal_sync /
#   transport_cost 都顯著贏過設定 1)。結論：**v12a（疊加 bonus 的設計）未能勝過 12@25k**，
#   問題不是參數沒調好，是「在 legacy 主公式上疊加 bonus」這個設計本身在 0.5～1.0 權重、
#   0.15～0.3 gate 範圍內打不穿 anti_phase 的天花板。下一步應參考 spec 第 8 節的 v12b
#   （EMA gate）或重新設計 reward 結構，而非繼續在這個方向微調。
#   詳細數據見 CHANGELOG.md 與 output/03_vs_12at25k_raw_episodes.csv。
#
#   cd ~/RL_Labcowork && MUJOCO_GL=egl python 12a_train_td3.py
import os

from tools.gait_train import finetune
from tools.gait_wrapper_12a import RealisticGaitWrapper as Wrapper12a

# ── reward 設定（= 12 + 三道對齊訊號；下方為三組驗證中最佳的一組，見上方結論）──────
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
        video_interval=int(os.environ.get("VIDEO_INTERVAL", 100_000)),
        checkpoint_freq=int(os.environ.get("CHECKPOINT_FREQ", 10_000)),
        wrapper_cls=Wrapper12a,
    )
