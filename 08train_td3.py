# 執行前請安裝：pip install "gymnasium[mujoco]" stable-baselines3 moviepy tqdm rich
# 08：在 07 基礎上「拿回 diagonal_sync + 找平滑甜蜜點」。
#
# 五方比較（output/03_vs_04_comparison.html）顯示 07 拿下速度雙冠（mean_speed 0.958、
# speed_error 0.100 全場最佳），anti_phase 也回升超過 03（0.267），但兩個缺口仍在：
#   1. diagonal_sync（同對角兩腳同步）始終卡 ~0.54，遠輸 03(0.712)/05(0.677)——
#      因 antiphase_gated 的 reward 主要獎勵「對角反相」(anti)，intra（同步）權重只 0.25。
#   2. 07 放鬆 smooth(0.30) 換踏步銳利，jerk 從 06 的 0.087 退回 0.120。
#
# 08 對策（仍在 forward_gated + anti 乘法閘下，安全不復活站著 attractor）：
#   - intra_weight 0.25→0.35：antiphase_gated 公式 base+intra_weight·(intra1+intra2)，
#     提高 intra 權重 = 更獎勵「同對角兩腳同步」，把 diagonal_sync 拉回。整體仍被 anti 閘住。
#   - smooth_weight 0.30→0.35：往 06 的 0.40 回靠一點，找「踏步銳利 ↔ jerk」的甜蜜點。
# 其餘沿用 07（gait 4.5 / tent 速度 / ctrl 1.2 / posture 1.5 / tilt 0.8 / alive 0.5）。
#
# 跑滿 1M、影片每 200k、TB 永遠開（與各版一致，便於五方→六方比較）：
#   cd ~/RL_Labcowork && MUJOCO_GL=egl python 08train_td3.py
import os

from tools.gait_train import train

# ── reward 設定（08：07 + 加重 intra 拿回 diagonal_sync + 平滑微升）──────────────
WRAP_KWARGS = dict(
    target_speed=1.0,
    ctrl_weight=1.2,                   # 同 07
    gait_weight=4.5,                   # 同 07
    posture_weight=1.5,                # 同 07
    alive_weight=0.5,                  # 同 07
    contact_threshold=1.0,
    gait_mode="antiphase_gated",
    forward_mode="progress",
    forward_weight=1.0,
    smooth_weight=0.35,                # ★ 07=0.30；往 06 回靠，找平滑甜蜜點
    tilt_weight=0.8,                   # 同 07
    reward_structure="forward_gated",
    forward_gate_shape="tent",         # 同 07，守住不超速
    intra_weight=0.35,                 # ★ 07=0.25；加重「同對角同步」拉回 diagonal_sync
)
# ─────────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    train(
        WRAP_KWARGS,
        output_dir=os.environ.get("OUTPUT_DIR", "output/08train_td3"),
        target_speed=WRAP_KWARGS["target_speed"],
        max_timesteps=int(os.environ.get("MAX_TIMESTEPS", 1_000_000)),
        eval_interval=int(os.environ.get("EVAL_INTERVAL", 50_000)),
        video_interval=int(os.environ.get("VIDEO_INTERVAL", 200_000)),  # 影片每 200k（與各版一致）
        checkpoint_freq=int(os.environ.get("CHECKPOINT_FREQ", 100_000)),
        n_envs=int(os.environ.get("N_ENVS", 1)),
    )
