# 執行前請安裝：pip install "gymnasium[mujoco]" stable-baselines3 moviepy tqdm rich
# 09：把「觀感主導」三項（diagonal_sync 對角同步 / action_jerk 平滑 / transport_cost 效率）
#     往 03 逼——這三項到 08 為止仍由 03(legacy) 領先（0.712 / 0.028 / 1.02）。
#
# 六方比較（output/03_vs_04_comparison.html）顯示一條「速度準↔節奏銳利↔平滑省力」取捨前緣：
# 07 速度最準、08 節奏最銳利，但兩者的 jerk/CoT 都輸 03，diagonal_sync 也只 ~0.55。
#
# 09 的賭注（與 06 的關鍵差異）：
#   - 06 是「弱 gait(3.0) + 重 smooth(0.40)」→ 步態被磨柔、anti_phase 掉到 0.212。
#   - 09 改成「強 gait(4.5，沿用 08) + 更重 smooth(0.50) + 更重 ctrl(2.0)」：
#     用強 gait/intra 撐住節奏與對角同步，同時把 smooth/ctrl 拉高去磨平滑、省力，
#     賭它能在「守住 08 的節奏」前提下把 jerk/CoT 往 03 壓。
#   - intra_weight 0.35→0.40：再推一點 diagonal_sync（08 已顯示邊際遞減，這是最後一搏）。
# 其餘沿用 08（tent 速度守不超速、posture 1.5 / tilt 0.8 / alive 0.5 底分）。
#
# 風險：smooth/ctrl 過重可能把步態壓回 06 的「柔」（anti_phase 回落）。這是觀感↔節奏的取捨，
# 依使用者要求「把觀感權重再拉上來」而為；結果用 eval_scorecard 10-ep 驗證。
#
# 跑滿 1M、影片每 200k、TB 永遠開（與各版一致）：
#   cd ~/RL_Labcowork && MUJOCO_GL=egl python 09train_td3.py
import os

from tools.gait_train import train

# ── reward 設定（09：08 + 大幅拉高 smooth/ctrl 磨觀感）──────────────────────────
WRAP_KWARGS = dict(
    target_speed=1.0,
    ctrl_weight=2.0,                   # ★ 08=1.2；壓 transport_cost 往 03 的 1.02 逼
    gait_weight=4.5,                   # 同 08，強 gait 撐住節奏（避免重回 06 的柔）
    posture_weight=1.5,                # 同 08
    alive_weight=0.5,                  # 同 08
    contact_threshold=1.0,
    gait_mode="antiphase_gated",
    forward_mode="progress",
    forward_weight=1.0,
    smooth_weight=0.50,                # ★ 08=0.35；壓 jerk 往 03 的 0.028 逼
    tilt_weight=0.8,                   # 同 08
    reward_structure="forward_gated",
    forward_gate_shape="tent",         # 同 08，守住不超速
    intra_weight=0.40,                 # ★ 08=0.35；再推一點 diagonal_sync
)
# ─────────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    train(
        WRAP_KWARGS,
        output_dir=os.environ.get("OUTPUT_DIR", "output/09train_td3"),
        target_speed=WRAP_KWARGS["target_speed"],
        max_timesteps=int(os.environ.get("MAX_TIMESTEPS", 1_000_000)),
        eval_interval=int(os.environ.get("EVAL_INTERVAL", 50_000)),
        video_interval=int(os.environ.get("VIDEO_INTERVAL", 200_000)),  # 影片每 200k（與各版一致）
        checkpoint_freq=int(os.environ.get("CHECKPOINT_FREQ", 100_000)),
        n_envs=int(os.environ.get("N_ENVS", 1)),
    )
