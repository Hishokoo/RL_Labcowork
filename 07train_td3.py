# 執行前請安裝：pip install "gymnasium[mujoco]" stable-baselines3 moviepy tqdm rich
# 07：在 06 的基礎上「把對角踏步銳利度（anti_phase）拉回 03/05 水準」。
#
# 四方比較（output/03_vs_04_comparison.html）顯示 06 修好了超速、抖動、直立度（jerk 0.087、
# uprightness 0.989 反超 03），但代價是重 smooth(0.40)+重 ctrl(1.5) 把踏步幅度過度阻尼，
# anti_phase 0.326(05)→0.212、diagonal_sync 0.677(05)→0.552 雙雙回落，輸給 05/03。
#
# 07 的假設與對策（forward_gated 下安全，不會復活站著 attractor）：
#   - gait_weight 3.0→4.5：anti_phase 在 antiphase_gated 公式裡是「乘法 gate」
#     （gait_contrib = forward·gait_weight·anti·(0.5+0.25·intra1+0.25·intra2)），加重 gait_weight
#     就是直接放大「對角一抬一踏」的收益，把 anti_phase / diagonal_sync 拉回。HTML 建議的安全做法。
#   - smooth_weight 0.40→0.30：06 的重平滑過度磨柔了踏步；放鬆但仍 > 05 的 0.25，守住大部分俐落感。
#   - ctrl_weight 1.5→1.2：ctrl 懲罰也會壓踏步幅度，略放鬆。
# 其餘沿用 06（tent 速度 reward 守住不超速、posture 1.5 / tilt 0.8 / alive 0.5 底分）。
#
# 跑滿 1M、影片每 200k、TB 永遠開（與 03/04/05/06 一致，便於四方→五方比較）：
#   cd ~/RL_Labcowork && MUJOCO_GL=egl python 07train_td3.py
import os

from tools.gait_train import train

# ── reward 設定（07：06 + 加重 gait 拉回 anti_phase + 略放鬆平滑/省力）────────────
WRAP_KWARGS = dict(
    target_speed=1.0,
    ctrl_weight=1.2,                   # 06=1.5；略放鬆，避免把踏步幅度壓死
    gait_weight=4.5,                   # ★ 06=3.0；主力：放大對角交替收益，拉回 anti_phase
    posture_weight=1.5,                # 同 06
    alive_weight=0.5,                  # 同 06，防摔死底分
    contact_threshold=1.0,
    gait_mode="antiphase_gated",
    forward_mode="progress",
    forward_weight=1.0,
    smooth_weight=0.30,                # ★ 06=0.40；放鬆過度阻尼，但仍 > 05 的 0.25 守住平滑
    tilt_weight=0.8,                   # 同 06
    reward_structure="forward_gated",
    forward_gate_shape="tent",         # 同 06，守住不超速
)
# ─────────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    train(
        WRAP_KWARGS,
        output_dir=os.environ.get("OUTPUT_DIR", "output/07train_td3"),
        target_speed=WRAP_KWARGS["target_speed"],
        max_timesteps=int(os.environ.get("MAX_TIMESTEPS", 1_000_000)),
        eval_interval=int(os.environ.get("EVAL_INTERVAL", 50_000)),
        video_interval=int(os.environ.get("VIDEO_INTERVAL", 200_000)),  # 影片每 200k（與各版一致）
        checkpoint_freq=int(os.environ.get("CHECKPOINT_FREQ", 100_000)),
        n_envs=int(os.environ.get("N_ENVS", 1)),
        seed=int(os.environ.get("SEED", 0)),  # 14 multi-seed 驗證用：SEED=1/2 + OUTPUT_DIR 覆寫
    )
