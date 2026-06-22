# 執行前請安裝：pip install "gymnasium[mujoco]" stable-baselines3 moviepy tqdm rich
# 06：05 融合的再進化 —— 修「超速」+ 再加重平滑，目標全面追上/超越 03。
#
# 三方比較（output/03_vs_04_comparison.html）顯示 05 融合勝過 04（jerk/CoT 減半、節奏最佳），
# 但仍輸 03 在「平滑、省力、速度控制」，且 05 超速最嚴重（x_velocity 1.37）。06 針對這兩點：
#   - forward_gate_shape="tent"：速度 reward 從「只設上限（會超速）」改成「恰在目標 1.0 滿分、
#     太快也遞減」的 tent 形（仍 ≥0 不摔死）→ 把速度壓回目標。
#   - smooth_weight 0.25→0.40、ctrl_weight 1.0→1.5：再加重抗抖與省力，往 03 的 jerk 0.028 逼近。
# 其餘沿用 05（forward_gated + antiphase_gated + alive 底分 + gait 3.0 / posture 1.5 / tilt 0.8）。
#
# 跑滿 1M、影片每 200k、TB 永遠開（與 03/04/05 一致，便於 HTML 三方→四方比較）：
#   cd ~/RL_Labcowork && MUJOCO_GL=egl python 06train_td3.py
import os

from tools.gait_train import train

# ── reward 設定（06：05 + tent 速度 + 更重平滑）────────────────────────────────
WRAP_KWARGS = dict(
    target_speed=1.0,
    ctrl_weight=1.5,                   # 05=1.0；再加重求省力（仍 < 03 的 5.0，避免壓死移動）
    gait_weight=3.0,                   # 同 05
    posture_weight=1.5,                # 同 05
    alive_weight=0.5,                  # 同 05，防摔死底分
    contact_threshold=1.0,
    gait_mode="antiphase_gated",
    forward_mode="progress",
    forward_weight=1.0,
    smooth_weight=0.40,                # 05=0.25；再加重抗抖（俐落感主要來源）
    tilt_weight=0.8,                   # 同 05
    reward_structure="forward_gated",
    forward_gate_shape="tent",         # ★ 新：速度 reward 改 tent，壓住 05 的超速
)
# ─────────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    train(
        WRAP_KWARGS,
        output_dir=os.environ.get("OUTPUT_DIR", "output/06train_td3"),
        target_speed=WRAP_KWARGS["target_speed"],
        max_timesteps=int(os.environ.get("MAX_TIMESTEPS", 1_000_000)),
        eval_interval=int(os.environ.get("EVAL_INTERVAL", 50_000)),
        video_interval=int(os.environ.get("VIDEO_INTERVAL", 200_000)),  # 影片每 200k（與各版一致）
        checkpoint_freq=int(os.environ.get("CHECKPOINT_FREQ", 100_000)),
        n_envs=int(os.environ.get("N_ENVS", 1)),
    )
