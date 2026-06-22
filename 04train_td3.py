# 執行前請安裝：pip install "gymnasium[mujoco]" stable-baselines3 moviepy tqdm rich
# 04：會走的基準配方 —— forward 主導 + 溫和懲罰 + forward-gated 步態。
#
# 經三次失敗（站著不動 / 原地踏步 / 快速摔倒，見 CHANGELOG 與記憶）後定下的配方：
#   - reward_structure="forward_gated"：步態 bonus 以「前進」為閘門，不前進整個正向 reward=0
#     → 站著、原地踏步都拿不到分（補掉前兩種洞）
#   - gait_mode="antiphase_gated"：步態分用 anti_phase 當乘法 gate，靜態=0
#   - 溫和懲罰（ctrl 0.5）+ 小 alive 底分（0.5）：站著≈+0.4 非負 → 不摔死（補掉第三種洞），
#     但 << 走路 → 不站著
# 1M 實測：x_velocity~1.2、speed_err 收斂 ~0.3、episode 全程 1000、return~1750、regularity 0.1→0.36。
#
# 共用訓練骨架在 tools/gait_train.py；reward 旋鈕定義在 tools/gait_wrapper_03.py。
# 執行控制可用 env var 覆寫（OUTPUT_DIR / MAX_TIMESTEPS / EVAL_INTERVAL / VIDEO_INTERVAL / N_ENVS）：
#   cd ~/RL_Labcowork && MUJOCO_GL=egl python 04train_td3.py            # 正式 1M + 錄影
#   MAX_TIMESTEPS=300000 VIDEO_INTERVAL=9999999 python 04train_td3.py   # 快速 300k 驗證、關錄影
import os

from tools.gait_train import train

# ── reward 設定（04 的會走配方）──────────────────────────────────────────────
WRAP_KWARGS = dict(
    target_speed=1.0,
    ctrl_weight=0.5,                 # 溫和（原 5.0 太重 → 逼出摔死）
    gait_weight=2.0,
    posture_weight=0.5,
    alive_weight=0.5,                # 小底分：站著非負 → 不摔死；但 << 走路 → 不站著
    contact_threshold=1.0,
    gait_mode="antiphase_gated",
    forward_mode="progress",
    forward_weight=1.0,
    smooth_weight=0.02,
    tilt_weight=0.2,
    reward_structure="forward_gated",  # 步態 bonus 以前進為閘門
)
# ─────────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    train(
        WRAP_KWARGS,
        output_dir=os.environ.get("OUTPUT_DIR", "output/04train_td3"),
        target_speed=WRAP_KWARGS["target_speed"],
        max_timesteps=int(os.environ.get("MAX_TIMESTEPS", 1_000_000)),
        eval_interval=int(os.environ.get("EVAL_INTERVAL", 50_000)),
        video_interval=int(os.environ.get("VIDEO_INTERVAL", 200_000)),
        checkpoint_freq=int(os.environ.get("CHECKPOINT_FREQ", 100_000)),
        n_envs=int(os.environ.get("N_ENVS", 1)),
    )
