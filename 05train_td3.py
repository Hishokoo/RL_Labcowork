# 執行前請安裝：pip install "gymnasium[mujoco]" stable-baselines3 moviepy tqdm rich
# 05：融合 —— 03 的「重 shaping 俐落步態」+ 04 的「forward-gated 穩健結構」。
#
# 動機：04 雖會走，但步態偏隨便（regularity~0.36、動作不夠平滑）；03 的重懲罰（ctrl 5.0、
# posture 2.0）看起來更俐落穩重，但有站著 attractor 風險。05 沿用 04 已驗證不會被鑽洞的
# 結構（forward_gated + antiphase_gated + alive 底分），只把「不會壓死移動」的 shaping 旋鈕
# 加重，找回精緻度：
#   - smooth_weight 0.02→0.25：大幅加重「相鄰動作差」懲罰 → 不抽搐（俐落感的主要來源）
#   - posture_weight 0.5→1.5、tilt_weight 0.2→0.8：軀幹更穩、更水平
#   - gait_weight 2.0→3.0：步態品質權重加重（forward-gated 下，越會走的 trot 乘越多）
#   - ctrl_weight 0.5→1.0：只微調加重（ctrl 罰動作「幅度」，太重會壓死前進，故不回 03 的 5.0）
# 預期：speed_err / len 維持 04 水準（仍會走不摔），但 jerk↓、uprightness↑、regularity↑。
#
# 共用骨架 tools/gait_train.py；和 04 的差別只在下面 WRAP_KWARGS。先跑 300k 驗證再決定 1M：
#   cd ~/RL_Labcowork && MAX_TIMESTEPS=300000 VIDEO_INTERVAL=9999999 MUJOCO_GL=egl python 05train_td3.py
import os

from tools.gait_train import train

# ── reward 設定（05 融合：04 結構 + 加重的安全 shaping）──────────────────────────
WRAP_KWARGS = dict(
    target_speed=1.0,
    ctrl_weight=1.0,                 # 04=0.5；微加重求省力，但不回 03 的 5.0（避免在 forward-gate 下壓死移動）
    gait_weight=3.0,                 # 04=2.0；加重步態品質權重（forward-gated 下安全）
    posture_weight=1.5,              # 04=0.5；軀幹更穩
    alive_weight=0.5,                # 同 04，保留防摔死底分
    contact_threshold=1.0,
    gait_mode="antiphase_gated",
    forward_mode="progress",
    forward_weight=1.0,
    smooth_weight=0.25,              # 04=0.02；大幅加重抗抽搐 → 俐落感主要來源
    tilt_weight=0.8,                 # 04=0.2；軀幹更水平
    reward_structure="forward_gated",  # 沿用 04 已驗證不被鑽洞的結構
)
# ─────────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    train(
        WRAP_KWARGS,
        output_dir=os.environ.get("OUTPUT_DIR", "output/05train_td3"),
        target_speed=WRAP_KWARGS["target_speed"],
        max_timesteps=int(os.environ.get("MAX_TIMESTEPS", 1_000_000)),
        eval_interval=int(os.environ.get("EVAL_INTERVAL", 50_000)),
        video_interval=int(os.environ.get("VIDEO_INTERVAL", 200_000)),
        checkpoint_freq=int(os.environ.get("CHECKPOINT_FREQ", 100_000)),
        n_envs=int(os.environ.get("N_ENVS", 1)),
    )
