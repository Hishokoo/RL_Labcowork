# 執行前請安裝：pip install "gymnasium[mujoco]" stable-baselines3 moviepy tqdm rich
# 15：★ 小修版 03 ★ —— 最終 reward 完全等於 03，只修「早期太容易摔死」的訓練起步問題。
#
# 13 multi-seed 證明原版 03 成功率僅 1/3（seed 1/2 從零訓練塌進 fast-fall）。根因推測：
# 03 的 ctrl_weight=5.0 太重——從零訓練時隨機動作配上 ctrl=5 產生巨大負 reward，
# 很容易學成「快速摔倒止損」。15 只改一個地方：ctrl_weight 漸進加重（其餘 100% 維持 03）。
#
# ctrl 排程（ctrl_schedule=(t0, t1, c0, c1)，wrapper 內依累積訓練步插值）：
#   0–100k：ctrl=0.5（放鬆，讓 agent 先學會站穩與移動，不被重 ctrl 逼到摔倒）
#   100k–300k：ctrl 線性 0.5→5.0（已會走後再慢慢加重）
#   300k 之後：ctrl=5.0（完全回到 03，保留省力自然步態）
# 其他全部維持 03：deviation 速度 / legacy 步態 / additive / gait 2.0 / posture 2.0 / alive 1.0，
# 不加 anti-phase、smooth 等新指標。最終 reward 幾乎沒變，只修訓練起步方式。
#
# 流程：先用 seed 1、2 跑到 400k 判斷是否還會 fast-fall（300k 後 ctrl 已回 5.0，能看回到重 ctrl
# 後是否仍穩）；過關再決定跑滿 1M。若穩定學會走，最後再接 12 的 gate 微調 25k 拿掉站著 gait 分。
#   cd ~/RL_Labcowork && SEED=1 OUTPUT_DIR=output/15multiseed/seed_1 MAX_TIMESTEPS=400000 MUJOCO_GL=egl python 15train_td3.py
import os

from tools.gait_train import train

# ── reward 設定（15：= 03 原設定 + 唯一變因 ctrl 漸進排程）─────────────────────
WRAP_KWARGS = dict(
    target_speed=1.0,
    ctrl_weight=5.0,                          # 排程的終值/asymptote（300k 後）
    gait_weight=2.0, posture_weight=2.0, alive_weight=1.0,
    contact_threshold=1.0,
    gait_mode="legacy", forward_mode="deviation",
    forward_weight=float(os.environ.get("FORWARD_WEIGHT", 1.0)),  # 15-A：=1.2 加速度拉力（其餘同 15）
    smooth_weight=0.0, tilt_weight=0.0, reward_structure="additive",
    ctrl_schedule=(100_000, 300_000, 0.5, 5.0),   # ★ 唯一變因：早期放鬆、300k 回到 03 的 5.0
)
# ─────────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    SEED = int(os.environ.get("SEED", 0))
    OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "output/15train_td3")
    MAX_TIMESTEPS = int(os.environ.get("MAX_TIMESTEPS", 1_000_000))
    INIT_MODEL = os.environ.get("INIT_MODEL", "")

    if INIT_MODEL:
        # resume 模式：載入既有 model + replay buffer 續訓（免重跑）。ctrl 已過排程終點（300k），
        # 故移除 schedule、固定 ctrl_weight=5.0（_gstep 從 0 起算無法接續排程相位）。MAX_TIMESTEPS=要續的步數。
        from tools.gait_train import resume
        RESUME_KWARGS = dict(WRAP_KWARGS)
        RESUME_KWARGS.pop("ctrl_schedule", None)
        RESUME_KWARGS["ctrl_weight"] = 5.0
        resume(
            RESUME_KWARGS, output_dir=OUTPUT_DIR,
            init_model_path=INIT_MODEL,
            replay_buffer_path=os.environ.get("REPLAY_BUFFER", ""),
            target_speed=WRAP_KWARGS["target_speed"],
            additional_timesteps=MAX_TIMESTEPS, seed=SEED,
        )
    else:
        train(
            WRAP_KWARGS, output_dir=OUTPUT_DIR,
            target_speed=WRAP_KWARGS["target_speed"],
            max_timesteps=MAX_TIMESTEPS,
            eval_interval=int(os.environ.get("EVAL_INTERVAL", 50_000)),
            video_interval=int(os.environ.get("VIDEO_INTERVAL", 200_000)),
            checkpoint_freq=int(os.environ.get("CHECKPOINT_FREQ", 100_000)),
            n_envs=int(os.environ.get("N_ENVS", 1)),
            seed=SEED,
        )
