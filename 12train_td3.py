# 執行前請安裝：pip install "gymnasium[mujoco]" stable-baselines3 moviepy tqdm rich
# 12：★ checkpoint curriculum 探針 ★ —— 從「已會走的 03」接續微調，只換上 11 的速度 gate。
#
# 11（從零訓練 03+gate）秒摔，證明「站著 gait 分數」是 03 的學習鷹架（早期靠它站穩才探索出走）。
# 12 改問另一個問題：鷹架是否「只在學會走之前」需要？做法 = 不從零訓練，而是：
#   - 載入 03 final_model 的 actor/critic（已會走，x_vel 0.94）
#   - 換成 11 的 reward（gait_speed_gate=0.3：只 gate r_gait、smoothstep、不乘 forward/alive）
#   - 清空 replay buffer（不載入 03 舊 buffer，避免混用新舊 reward）
#   - 不做 random warmup（learning_starts=0，第一 episode 即用載入策略）
#   - lr 1e-4、action noise 0.03、只微調 ~120k、每 25k 評估+錄影
#
# 成功標準（看影片 + scorecard）：episode_length=1000、speed≥0.9、jerk≤0.05、CoT≤1.5、觀感不退化。
#
# 這實驗能回答什麼：
#   - 若成功 → gait 分數只在「學會走前」需要。正式方案 = 兩階段 curriculum（03 學會走 → 開 gate 微調）。
#   - 若仍退化 → 問題不只鷹架；11 用「瞬時速度」gate 會在正常步態的減速相位切斷 reward、干擾整個
#     gait cycle。此時應放棄 gait gate，改成「保留完整 03 reward + 只對持續站著加懲罰」
#     （用 ~1 秒 EMA 平均速度，連續 < 0.1 m/s 才扣，正常步態瞬間減速不受影響）。
#
#   cd ~/RL_Labcowork && MUJOCO_GL=egl python 12train_td3.py
import os

from tools.gait_train import finetune

# ── reward 設定（= 11：03 原設定 + 唯一變因 gait_speed_gate=0.3）─────────────────
WRAP_KWARGS = dict(
    target_speed=1.0,
    ctrl_weight=5.0, gait_weight=2.0, posture_weight=2.0, alive_weight=1.0,
    contact_threshold=1.0,
    gait_mode="legacy", forward_mode="deviation", forward_weight=1.0,
    smooth_weight=0.0, tilt_weight=0.0, reward_structure="additive",
    gait_speed_gate=0.3,
)
# ─────────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    finetune(
        WRAP_KWARGS,
        output_dir=os.environ.get("OUTPUT_DIR", "output/12train_td3"),
        init_model_path=os.environ.get("INIT_MODEL", "output/03train_td3/final_model.zip"),
        target_speed=WRAP_KWARGS["target_speed"],
        max_timesteps=int(os.environ.get("MAX_TIMESTEPS", 120_000)),
        learning_rate=float(os.environ.get("LEARNING_RATE", 1e-4)),
        action_noise_sigma=float(os.environ.get("ACTION_NOISE", 0.03)),
        eval_interval=int(os.environ.get("EVAL_INTERVAL", 25_000)),
        video_interval=int(os.environ.get("VIDEO_INTERVAL", 25_000)),
        checkpoint_freq=int(os.environ.get("CHECKPOINT_FREQ", 25_000)),
    )
