# 執行前請安裝：pip install "gymnasium[mujoco]" stable-baselines3 moviepy tqdm rich
# 13：★ multi-seed 穩健性驗證 ★ —— 配方已定案（03 → gate curriculum 取 ~25k），跨 seed 驗證。
#
# 背景：12@25k（seed 0）是目前最佳模型（五約束全達標、多項勝 03，站著 attractor 已用 gate 拿掉）。
# 但「25k 甜蜜點 + 速度漂移」只在 seed 0 觀察過。13 用新 seed 重跑整條兩階段 pipeline，
# 確認結論跨 seed 穩定（而非 seed 0 的偶然）。
#
# 每個 SEED 完整跑兩階段（seed 1、2 依序跑，避免同機資源競爭）：
#   Stage 1：原版 03（無 gate），1M steps，seed=SEED。
#   Stage 2：載入該 seed 的 03，套 gate curriculum（= 12 的 finetune），50k steps，每 25k 評估+錄影。
#
# 重要紀律（依設計）：
#   - 預先固定「stage2 的 25k checkpoint」為主要結果，不可事後替每個 seed 挑最好 checkpoint。
#   - 50k 只用來觀察「速度漂移」是否跨 seed 重現（25k→50k 是否同樣往下漂）。
#   - 評估用相同 10 個 evaluation seeds（eval_scorecard 內建 reset seed 100..109），比較 mean±std、
#     成功率，五約束（ep_len=1000 / speed≥0.9 / jerk≤0.05 / CoT≤1.5 / diagonal_sync≥0.65）
#     + stationary_fraction（確認沒在站著）。
#
# seed 0 已有（output/03train_td3 + output/12train_td3）；本檔只跑新 seed：
#   cd ~/RL_Labcowork && SEED=1 MUJOCO_GL=egl python 13multiseed_td3.py
#   cd ~/RL_Labcowork && SEED=2 MUJOCO_GL=egl python 13multiseed_td3.py
import os

from tools.gait_train import train, finetune

SEED = int(os.environ.get("SEED", 1))
BASE = os.environ.get("OUTPUT_BASE", f"output/13multiseed/seed_{SEED}")

# Stage 1 reward = 原版 03（無 gate）
WRAP_03 = dict(
    target_speed=1.0,
    ctrl_weight=5.0, gait_weight=2.0, posture_weight=2.0, alive_weight=1.0,
    contact_threshold=1.0,
    gait_mode="legacy", forward_mode="deviation", forward_weight=1.0,
    smooth_weight=0.0, tilt_weight=0.0, reward_structure="additive",
)
# Stage 2 reward = 03 + gate（= 11/12 的唯一變因）
WRAP_GATE = dict(WRAP_03, gait_speed_gate=0.3)


if __name__ == "__main__":
    # ── Stage 1：原版 03，1M ──────────────────────────────────────────────
    train(
        WRAP_03, output_dir=f"{BASE}/stage1_03", target_speed=1.0,
        max_timesteps=int(os.environ.get("STAGE1_STEPS", 1_000_000)),
        eval_interval=50_000, video_interval=200_000, checkpoint_freq=100_000,
        n_envs=1, seed=SEED,
    )
    # ── Stage 2：gate curriculum，50k（25k checkpoint 為主要結果）──────────
    finetune(
        WRAP_GATE, output_dir=f"{BASE}/stage2_gate",
        init_model_path=f"{BASE}/stage1_03/final_model.zip",
        target_speed=1.0,
        max_timesteps=int(os.environ.get("STAGE2_STEPS", 50_000)),
        learning_rate=1e-4, action_noise_sigma=0.03,
        eval_interval=25_000, video_interval=25_000, checkpoint_freq=25_000,
        seed=SEED,
    )
    print(f"=== SEED {SEED} 兩階段完成 ===")
