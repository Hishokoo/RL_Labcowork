# 執行前請安裝：pip install "gymnasium[mujoco]" stable-baselines3 moviepy tqdm rich
# 10：★ 結構性換設計（非調權重）★ —— 把 03 的「平滑步態來源」接上前進閘門。
#
# 為什麼換結構（而非繼續調 04–09 的權重）：
#   08 把 intra_weight 0.25→0.35，anti_phase 大漲（0.267→0.358）但 diagonal_sync 幾乎不動
#   （0.544→0.575）。這證明 diagonal_sync / jerk / CoT 這三個「觀感項」的 ~0.58/0.12/3.0 是
#   antiphase_gated 結構的天花板，不是還沒找到的權重——antiphase_gated 獎勵「單腳支撐的瞬間
#   交替」，步態本質上比較跳（jerk 高）。03 在這三項贏，是因為它用了不同結構：
#   gait_mode="legacy"（逐幀連續獎勵「對角同步」diag1/diag2，天生平滑、同步）。
#
# 10 的設計 = 取 03 的 legacy 步態獎勵（平滑來源）× forward_gated 閘門：
#   - gait_mode="legacy"：r_gait = gait_weight·(0.4·diag1_sync + 0.4·diag2_sync + 0.2·cross)，
#     直接、連續地獎勵「同對角兩腳同步」→ 把 diagonal_sync 往 03 的 0.712 拉，步態也更平滑。
#   - reward_structure="forward_gated"：legacy 站著時 diag1=diag2=1 → r_gait=0.8·gait_weight 是
#     站著 attractor（03 靠 deviation 速度罰站著、我們改用閘門）；gait_contrib=forward·r_gait，
#     站著 forward≈0 → 整個步態收益歸零，堵掉站著且不用負向 penalty。
#   - forward_gate_shape="tent"：沿用 07/08 已驗證的速度控制（不超速、speed_error 低）。
#   - 懲罰回到「03 風格的省力 + 溫和平滑」：ctrl 2.0（省力，往 03 的 CoT 1.02 逼）、smooth 0.25、
#     tilt 0.6、posture 1.5、alive 0.5 底分（forward_gated 安全區，不會像早期 ctrl=5 摔死）。
#
# 假設：legacy×gated 的 diagonal_sync / jerk / CoT 應顯著優於 antiphase_gated 各版（06–09），
# 逼近 03，同時不站著、速度受控。代價可能是 anti_phase（legacy 不直接獎勵反相）偏低——
# 但那本來就是 05/08 的強項，10 專攻「觀感三項」。
#
# 跑滿 1M、影片每 200k、TB 永遠開（與各版一致）：
#   cd ~/RL_Labcowork && MUJOCO_GL=egl python 10train_td3.py
import os

from tools.gait_train import train

# ── reward 設定（10：legacy 步態 × forward_gated，結構性新設計）─────────────────
WRAP_KWARGS = dict(
    target_speed=1.0,
    ctrl_weight=2.0,                   # 省力，往 03 的 transport_cost 1.02 逼（< 03 的 5.0，避免摔死）
    gait_weight=2.5,                   # legacy 連續對角同步獎勵的權重（03 用 2.0，略加強）
    posture_weight=1.5,
    alive_weight=0.5,                  # forward_gated 安全底分，防摔死
    contact_threshold=1.0,
    gait_mode="legacy",                # ★ 結構核心：03 的逐幀對角同步（平滑/同步來源）
    forward_mode="progress",
    forward_weight=1.0,
    smooth_weight=0.25,                # 溫和抗抖
    tilt_weight=0.6,
    reward_structure="forward_gated",  # ★ 結構核心：前進閘門堵掉 legacy 的站著 attractor
    forward_gate_shape="tent",         # 沿用速度控制
)
# ─────────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    train(
        WRAP_KWARGS,
        output_dir=os.environ.get("OUTPUT_DIR", "output/10train_td3"),
        target_speed=WRAP_KWARGS["target_speed"],
        max_timesteps=int(os.environ.get("MAX_TIMESTEPS", 1_000_000)),
        eval_interval=int(os.environ.get("EVAL_INTERVAL", 50_000)),
        video_interval=int(os.environ.get("VIDEO_INTERVAL", 200_000)),  # 影片每 200k（與各版一致）
        checkpoint_freq=int(os.environ.get("CHECKPOINT_FREQ", 100_000)),
        n_envs=int(os.environ.get("N_ENVS", 1)),
    )
