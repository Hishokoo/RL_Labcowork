# 執行前請安裝：pip install "gymnasium[mujoco]" stable-baselines3 moviepy tqdm rich
# 11：★ 回到 03，只修一個缺點 ★ —— 嚴格單變因實驗（推翻 04–10 的「堆權重 / 換結構」路線）。
#
# 反省 04–10 的核心問題：
#   1. 太積極追代理指標（anti_phase / regularity）。接觸是 0/1 門檻切的（gait_wrapper_03.py 的
#      _get_foot_contacts），antiphase_gated 又放大這個跳變訊號 → 模型學成「為踩中 reward 條件
#      而誇張抬腳/頓腳」，不是自然走（解釋了 anti_phase 漂亮但 jerk/CoT 差）。
#   2. forward_gated 把整個正 reward × 速度 → 早期走不動幾乎零正 reward → 10 在 200k–800k 影片
#      秒摔、1M 才突然找到策略 = 學習極不穩。
#   3. 10 同時改了 ctrl/alive/posture/速度公式/reward 結構/gait weight 六個變因，無法隔離歸因。
#
# 11 的設計 = 03 的設定原封不動，只加「一個變因」：對 r_gait 加柔和速度 gate，修掉 legacy 唯一
# 的缺點（站著四腳著地時 r_gait=0.8·gait_weight 的站著 attractor）。
#   gait_speed_gate=0.3 → wrapper 內：p=clip(max(x_vel,0)/0.3,0,1); r_gait *= p²(3−2p)（smoothstep）
#   效果：站著 r_gait=0；x_vel>0.3 後幾乎完全恢復 03 reward；不把整個正 reward 與速度相乘；
#         smoothstep 連續無跳變 → 不放大 0/1 接觸訊號 → 不誘發頓腳。保留 03 的平滑/省力/自然步態。
#
# 流程（依建議）：先跑 300k 看影片，若仍保留 03 觀感，再跑滿 1M。先不加 smooth / anti_phase /
# 週期 reward。03 即為目前正式 baseline。
#
#   cd ~/RL_Labcowork && MUJOCO_GL=egl MAX_TIMESTEPS=300000 python 11train_td3.py   # 探針
#   cd ~/RL_Labcowork && MUJOCO_GL=egl python 11train_td3.py                         # 跑滿 1M
import os

from tools.gait_train import train

# ── reward 設定（11：= 03 原設定 + 唯一變因 gait_speed_gate）────────────────────
WRAP_KWARGS = dict(
    target_speed=1.0,
    ctrl_weight=5.0,                   # 03 原值（重 ctrl = 03 平滑/省力的可能來源，保持不動以隔離變因）
    gait_weight=2.0,                   # 03 原值
    posture_weight=2.0,                # 03 原值
    alive_weight=1.0,                  # 03 原值
    contact_threshold=1.0,
    gait_mode="legacy",                # 03 原值
    forward_mode="deviation",          # 03 原值（偏離 target 即罰，站著=-1）
    forward_weight=1.0,                # 03 原值
    smooth_weight=0.0,                 # 03 原值（不加 jerk 懲罰）
    tilt_weight=0.0,                   # 03 原值
    reward_structure="additive",       # 03 原值（不乘速度閘門）
    gait_speed_gate=0.3,               # ★ 唯一變因：只 gate r_gait，修站著 attractor
)
# ─────────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    train(
        WRAP_KWARGS,
        output_dir=os.environ.get("OUTPUT_DIR", "output/11train_td3"),
        target_speed=WRAP_KWARGS["target_speed"],
        max_timesteps=int(os.environ.get("MAX_TIMESTEPS", 1_000_000)),
        eval_interval=int(os.environ.get("EVAL_INTERVAL", 50_000)),
        video_interval=int(os.environ.get("VIDEO_INTERVAL", 100_000)),  # 探針期影片每 100k 方便看
        checkpoint_freq=int(os.environ.get("CHECKPOINT_FREQ", 100_000)),
        n_envs=int(os.environ.get("N_ENVS", 1)),
    )
