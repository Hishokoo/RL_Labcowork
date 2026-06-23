import gymnasium as gym
import numpy as np

from tools import gait_metrics

# Ant-v5 的四隻腳在這個 MuJoCo 模型裡沒有命名（body name 是空字串），
# 用 print(model.body(i).name for i in range(model.nbody)) 配合 geom bodyid 反查得到的對應關係：
#   body 4  (left_ankle_geom)   = front_left  (FL)
#   body 7  (right_ankle_geom)  = front_right (FR)
#   body 10 (third_ankle_geom)  = back_left   (BL)
#   body 13 (fourth_ankle_geom) = back_right  (BR)
FOOT_BODY_IDS = [4, 7, 10, 13]  # 順序：[FL, FR, BL, BR]


class RealisticGaitWrapper(gym.Wrapper):
    """
    v12a：在 03 wrapper 基礎上新增「antiphase_bonus_weight」做對齊微調。

    動機 ─ 12 (legacy + speed_gate=0.3, 從 03 finetune) 在 25k 達峰、之後退化。根因：
    wrapper 的 r_gait 公式 (0.4·diag1 + 0.4·diag2 + 0.2·cross_pattern) 不直接獎勵
    scorecard 量測的 anti_phase。actor 25k 後越優化 wrapper、anti_phase 反而被推更低。

    v12a 補三道對齊訊號 (與 03 完全向後相容)：
      1. antiphase_bonus_weight (新增): r_gait 之外疊加 w·anti_phase(contacts)，受同一
         speed gate 保護。直接對齊 scorecard 的 anti_phase 指標，而非透過 cross_pattern
         的 weak proxy。預設 0 = 不啟用。
      2. smooth_weight (沿用 03 已實作欄位, 03/12 預設 0): 直接對齊 scorecard 的
         action_jerk，保護 03 的 0.028 強項在 finetune 期間漂掉。
      3. tilt_weight   (沿用 03 已實作欄位, 03/12 預設 0): 對齊 scorecard 的
         uprightness。各版本本來就高，貢獻量小，純粹當對齊保險。

    為何不切到 antiphase_gated 模式：那會把整個 r_gait 乘上 anti_phase。03 訓練時
    anti_phase ≈ 0.2，切換後 r_gait 從 ~1.6 跌到 ~0.32，actor/critic value function
    受大 shock，浪費 03 checkpoint 的 prior。疊加 bonus 是更穩的增量改動。

    其餘行為（gait_mode 三種模式、forward_mode 兩種、reward_structure 兩種、
    ctrl_schedule、gait_speed_gate、intra_weight 等）與 gait_wrapper_03.py 完全一致。
    """

    def __init__(
        self,
        env,
        target_speed: float = 1.0,
        ctrl_weight: float = 5.0,
        gait_weight: float = 2.0,
        posture_weight: float = 2.0,
        alive_weight: float = 1.0,
        contact_threshold: float = 1.0,
        gait_mode: str = "legacy",
        forward_mode: str = "deviation",
        forward_weight: float = 1.0,
        smooth_weight: float = 0.0,
        tilt_weight: float = 0.0,
        reward_structure: str = "additive",
        forward_gate_shape: str = "cap",
        intra_weight: float = 0.25,
        gait_speed_gate: float = 0.0,
        ctrl_schedule: tuple | None = None,
        antiphase_bonus_weight: float = 0.0,  # v12a 新增
    ):
        super().__init__(env)
        self.target_speed = target_speed
        self.ctrl_weight = ctrl_weight
        self.gait_weight = gait_weight
        self.posture_weight = posture_weight
        self.alive_weight = alive_weight
        self.contact_threshold = contact_threshold
        if gait_mode not in ("legacy", "antiphase", "antiphase_gated"):
            raise ValueError(f"未知的 gait_mode：{gait_mode}（可用 'legacy' / 'antiphase' / 'antiphase_gated'）")
        if forward_mode not in ("deviation", "progress"):
            raise ValueError(f"未知的 forward_mode：{forward_mode}（可用 'deviation' / 'progress'）")
        if reward_structure not in ("additive", "forward_gated"):
            raise ValueError(f"未知的 reward_structure：{reward_structure}（可用 'additive' / 'forward_gated'）")
        if forward_gate_shape not in ("cap", "tent"):
            raise ValueError(f"未知的 forward_gate_shape：{forward_gate_shape}（可用 'cap' / 'tent'）")
        self.gait_mode = gait_mode
        self.forward_mode = forward_mode
        self.forward_weight = forward_weight
        self.smooth_weight = smooth_weight
        self.tilt_weight = tilt_weight
        self.forward_gate_shape = forward_gate_shape
        self.reward_structure = reward_structure
        if not 0.0 <= intra_weight <= 0.5:
            raise ValueError(f"intra_weight 須在 [0, 0.5]（兩個 intra 項合計 ≤ 1）：{intra_weight}")
        self.intra_weight = intra_weight
        if gait_speed_gate < 0.0:
            raise ValueError(f"gait_speed_gate 須 ≥ 0（0=關閉、>0=啟用的速度門檻 m/s）：{gait_speed_gate}")
        self.gait_speed_gate = gait_speed_gate
        if antiphase_bonus_weight < 0.0:
            raise ValueError(f"antiphase_bonus_weight 須 ≥ 0：{antiphase_bonus_weight}")
        self.antiphase_bonus_weight = antiphase_bonus_weight
        # ctrl 排程（None=關閉，用固定 ctrl_weight）。(t0, t1, c0, c1)：step≤t0→c0、
        # t0..t1 線性 c0→c1、>t1→c1。
        if ctrl_schedule is not None:
            t0, t1, c0, c1 = ctrl_schedule
            if not (0 <= t0 <= t1):
                raise ValueError(f"ctrl_schedule 須 0≤t0≤t1：{ctrl_schedule}")
        self.ctrl_schedule = ctrl_schedule
        self._gstep = 0
        self.prev_contacts = np.zeros(4)
        self.prev_action = np.zeros(env.action_space.shape[0], dtype=np.float64)
        self.foot_body_ids = FOOT_BODY_IDS

    def _effective_ctrl_weight(self) -> float:
        if self.ctrl_schedule is None:
            return self.ctrl_weight
        t0, t1, c0, c1 = self.ctrl_schedule
        if self._gstep <= t0:
            return c0
        if self._gstep >= t1:
            return c1
        return c0 + (c1 - c0) * (self._gstep - t0) / (t1 - t0)

    def reset(self, **kwargs):
        self.prev_contacts = np.zeros(4)
        self.prev_action = np.zeros(self.env.action_space.shape[0], dtype=np.float64)
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, _orig_reward, terminated, truncated, info = self.env.step(action)

        contacts = self._get_foot_contacts()
        is_healthy = not terminated
        reward, components = self._compute_reward(action, info, is_healthy, contacts)

        info["foot_contacts"] = contacts
        info["reward_components"] = components
        info["original_reward"] = _orig_reward
        self.prev_contacts = contacts
        self.prev_action = np.asarray(action, dtype=np.float64)
        self._gstep += 1

        return obs, reward, terminated, truncated, info

    def _get_foot_contacts(self) -> np.ndarray:
        cfrc = self.env.unwrapped.data.cfrc_ext[self.foot_body_ids]
        contact_magnitudes = np.linalg.norm(cfrc, axis=1)
        return (contact_magnitudes > self.contact_threshold).astype(np.float32)

    def _compute_reward(self, action, info, is_healthy, contacts):
        # 1. 速度 reward
        x_vel = info.get("x_velocity", 0.0)
        if self.forward_mode == "progress":
            r_forward = self.forward_weight * max(0.0, min(x_vel, self.target_speed))
        else:  # "deviation"
            r_forward = -self.forward_weight * abs(x_vel - self.target_speed)

        # 2. 存活基本分
        r_alive = self.alive_weight if is_healthy else 0.0

        # 3. 控制懲罰
        r_ctrl = -self._effective_ctrl_weight() * float(np.sum(np.square(action)))

        # 4. 步態 reward 主項（與 03 完全一致）
        if self.gait_mode == "antiphase_gated":
            intra1 = 1.0 - abs(contacts[0] - contacts[3])
            intra2 = 1.0 - abs(contacts[1] - contacts[2])
            anti = gait_metrics.anti_phase(contacts)
            base = 1.0 - 2.0 * self.intra_weight
            r_gait = self.gait_weight * anti * (base + self.intra_weight * intra1 + self.intra_weight * intra2)
        elif self.gait_mode == "antiphase":
            intra1 = 1.0 - abs(contacts[0] - contacts[3])
            intra2 = 1.0 - abs(contacts[1] - contacts[2])
            anti = gait_metrics.anti_phase(contacts)
            r_gait = self.gait_weight * (0.2 * intra1 + 0.2 * intra2 + 0.6 * anti)
        else:  # "legacy"
            diag1_sync = 1.0 - abs(contacts[0] - contacts[3])
            diag2_sync = 1.0 - abs(contacts[1] - contacts[2])
            cross_pattern = abs(contacts[0] - contacts[1])
            r_gait = self.gait_weight * (0.4 * diag1_sync + 0.4 * diag2_sync + 0.2 * cross_pattern)

        # 4b. 柔和速度 gate（v12a：抽出 gate 變數，讓 anti_phase bonus 共用同一個 gate）
        if self.gait_speed_gate > 0.0:
            p = float(np.clip(max(x_vel, 0.0) / self.gait_speed_gate, 0.0, 1.0))
            gate = p * p * (3.0 - 2.0 * p)
        else:
            gate = 1.0
        r_gait *= gate

        # 4c. v12a 新增：anti_phase 對齊獎金（受同一 speed gate 保護）
        # 設計意圖：legacy 主公式只用 cross_pattern 弱代理 anti_phase，這裡直接疊加
        # 完整 anti_phase(contacts) 訊號當 bonus。站著時 anti_phase ≈ 0，gate 也 ≈ 0，
        # 雙重歸零；真 trot 單腳支撐瞬間 anti_phase = 1，bonus 最大 = antiphase_bonus_weight。
        antiphase_value = float(gait_metrics.anti_phase(contacts))
        r_antiphase_bonus = self.antiphase_bonus_weight * antiphase_value * gate
        r_gait += r_antiphase_bonus

        # 5. 姿態 reward（軀幹高度）
        torso_z = self.env.unwrapped.data.qpos[2]
        r_posture = -self.posture_weight * abs(torso_z - 0.6)

        # 6. 動作平滑懲罰（jerk）
        action_arr = np.asarray(action, dtype=np.float64)
        jerk = float(np.sum(np.square(action_arr - self.prev_action)))
        r_smooth = -self.smooth_weight * jerk

        # 7. 軀幹直立懲罰
        qpos = self.env.unwrapped.data.qpos
        upright = gait_metrics.uprightness(qpos)
        r_tilt = -self.tilt_weight * (1.0 - upright)

        if self.reward_structure == "forward_gated":
            if self.forward_gate_shape == "tent":
                forward_progress = self.target_speed * max(
                    0.0, 1.0 - abs(x_vel - self.target_speed) / self.target_speed)
            else:
                forward_progress = max(0.0, min(x_vel, self.target_speed))
            gait_contrib = forward_progress * r_gait
            total = forward_progress + gait_contrib + r_alive + r_ctrl + r_smooth + r_tilt + r_posture
            r_forward = forward_progress
            r_gait = gait_contrib
        else:
            total = r_forward + r_alive + r_ctrl + r_gait + r_posture + r_smooth + r_tilt

        components = {
            "forward": r_forward,
            "alive": r_alive,
            "ctrl": r_ctrl,
            "gait": r_gait,
            "antiphase_bonus": r_antiphase_bonus,  # v12a 新增 logging 欄位
            "posture": r_posture,
            "smooth": r_smooth,
            "tilt": r_tilt,
            "x_velocity": x_vel,
            "torso_z": torso_z,
            "uprightness": upright,
            "anti_phase": antiphase_value,
        }
        return float(total), components
