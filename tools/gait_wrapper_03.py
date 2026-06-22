import gymnasium as gym
import numpy as np

# Ant-v5 的四隻腳在這個 MuJoCo 模型裡沒有命名（body name 是空字串），
# 用 print(model.body(i).name for i in range(model.nbody)) 配合 geom bodyid 反查得到的對應關係：
#   body 4  (left_ankle_geom)   = front_left  (FL)
#   body 7  (right_ankle_geom)  = front_right (FR)
#   body 10 (third_ankle_geom)  = back_left   (BL)
#   body 13 (fourth_ankle_geom) = back_right  (BR)
FOOT_BODY_IDS = [4, 7, 10, 13]  # 順序：[FL, FR, BL, BR]


class RealisticGaitWrapper(gym.Wrapper):
    """
    將 Ant-v5 的 reward 改寫為步態導向：
    - 速度有上限（不是越快越好）
    - 強懲罰大幅動作（鼓勵省力）
    - 獎勵四足交替著地（trot 步態）
    - 維持軀幹姿態穩定
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
    ):
        super().__init__(env)
        self.target_speed = target_speed
        self.ctrl_weight = ctrl_weight
        self.gait_weight = gait_weight
        self.posture_weight = posture_weight
        self.alive_weight = alive_weight
        self.contact_threshold = contact_threshold
        self.prev_contacts = np.zeros(4)
        self.foot_body_ids = FOOT_BODY_IDS

    def reset(self, **kwargs):
        self.prev_contacts = np.zeros(4)
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, _orig_reward, terminated, truncated, info = self.env.step(action)

        contacts = self._get_foot_contacts()
        # Ant-v5 的 info 沒有 "is_healthy" 鍵（已實測確認），用 terminated 推導：
        # 這一步若導致 episode 終止（摔倒/翻覆），代表這一步把自己摔成不健康狀態
        is_healthy = not terminated
        reward, components = self._compute_reward(action, info, is_healthy, contacts)

        info["foot_contacts"] = contacts
        info["reward_components"] = components
        info["original_reward"] = _orig_reward
        self.prev_contacts = contacts

        return obs, reward, terminated, truncated, info

    def _get_foot_contacts(self) -> np.ndarray:
        """回傳四隻腳的接觸狀態（0 或 1）。"""
        cfrc = self.env.unwrapped.data.cfrc_ext[self.foot_body_ids]
        contact_magnitudes = np.linalg.norm(cfrc, axis=1)
        return (contact_magnitudes > self.contact_threshold).astype(np.float32)

    def _compute_reward(self, action, info, is_healthy, contacts):
        # 1. 速度向目標靠近（不是越快越好）
        x_vel = info.get("x_velocity", 0.0)
        r_forward = -abs(x_vel - self.target_speed)

        # 2. 存活基本分（沿用原 env 的 healthy 判定）
        r_alive = self.alive_weight if is_healthy else 0.0

        # 3. 控制懲罰（加重版）
        r_ctrl = -self.ctrl_weight * float(np.sum(np.square(action)))

        # 4. 步態 reward（核心）：trot 對角線同步
        diag1_sync = 1.0 - abs(contacts[0] - contacts[3])  # FL vs BR
        diag2_sync = 1.0 - abs(contacts[1] - contacts[2])  # FR vs BL
        cross_pattern = abs(contacts[0] - contacts[1])     # FL vs FR 應反相
        r_gait = self.gait_weight * (0.4 * diag1_sync + 0.4 * diag2_sync + 0.2 * cross_pattern)

        # 5. 姿態 reward（軀幹高度）
        torso_z = self.env.unwrapped.data.qpos[2]
        r_posture = -self.posture_weight * abs(torso_z - 0.6)

        total = r_forward + r_alive + r_ctrl + r_gait + r_posture
        components = {
            "forward": r_forward,
            "alive": r_alive,
            "ctrl": r_ctrl,
            "gait": r_gait,
            "posture": r_posture,
            "x_velocity": x_vel,
            "torso_z": torso_z,
        }
        return float(total), components
