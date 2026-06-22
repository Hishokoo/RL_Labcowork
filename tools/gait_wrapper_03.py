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
    將 Ant-v5 的 reward 改寫為步態導向：
    - 速度有上限（不是越快越好）
    - 強懲罰大幅動作（鼓勵省力）
    - 獎勵四足交替著地（trot 步態）
    - 維持軀幹姿態穩定

    gait_mode 控制步態 reward 的公式（向後相容，預設沿用 03）：
    - "legacy"   ：03 的逐幀對角線同步。站著不動的 r_gait 反而偏高（靜態可作弊）。
    - "antiphase"：以對角線「反相」為主導（見 gait_metrics.anti_phase），
                   靜態姿勢拿不到分，逼出真正的交替踏步。04 起採用。

    forward_mode 控制速度 reward（與「站著 attractor」直接相關）：
    - "deviation"：03 的 -|x_vel - target|。站著時 = -1.0，等於一個 speed penalty——
                   而 markdown/ant_v5_attractor_fix.md 實測這類懲罰會逼出「快速摔倒擺爛」。
    - "progress" ：max(0, min(x_vel, target))。站著 = 0（不是負的）、達到目標才給正分，
                   讓「走路」成為唯一的正收益來源，符合 02 已驗證的 healthy_reward=0 思路。04 起採用。

    smooth_weight / tilt_weight 預設 0（不影響 03）：
    - smooth_weight：懲罰相鄰動作差（jerk），抓「抽搐 / 慣性甩動」。
    - tilt_weight  ：懲罰軀幹傾斜（直立度），補足只看高度 z 的姿態 reward。

    gait_mode "antiphase_gated"（05 起，修正 04 的站著 attractor）：
    - 04 的 "antiphase" 在靜態站姿仍給 r_gait≈0.77（同對角線兩腳同步的 intra 項），1M 實測
      agent 收斂到站著不動。"antiphase_gated" 把整個步態 reward 乘上 anti_phase 當 gate：
      r_gait = gait_weight · anti_phase · (0.5 + 0.25·intra1 + 0.25·intra2)。
      站著 anti_phase≈0 → r_gait≈0（站著零步態收益），真 trot 單腳支撐瞬間 → 滿分。
    forward_weight：速度 reward 的權重（預設 1.0 = 03/04 行為）。05 調高（如 2.0）給更強的
      「要移動才有分」拉力，配合 gated gait 把 agent 推出站著盆地。
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
        # ctrl 排程（None=關閉，用固定 ctrl_weight）。(t0, t1, c0, c1)：step≤t0→c0、
        # t0..t1 線性 c0→c1、>t1→c1。用來在「從零訓練」時先放鬆 ctrl 讓 agent 學會站穩/移動，
        # 再漸進加重回到目標，避免早期隨機動作配重 ctrl 產生巨大負 reward → fast-fall。
        if ctrl_schedule is not None:
            t0, t1, c0, c1 = ctrl_schedule
            if not (0 <= t0 <= t1):
                raise ValueError(f"ctrl_schedule 須 0≤t0≤t1：{ctrl_schedule}")
        self.ctrl_schedule = ctrl_schedule
        self._gstep = 0  # 累積訓練步數（跨 episode 不重置）——n_envs=1 下 = 全域訓練步
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
        # Ant-v5 的 info 沒有 "is_healthy" 鍵（已實測確認），用 terminated 推導：
        # 這一步若導致 episode 終止（摔倒/翻覆），代表這一步把自己摔成不健康狀態
        is_healthy = not terminated
        reward, components = self._compute_reward(action, info, is_healthy, contacts)

        info["foot_contacts"] = contacts
        info["reward_components"] = components
        info["original_reward"] = _orig_reward
        self.prev_contacts = contacts
        self.prev_action = np.asarray(action, dtype=np.float64)
        self._gstep += 1  # 累積訓練步（給 ctrl_schedule 用）

        return obs, reward, terminated, truncated, info

    def _get_foot_contacts(self) -> np.ndarray:
        """回傳四隻腳的接觸狀態（0 或 1）。"""
        cfrc = self.env.unwrapped.data.cfrc_ext[self.foot_body_ids]
        contact_magnitudes = np.linalg.norm(cfrc, axis=1)
        return (contact_magnitudes > self.contact_threshold).astype(np.float32)

    def _compute_reward(self, action, info, is_healthy, contacts):
        # 1. 速度 reward（不是越快越好；forward_mode 決定站著要不要被罰）
        x_vel = info.get("x_velocity", 0.0)
        if self.forward_mode == "progress":
            # 站著 = 0、達到 target 才滿分、超過 target 不再加分（速度上限仍在）
            r_forward = self.forward_weight * max(0.0, min(x_vel, self.target_speed))
        else:  # "deviation"：03 行為，偏離 target 即罰（站著 = -target_speed）
            r_forward = -self.forward_weight * abs(x_vel - self.target_speed)

        # 2. 存活基本分（沿用原 env 的 healthy 判定）
        r_alive = self.alive_weight if is_healthy else 0.0

        # 3. 控制懲罰（加重版）。ctrl_schedule 啟用時用漸進權重（早期放鬆防 fast-fall）。
        r_ctrl = -self._effective_ctrl_weight() * float(np.sum(np.square(action)))

        # 4. 步態 reward（核心）：trot 對角線交替
        if self.gait_mode == "antiphase_gated":
            # 用 anti_phase 當乘法 gate：靜態姿勢 anti_phase≈0 → r_gait≈0（站著零步態收益,
            # 修正 04 站著仍拿 0.77 的 attractor）。真 trot 單腳支撐瞬間 anti=1、intra=1 → 滿分。
            intra1 = 1.0 - abs(contacts[0] - contacts[3])  # FL, BR 同相
            intra2 = 1.0 - abs(contacts[1] - contacts[2])  # FR, BL 同相
            anti = gait_metrics.anti_phase(contacts)
            # base + intra_weight·(intra1+intra2)，base 自動補成「滿分=1」。
            # 提高 intra_weight（如 0.35）= 更獎勵「同對角兩腳同步」(diagonal_sync)，
            # 整體仍被 anti 乘法閘住（站著 anti≈0 → r_gait≈0），不會復活站著 attractor。
            base = 1.0 - 2.0 * self.intra_weight
            r_gait = self.gait_weight * anti * (base + self.intra_weight * intra1 + self.intra_weight * intra2)
        elif self.gait_mode == "antiphase":
            # 反相為主導（0.6）：兩條對角線「一抬一踏」才給分，靜態姿勢 = 0。
            # 同步為輔（各 0.2）：維持同一對角線內兩腳協調，避免單腳亂跳。
            # 註：靜態時 intra 項仍給 0.4·gait_weight≈0.8（04 站著 attractor 的成因），改用 *_gated。
            intra1 = 1.0 - abs(contacts[0] - contacts[3])  # FL, BR 同相
            intra2 = 1.0 - abs(contacts[1] - contacts[2])  # FR, BL 同相
            anti = gait_metrics.anti_phase(contacts)
            r_gait = self.gait_weight * (0.2 * intra1 + 0.2 * intra2 + 0.6 * anti)
        else:  # "legacy"：沿用 03 的逐幀對角線同步公式
            diag1_sync = 1.0 - abs(contacts[0] - contacts[3])  # FL vs BR
            diag2_sync = 1.0 - abs(contacts[1] - contacts[2])  # FR vs BL
            cross_pattern = abs(contacts[0] - contacts[1])     # FL vs FR 應反相
            r_gait = self.gait_weight * (0.4 * diag1_sync + 0.4 * diag2_sync + 0.2 * cross_pattern)

        # 4b. 柔和速度 gate（只作用在 r_gait，不乘整個 reward）：
        # 修正 legacy 的唯一缺點——站著（四腳著地）時 diag1=diag2=1 → r_gait=0.8·gait_weight 是
        # 站著 attractor。用 smoothstep 把 r_gait 在低速時平滑歸零：x_vel≥gate 幾乎完全恢復、
        # x_vel=0 時為 0。與 forward_gated 的差別：(1) 只 gate r_gait 不 gate forward/alive，
        # 早期走不動仍有 alive/forward 訊號（學習穩定）；(2) smoothstep 連續無跳變，不放大 0/1
        # 接觸訊號（不誘發頓腳）。gait_speed_gate=0 時關閉，r_gait 完全等於 03。
        if self.gait_speed_gate > 0.0:
            p = float(np.clip(max(x_vel, 0.0) / self.gait_speed_gate, 0.0, 1.0))
            r_gait *= p * p * (3.0 - 2.0 * p)

        # 5. 姿態 reward（軀幹高度）
        torso_z = self.env.unwrapped.data.qpos[2]
        r_posture = -self.posture_weight * abs(torso_z - 0.6)

        # 6. 動作平滑懲罰（jerk）：相鄰動作差的平方和。weight=0 時為 0，不影響 legacy。
        action_arr = np.asarray(action, dtype=np.float64)
        jerk = float(np.sum(np.square(action_arr - self.prev_action)))
        r_smooth = -self.smooth_weight * jerk

        # 7. 軀幹直立懲罰：偏離垂直越多罰越重。weight=0 時為 0，不影響 legacy。
        qpos = self.env.unwrapped.data.qpos
        upright = gait_metrics.uprightness(qpos)
        r_tilt = -self.tilt_weight * (1.0 - upright)

        if self.reward_structure == "forward_gated":
            # 正向 reward 全部以「前進」為閘門：total_正向 = forward_progress·(1 + r_gait)。
            # 不前進 → forward_progress≈0 → 連步態 bonus 都歸零,站著/原地踏步都拿不到正分
            # （修正 04 站著、05 原地踏步的鑽洞）。懲罰（ctrl/smooth/tilt/posture）仍照算。
            # r_gait 在此當品質乘子（搭 antiphase_gated 時 ∈[0, gait_weight]）。
            # forward_gate_shape 決定速度形狀：
            #   "cap" ：max(0, min(x_vel, target))，到目標即滿分、超過不再加分也不罰（會超速）。
            #   "tent"：target·max(0, 1−|x_vel−target|/target)，恰在目標滿分、太慢或太快都遞減、
            #           但 ≥0（不為負 → 不摔死）。用來壓住 05 的超速（x_vel 1.37 → 拉回目標 1.0）。
            if self.forward_gate_shape == "tent":
                forward_progress = self.target_speed * max(
                    0.0, 1.0 - abs(x_vel - self.target_speed) / self.target_speed)
            else:
                forward_progress = max(0.0, min(x_vel, self.target_speed))
            gait_contrib = forward_progress * r_gait
            total = forward_progress + gait_contrib + r_alive + r_ctrl + r_smooth + r_tilt + r_posture
            r_forward = forward_progress   # 記錄用：實際前進量
            r_gait = gait_contrib          # 記錄用：步態的實際貢獻（已被前進閘門）
        else:
            total = r_forward + r_alive + r_ctrl + r_gait + r_posture + r_smooth + r_tilt
        components = {
            "forward": r_forward,
            "alive": r_alive,
            "ctrl": r_ctrl,
            "gait": r_gait,
            "posture": r_posture,
            "smooth": r_smooth,
            "tilt": r_tilt,
            "x_velocity": x_vel,
            "torso_z": torso_z,
            "uprightness": upright,
            "anti_phase": gait_metrics.anti_phase(contacts),
        }
        return float(total), components
