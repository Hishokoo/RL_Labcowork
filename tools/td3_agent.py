import copy
import torch
import torch.nn.functional as F
from tools.networks import Actor, Critic
from tools.replay_buffer import ReplayBuffer


class TD3Agent:
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        max_action: float,
        device: torch.device,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        policy_delay: int = 2,
        noise_std: float = 0.2,
        noise_clip: float = 0.5,
    ):
        self.device = device
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.policy_delay = policy_delay
        self.noise_std = noise_std
        self.noise_clip = noise_clip
        self.total_it = 0

        self.actor = Actor(obs_dim, act_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = Critic(obs_dim, act_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    @torch.no_grad()
    def select_action(self, obs) -> "np.ndarray":
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        return self.actor(obs_t).cpu().numpy().flatten()

    def train(self, replay_buffer: ReplayBuffer, batch_size: int = 256):
        self.total_it += 1
        obs, action, next_obs, reward, done = replay_buffer.sample(batch_size, self.device)

        # ── Critic update ────────────────────────────────────────────────
        with torch.no_grad():
            # Target policy smoothing: add clipped noise to target action
            noise = (torch.randn_like(action) * self.noise_std).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_action = (self.actor_target(next_obs) + noise).clamp(
                -self.max_action, self.max_action
            )
            q1_t, q2_t = self.critic_target(next_obs, next_action)
            q_target = reward + (1.0 - done) * self.gamma * torch.min(q1_t, q2_t)

        q1, q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # ── Delayed actor + target update ────────────────────────────────
        actor_loss_val = None
        if self.total_it % self.policy_delay == 0:
            actor_loss = -self.critic.q1_value(obs, self.actor(obs)).mean()
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()
            actor_loss_val = actor_loss.item()

            self._soft_update(self.critic, self.critic_target)
            self._soft_update(self.actor, self.actor_target)

        return critic_loss.item(), actor_loss_val

    def _soft_update(self, src: torch.nn.Module, tgt: torch.nn.Module):
        for p, tp in zip(src.parameters(), tgt.parameters()):
            tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)

    def save(self, path: str):
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "critic_target": self.critic_target.state_dict(),
            },
            path,
        )

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.actor_target.load_state_dict(ckpt["actor_target"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
