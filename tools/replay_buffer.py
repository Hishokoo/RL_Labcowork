import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, obs_dim: int, act_dim: int, max_size: int = 1_000_000):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.obs      = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.action   = np.zeros((max_size, act_dim), dtype=np.float32)
        self.next_obs = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.reward   = np.zeros((max_size, 1),       dtype=np.float32)
        self.done     = np.zeros((max_size, 1),       dtype=np.float32)

    def add(self, obs, action, next_obs, reward, done):
        self.obs[self.ptr]      = obs
        self.action[self.ptr]   = action
        self.next_obs[self.ptr] = next_obs
        self.reward[self.ptr]   = reward
        self.done[self.ptr]     = done
        self.ptr  = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int, device: torch.device):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.obs[idx]).to(device),
            torch.FloatTensor(self.action[idx]).to(device),
            torch.FloatTensor(self.next_obs[idx]).to(device),
            torch.FloatTensor(self.reward[idx]).to(device),
            torch.FloatTensor(self.done[idx]).to(device),
        )

    def __len__(self):
        return self.size
