from __future__ import annotations

import torch
from tensordict import TensorDict


class ReplayBuffer:
    def __init__(self, obs, num_envs: int, capacity: int, storage_device: str = "cpu", sample_device: str = "cpu"):
        self.capacity = int(capacity)
        self.num_envs = int(num_envs)
        self.storage_device = torch.device(storage_device)
        self.sample_device = torch.device(sample_device)
        self.ptr = 0
        self.size = 0

        self.obs = {
            key: torch.empty((self.capacity, *value.shape[1:]), dtype=value.dtype, device=self.storage_device)
            for key, value in obs.items()
        }
        self.next_obs = {
            key: torch.empty((self.capacity, *value.shape[1:]), dtype=value.dtype, device=self.storage_device)
            for key, value in obs.items()
        }
        self.actions = None
        self.rewards = torch.empty((self.capacity, 1), dtype=torch.float32, device=self.storage_device)
        self.dones = torch.empty((self.capacity, 1), dtype=torch.float32, device=self.storage_device)
        self.time_outs = torch.empty((self.capacity, 1), dtype=torch.float32, device=self.storage_device)

    def __len__(self) -> int:
        return self.size

    def add(self, obs, actions: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor, next_obs, extras: dict):
        actions = actions.detach().to(self.storage_device)
        if self.actions is None:
            self.actions = torch.empty((self.capacity, actions.shape[-1]), dtype=actions.dtype, device=self.storage_device)

        rewards = rewards.detach().reshape(-1, 1).to(self.storage_device)
        dones = dones.detach().reshape(-1, 1).to(dtype=torch.float32, device=self.storage_device)
        time_outs = extras.get("time_outs", torch.zeros_like(dones, dtype=torch.bool))
        time_outs = time_outs.detach().reshape(-1, 1).to(dtype=torch.float32, device=self.storage_device)

        n = actions.shape[0]
        indices = (torch.arange(n, device=self.storage_device) + self.ptr) % self.capacity
        for key in self.obs:
            self.obs[key][indices].copy_(obs[key].detach().to(self.storage_device))
            self.next_obs[key][indices].copy_(next_obs[key].detach().to(self.storage_device))
        self.actions[indices].copy_(actions)
        self.rewards[indices].copy_(rewards)
        self.dones[indices].copy_(dones)
        self.time_outs[indices].copy_(time_outs)

        self.ptr = int((self.ptr + n) % self.capacity)
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size: int) -> dict:
        if self.size <= 0:
            raise RuntimeError("Cannot sample from an empty replay buffer.")
        indices = torch.randint(0, self.size, (int(batch_size),), device=self.storage_device)
        obs = TensorDict(
            {key: value[indices].to(self.sample_device) for key, value in self.obs.items()},
            batch_size=[len(indices)],
            device=self.sample_device,
        )
        next_obs = TensorDict(
            {key: value[indices].to(self.sample_device) for key, value in self.next_obs.items()},
            batch_size=[len(indices)],
            device=self.sample_device,
        )
        return {
            "obs": obs,
            "actions": self.actions[indices].to(self.sample_device),
            "rewards": self.rewards[indices].to(self.sample_device),
            "dones": self.dones[indices].to(self.sample_device),
            "time_outs": self.time_outs[indices].to(self.sample_device),
            "next_obs": next_obs,
        }
