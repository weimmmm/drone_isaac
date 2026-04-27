from __future__ import annotations

import gymnasium as gym
import torch
from tensordict import TensorDict

from rsl_rl.env import VecEnv

from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv


class RslRlVecEnvWrapper(VecEnv):
    def __init__(self, env: ManagerBasedRLEnv | DirectRLEnv | gym.Env, clip_actions: float | None = None):
        if not self._is_supported_env(env):
            raise ValueError(
                "The environment must inherit from ManagerBasedRLEnv/DirectRLEnv or expose the same vector-env "
                f"interface. Received: {type(env)}"
            )

        self.env = env
        self.clip_actions = clip_actions
        self.num_envs = self.unwrapped.num_envs
        self.device = self.unwrapped.device
        self.max_episode_length = self.unwrapped.max_episode_length

        if hasattr(self.unwrapped, "action_manager"):
            self.num_actions = self.unwrapped.action_manager.total_action_dim
        else:
            self.num_actions = gym.spaces.flatdim(self.unwrapped.single_action_space)

        if hasattr(self.unwrapped, "observation_manager"):
            policy_dim = self.unwrapped.observation_manager.group_obs_dim["policy"]
            if isinstance(policy_dim, list):
                self.num_obs = sum(int(torch.tensor(dim).prod().item()) for dim in policy_dim)
            else:
                self.num_obs = int(torch.tensor(policy_dim).prod().item())
        else:
            self.num_obs = gym.spaces.flatdim(self.unwrapped.single_observation_space["policy"])

        if (
            hasattr(self.unwrapped, "observation_manager")
            and "critic" in self.unwrapped.observation_manager.group_obs_dim
        ):
            self.num_privileged_obs = self.unwrapped.observation_manager.group_obs_dim["critic"][0]
        elif hasattr(self.unwrapped, "num_states") and "critic" in self.unwrapped.single_observation_space:
            self.num_privileged_obs = gym.spaces.flatdim(self.unwrapped.single_observation_space["critic"])
        else:
            self.num_privileged_obs = 0

        self._modify_action_space()
        self.env.reset()

    @property
    def cfg(self) -> object:
        return self.unwrapped.cfg

    @property
    def unwrapped(self) -> ManagerBasedRLEnv | DirectRLEnv | gym.Env:
        return self.env.unwrapped

    def get_observations(self) -> TensorDict:
        if hasattr(self.unwrapped, "observation_manager"):
            obs_dict = self.unwrapped.observation_manager.compute()
        else:
            obs_dict = self.unwrapped._get_observations()
        return self._to_tensordict(obs_dict)

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor):
        self.unwrapped.episode_length_buf = value

    def seed(self, seed: int = -1) -> int:
        return self.unwrapped.seed(seed)

    def reset(self) -> TensorDict:
        obs_dict, _ = self.env.reset()
        return self._to_tensordict(obs_dict)

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)

        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
        dones = (terminated | truncated).to(dtype=torch.long)
        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated

        return self._to_tensordict(obs_dict), rew, dones, extras

    def close(self):
        return self.env.close()

    def _modify_action_space(self):
        if self.clip_actions is None:
            return

        self.env.unwrapped.single_action_space = gym.spaces.Box(
            low=-self.clip_actions, high=self.clip_actions, shape=(self.num_actions,)
        )
        self.env.unwrapped.action_space = gym.vector.utils.batch_space(
            self.env.unwrapped.single_action_space, self.num_envs
        )

    def _to_tensordict(self, obs_dict: dict) -> TensorDict:
        if isinstance(obs_dict, TensorDict):
            return obs_dict
        if isinstance(obs_dict, dict):
            obs_dict = self._flatten_obs_dict(obs_dict)
        return TensorDict(obs_dict, batch_size=[self.num_envs], device=self.device)

    def _flatten_obs_dict(self, obs_dict: dict) -> dict:
        flat_obs = {}
        for key, value in obs_dict.items():
            if isinstance(value, dict):
                nested = self._flatten_obs_dict(value)
                for nested_key, nested_value in nested.items():
                    if nested_key in flat_obs:
                        raise KeyError(f"Duplicate observation key while flattening nested dict: {nested_key}")
                    flat_obs[nested_key] = nested_value
            else:
                flat_obs[key] = value
        return flat_obs

    @staticmethod
    def _is_supported_env(env: gym.Env) -> bool:
        unwrapped = env.unwrapped
        if isinstance(unwrapped, (ManagerBasedRLEnv, DirectRLEnv)):
            return True
        required_attrs = [
            "num_envs",
            "device",
            "max_episode_length",
            "cfg",
            "single_action_space",
            "single_observation_space",
        ]
        return all(hasattr(unwrapped, attr) for attr in required_attrs)
