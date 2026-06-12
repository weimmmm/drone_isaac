# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.networks import MLP, EmpiricalNormalization


class MultiHeadObsEncoder(nn.Module):
    """Encode static-obstacle, dynamic-obstacle, and remaining observations separately."""

    def __init__(
        self,
        input_dim: int,
        static_obs_start_idx: int,
        static_obs_dim: int,
        dynamic_obs_start_idx: int,
        dynamic_obs_dim: int,
        static_encoder_hidden_dims: list[int],
        dynamic_encoder_hidden_dims: list[int],
        other_encoder_hidden_dims: list[int],
        static_latent_dim: int,
        dynamic_latent_dim: int,
        other_latent_dim: int,
        activation: str,
    ):
        super().__init__()

        self.static_slice = self._make_slice(input_dim, static_obs_start_idx, static_obs_dim, "static")
        self.dynamic_slice = self._make_slice(input_dim, dynamic_obs_start_idx, dynamic_obs_dim, "dynamic")

        used = torch.zeros(input_dim, dtype=torch.bool)
        used[self.static_slice] = True
        if used[self.dynamic_slice].any():
            raise ValueError("Static and dynamic observation slices overlap.")
        used[self.dynamic_slice] = True

        other_indices = torch.nonzero(~used, as_tuple=False).squeeze(-1)
        if other_indices.numel() == 0:
            raise ValueError("Other observation head has zero input dimensions.")
        self.register_buffer("_other_indices", other_indices, persistent=False)

        self.static_encoder = MLP(static_obs_dim, static_latent_dim, static_encoder_hidden_dims, activation)
        self.dynamic_encoder = MLP(dynamic_obs_dim, dynamic_latent_dim, dynamic_encoder_hidden_dims, activation)
        self.other_encoder = MLP(int(other_indices.numel()), other_latent_dim, other_encoder_hidden_dims, activation)
        self.output_dim = static_latent_dim + dynamic_latent_dim + other_latent_dim

    @staticmethod
    def _make_slice(input_dim: int, start_idx: int, dim: int, name: str) -> slice:
        if dim <= 0:
            raise ValueError(f"{name}_obs_dim must be positive, got {dim}.")
        if start_idx < 0 or start_idx + dim > input_dim:
            raise ValueError(
                f"{name} observation slice [{start_idx}, {start_idx + dim}) is outside input dim {input_dim}."
            )
        return slice(start_idx, start_idx + dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        static_feature = self.static_encoder(obs[:, self.static_slice])
        dynamic_feature = self.dynamic_encoder(obs[:, self.dynamic_slice])
        other_feature = self.other_encoder(obs.index_select(dim=-1, index=self._other_indices))
        return torch.cat([static_feature, dynamic_feature, other_feature], dim=-1)


class ActorCriticMultiHeadObs(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
        static_encoder_hidden_dims=[64, 64],
        dynamic_encoder_hidden_dims=[128, 64],
        other_encoder_hidden_dims=[64, 64],
        static_latent_dim=64,
        dynamic_latent_dim=64,
        other_latent_dim=64,
        static_obs_start_idx=15,
        static_obs_dim=32,
        dynamic_obs_start_idx=47,
        dynamic_obs_dim=50,
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticMultiHeadObs.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        self.obs_groups = obs_groups
        num_actor_obs = self._sum_flat_obs_dim(obs, obs_groups["policy"])
        num_critic_obs = self._sum_flat_obs_dim(obs, obs_groups["critic"])
        if num_actor_obs != num_critic_obs:
            raise ValueError(
                "ActorCriticMultiHeadObs expects policy and critic observations to have the same flat dimension. "
                f"Got policy={num_actor_obs}, critic={num_critic_obs}."
            )

        self.obs_encoder = MultiHeadObsEncoder(
            input_dim=num_actor_obs,
            static_obs_start_idx=static_obs_start_idx,
            static_obs_dim=static_obs_dim,
            dynamic_obs_start_idx=dynamic_obs_start_idx,
            dynamic_obs_dim=dynamic_obs_dim,
            static_encoder_hidden_dims=static_encoder_hidden_dims,
            dynamic_encoder_hidden_dims=dynamic_encoder_hidden_dims,
            other_encoder_hidden_dims=other_encoder_hidden_dims,
            static_latent_dim=static_latent_dim,
            dynamic_latent_dim=dynamic_latent_dim,
            other_latent_dim=other_latent_dim,
            activation=activation,
        )

        self.actor = MLP(self.obs_encoder.output_dim, num_actions, actor_hidden_dims, activation)
        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()

        self.critic = MLP(self.obs_encoder.output_dim, 1, critic_hidden_dims, activation)
        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs)
        else:
            self.critic_obs_normalizer = torch.nn.Identity()

        print(
            "Multi-head observation split: "
            f"other={num_actor_obs - static_obs_dim - dynamic_obs_dim}, "
            f"static={static_obs_dim}, dynamic={dynamic_obs_dim}, "
            f"latent={self.obs_encoder.output_dim}"
        )
        print(f"Observation encoder: {self.obs_encoder}")
        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        self.distribution = None
        Normal.set_default_validate_args(False)

    @staticmethod
    def _sum_flat_obs_dim(obs, obs_group_names: list[str]) -> int:
        flat_dim = 0
        for obs_group in obs_group_names:
            assert len(obs[obs_group].shape) == 2, "ActorCriticMultiHeadObs only supports 1D observations."
            flat_dim += obs[obs_group].shape[-1]
        return flat_dim

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, obs):
        feature = self.obs_encoder(obs)
        mean = self.actor(feature)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        self.distribution = Normal(mean, std)

    def act(self, obs, **kwargs):
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        self.update_distribution(obs)
        return self.distribution.sample()

    def act_inference(self, obs):
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        return self.actor(self.obs_encoder(obs))

    def evaluate(self, obs, **kwargs):
        obs = self.get_critic_obs(obs)
        obs = self.critic_obs_normalizer(obs)
        return self.critic(self.obs_encoder(obs))

    def get_actor_obs(self, obs):
        obs_list = []
        for obs_group in self.obs_groups["policy"]:
            obs_list.append(obs[obs_group])
        return torch.cat(obs_list, dim=-1)

    def get_critic_obs(self, obs):
        obs_list = []
        for obs_group in self.obs_groups["critic"]:
            obs_list.append(obs[obs_group])
        return torch.cat(obs_list, dim=-1)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs):
        if self.actor_obs_normalization:
            actor_obs = self.get_actor_obs(obs)
            self.actor_obs_normalizer.update(actor_obs)
        if self.critic_obs_normalization:
            critic_obs = self.get_critic_obs(obs)
            self.critic_obs_normalizer.update(critic_obs)

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict=strict)
        return True
