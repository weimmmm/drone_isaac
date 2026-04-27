# Copyright (c) 2026
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.networks import EmpiricalNormalization, MLP, Memory
from rsl_rl.utils import unpad_trajectories


def _activation(name: str) -> nn.Module:
    activations = {
        "elu": nn.ELU,
        "relu": nn.ReLU,
        "selu": nn.SELU,
        "crelu": nn.ReLU,
        "lrelu": nn.LeakyReLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
    }
    if name not in activations:
        raise ValueError(f"Unsupported activation: {name}")
    return activations[name]()


class ActorCriticRayRecurrent(nn.Module):
    """REASAN-style recurrent actor-critic:
    state encoder + 1D circular ray encoder + LSTM memory.
    """

    is_recurrent = True

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        cnn_channels=[8, 16, 32],
        cnn_kernel_sizes=[5, 3, 3],
        cnn_strides=[2, 2, 2],
        state_hidden_dims=[128, 128],
        image_latent_dim=128,
        rnn_type="lstm",
        rnn_hidden_dim=128,
        rnn_num_layers=1,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticRayRecurrent.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()]),
                flush=True,
            )
        super().__init__()

        self.obs_groups = obs_groups
        self.actor_obs_normalization = actor_obs_normalization
        self.critic_obs_normalization = critic_obs_normalization
        self.noise_std_type = noise_std_type
        self._group_shapes = {name: tuple(obs[name].shape[1:]) for name in obs.keys()}

        actor_state_dim, actor_ray_shape = self._infer_group_shapes(obs_groups["policy"])
        num_critic_obs = 0
        for group_name in obs_groups["critic"]:
            num_critic_obs += int(torch.tensor(self._group_shapes[group_name]).prod().item())

        self.actor_state_normalizer = (
            EmpiricalNormalization(actor_state_dim) if actor_obs_normalization and actor_state_dim > 0 else nn.Identity()
        )
        self.actor_ray_normalizer = (
            EmpiricalNormalization(actor_ray_shape) if actor_obs_normalization and actor_ray_shape is not None else nn.Identity()
        )
        self.critic_obs_normalization = critic_obs_normalization
        self.critic_obs_normalizer = (
            EmpiricalNormalization(num_critic_obs) if critic_obs_normalization and num_critic_obs > 0 else nn.Identity()
        )

        self.actor_state_encoder, actor_state_out_dim = self._build_state_encoder(
            actor_state_dim, state_hidden_dims, activation
        )
        self.actor_ray_encoder, actor_ray_out_dim = self._build_ray_encoder(
            actor_ray_shape, cnn_channels, cnn_kernel_sizes, cnn_strides, activation, image_latent_dim
        )

        actor_encoder_dim = actor_state_out_dim + actor_ray_out_dim

        self.memory_a = Memory(actor_encoder_dim, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_dim)
        # Keep critic memory only for recurrent runner/storage compatibility; value prediction itself stays teacher-aligned MLP.
        self.memory_c = Memory(num_critic_obs, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_dim)
        self.actor = MLP(rnn_hidden_dim, num_actions, actor_hidden_dims, activation)
        self.critic = MLP(num_critic_obs, 1, critic_hidden_dims, activation)

        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        self.distribution = None
        Normal.set_default_validate_args(False)

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def reset(self, dones=None):
        self.memory_a.reset(dones)
        self.memory_c.reset(dones)

    def forward(self):
        raise NotImplementedError

    def update_distribution(self, obs):
        mean = self.actor(obs)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        self.distribution = Normal(mean, std)

    def act(self, obs, masks=None, hidden_states=None):
        features = self.get_actor_features(obs)
        out_mem = self.memory_a(features, masks, hidden_states).squeeze(0)
        self.update_distribution(out_mem)
        return self.distribution.sample()

    def act_inference(self, obs):
        features = self.get_actor_features(obs)
        out_mem = self.memory_a(features).squeeze(0)
        return self.actor(out_mem)

    def evaluate(self, obs, masks=None, hidden_states=None):
        critic_obs = self.get_critic_obs(obs)
        leading_shape = critic_obs.shape[:-1]
        flat_critic_obs = critic_obs.reshape(-1, critic_obs.shape[-1])
        flat_critic_obs = self.critic_obs_normalizer(flat_critic_obs)
        critic_obs = flat_critic_obs.reshape(*leading_shape, -1)
        # Keep critic hidden-state bookkeeping compatible with recurrent PPO, but compute values with a plain MLP.
        _ = self.memory_c(critic_obs, masks, hidden_states).squeeze(0)
        if masks is not None:
            critic_obs = unpad_trajectories(critic_obs, masks).squeeze(0)
        return self.critic(critic_obs)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def get_hidden_states(self):
        return self.memory_a.hidden_states, self.memory_c.hidden_states

    def update_normalization(self, obs):
        if self.actor_obs_normalization:
            actor_state, actor_ray = self._split_groups(obs, self.obs_groups["policy"])
            if actor_state is not None:
                self.actor_state_normalizer.update(actor_state.reshape(-1, actor_state.shape[-1]))
            if actor_ray is not None:
                self.actor_ray_normalizer.update(actor_ray.reshape(-1, actor_ray.shape[-2], actor_ray.shape[-1]))
        if self.critic_obs_normalization:
            critic_obs = self.get_critic_obs(obs)
            self.critic_obs_normalizer.update(critic_obs.reshape(-1, critic_obs.shape[-1]))

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict=strict)
        return True

    def get_actor_features(self, obs):
        state, ray = self._split_groups(obs, self.obs_groups["policy"])
        return self._encode_inputs(
            state, ray, self.actor_state_normalizer, self.actor_ray_normalizer, self.actor_state_encoder, self.actor_ray_encoder
        )

    def get_critic_obs(self, obs):
        obs_list = []
        for obs_group in self.obs_groups["critic"]:
            obs_list.append(obs[obs_group].reshape(*obs[obs_group].shape[:-1], -1))
        return torch.cat(obs_list, dim=-1)

    def _encode_inputs(self, state, ray, state_normalizer, ray_normalizer, state_encoder, ray_encoder):
        features = []
        leading_shape = None
        if state is not None:
            leading_shape = state.shape[:-1]
            flat_state = state.reshape(-1, state.shape[-1])
            flat_state = state_normalizer(flat_state)
            features.append(state_encoder(flat_state).reshape(*leading_shape, -1))
        if ray is not None:
            if leading_shape is None:
                leading_shape = ray.shape[:-2]
            flat_ray = ray.reshape(-1, ray.shape[-2], ray.shape[-1])
            flat_ray = ray_normalizer(flat_ray)
            features.append(ray_encoder(flat_ray).reshape(*leading_shape, -1))
        return torch.cat(features, dim=-1)

    def _split_groups(self, obs, group_names):
        state_tensors = []
        ray_tensors = []
        for group_name in group_names:
            tensor = obs[group_name]
            shape = self._group_shapes[group_name]
            if len(shape) == 2:
                ray_tensors.append(tensor.reshape(*tensor.shape[:-2], shape[0], shape[1]))
            else:
                state_tensors.append(tensor.reshape(*tensor.shape[:-1], -1))
        state = torch.cat(state_tensors, dim=-1) if state_tensors else None
        ray = torch.cat(ray_tensors, dim=-2) if ray_tensors else None
        return state, ray

    def _infer_group_shapes(self, group_names):
        state_dim = 0
        ray_shape = None
        for group_name in group_names:
            shape = self._group_shapes[group_name]
            if len(shape) == 1:
                state_dim += shape[0]
            elif len(shape) == 2:
                channels, num_rays = shape
                if ray_shape is None:
                    ray_shape = [channels, num_rays]
                else:
                    if ray_shape[1] != num_rays:
                        raise ValueError(
                            f"All ray groups must have same ray count, got {ray_shape[1]} and {num_rays} for {group_name}"
                        )
                    ray_shape[0] += channels
            else:
                state_dim += int(torch.tensor(shape).prod().item())
        return state_dim, tuple(ray_shape) if ray_shape is not None else None

    def _build_state_encoder(self, input_dim, hidden_dims, activation):
        if input_dim == 0:
            return nn.Identity(), 0
        if not hidden_dims:
            return nn.Identity(), input_dim
        if len(hidden_dims) == 1:
            return nn.Sequential(nn.Linear(input_dim, hidden_dims[0]), _activation(activation)), hidden_dims[0]
        return MLP(input_dim, hidden_dims[-1], hidden_dims[:-1], activation), hidden_dims[-1]

    def _build_ray_encoder(self, ray_shape, channels, kernels, strides, activation, ray_latent_dim):
        if ray_shape is None:
            return nn.Identity(), 0
        if not (len(channels) == len(kernels) == len(strides)):
            raise ValueError("cnn_channels, cnn_kernel_sizes and cnn_strides must have the same length.")

        conv_layers = []
        in_channels = ray_shape[0]
        for out_channels, kernel_size, stride in zip(channels, kernels, strides):
            padding = kernel_size // 2
            conv_layers.append(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                )
            )
            conv_layers.append(_activation(activation))
            in_channels = out_channels
        conv_layers.append(nn.AdaptiveAvgPool1d(15))
        conv_layers.append(nn.Flatten())
        conv = nn.Sequential(*conv_layers)

        with torch.no_grad():
            sample = torch.zeros(1, *ray_shape)
            conv_out_dim = conv(sample).shape[-1]

        encoder = nn.Sequential(
            conv,
            nn.Linear(conv_out_dim, 256),
            _activation(activation),
            nn.Linear(256, 256),
            _activation(activation),
            nn.Linear(256, ray_latent_dim),
            _activation(activation),
        )
        return encoder, ray_latent_dim
