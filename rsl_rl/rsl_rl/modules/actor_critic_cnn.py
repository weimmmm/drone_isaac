# Copyright (c) 2026
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.networks import MLP, EmpiricalNormalization


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


class ActorCriticCnn(nn.Module):
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
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        cnn_channels=[16, 32, 64],
        cnn_kernel_sizes=[5, 3, 3],
        cnn_strides=[2, 2, 1],
        state_hidden_dims=[128],
        image_latent_dim=128,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticCnn.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        self.obs_groups = obs_groups
        self.actor_obs_normalization = actor_obs_normalization
        self.critic_obs_normalization = critic_obs_normalization
        self.noise_std_type = noise_std_type

        actor_state_dim, actor_image_shape = self._infer_group_shapes(obs_groups["policy"], obs)
        critic_state_dim, critic_image_shape = self._infer_group_shapes(obs_groups["critic"], obs)

        self.actor_state_normalizer = (
            EmpiricalNormalization(actor_state_dim) if actor_obs_normalization and actor_state_dim > 0 else nn.Identity()
        )
        self.critic_state_normalizer = (
            EmpiricalNormalization(critic_state_dim)
            if critic_obs_normalization and critic_state_dim > 0
            else nn.Identity()
        )

        self.actor_state_encoder, actor_state_out_dim = self._build_state_encoder(
            actor_state_dim, state_hidden_dims, activation
        )
        self.critic_state_encoder, critic_state_out_dim = self._build_state_encoder(
            critic_state_dim, state_hidden_dims, activation
        )

        self.actor_image_encoder, actor_image_out_dim = self._build_image_encoder(
            actor_image_shape, cnn_channels, cnn_kernel_sizes, cnn_strides, activation, image_latent_dim
        )
        self.critic_image_encoder, critic_image_out_dim = self._build_image_encoder(
            critic_image_shape, cnn_channels, cnn_kernel_sizes, cnn_strides, activation, image_latent_dim
        )

        actor_input_dim = actor_state_out_dim + actor_image_out_dim
        critic_input_dim = critic_state_out_dim + critic_image_out_dim

        self.actor = MLP(actor_input_dim, num_actions, actor_hidden_dims, activation)
        self.critic = MLP(critic_input_dim, 1, critic_hidden_dims, activation)

        print(f"Actor CNN-MLP: {self.actor}")
        print(f"Critic CNN-MLP: {self.critic}")

        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        self.distribution = None
        Normal.set_default_validate_args(False)

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
        mean = self.actor(self.get_actor_features(obs))
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        self.distribution = Normal(mean, std)

    def act(self, obs, **kwargs):
        self.update_distribution(obs)
        return self.distribution.sample()

    def act_inference(self, obs):
        return self.actor(self.get_actor_features(obs))

    def evaluate(self, obs, **kwargs):
        return self.critic(self.get_critic_features(obs))

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs):
        if self.actor_obs_normalization:
            actor_state, _ = self._split_groups(obs, self.obs_groups["policy"])
            if actor_state is not None:
                self.actor_state_normalizer.update(actor_state)
        if self.critic_obs_normalization:
            critic_state, _ = self._split_groups(obs, self.obs_groups["critic"])
            if critic_state is not None:
                self.critic_state_normalizer.update(critic_state)

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict=strict)
        return True

    def get_actor_features(self, obs):
        state, image = self._split_groups(obs, self.obs_groups["policy"])
        return self._encode_inputs(state, image, self.actor_state_normalizer, self.actor_state_encoder, self.actor_image_encoder)

    def get_critic_features(self, obs):
        state, image = self._split_groups(obs, self.obs_groups["critic"])
        return self._encode_inputs(
            state,
            image,
            self.critic_state_normalizer,
            self.critic_state_encoder,
            self.critic_image_encoder,
        )

    def _encode_inputs(self, state, image, normalizer, state_encoder, image_encoder):
        features = []
        if state is not None:
            state = normalizer(state)
            features.append(state_encoder(state))
        if image is not None:
            features.append(image_encoder(image))
        return torch.cat(features, dim=-1)

    def _split_groups(self, obs, group_names):
        state_tensors = []
        image_tensors = []
        for group_name in group_names:
            tensor = obs[group_name]
            if tensor.ndim == 2:
                state_tensors.append(tensor)
            elif tensor.ndim == 4:
                image_tensors.append(tensor)
            else:
                state_tensors.append(tensor.flatten(start_dim=1))
        state = torch.cat(state_tensors, dim=-1) if state_tensors else None
        image = torch.cat(image_tensors, dim=1) if image_tensors else None
        return state, image

    def _infer_group_shapes(self, group_names, obs):
        state_dim = 0
        image_shape = None
        for group_name in group_names:
            shape = obs[group_name].shape[1:]
            if len(shape) == 1:
                state_dim += shape[0]
            elif len(shape) == 3:
                if image_shape is None:
                    image_shape = list(shape)
                else:
                    image_shape[0] += shape[0]
            else:
                state_dim += int(torch.tensor(shape).prod().item())
        return state_dim, tuple(image_shape) if image_shape is not None else None

    def _build_state_encoder(self, input_dim, hidden_dims, activation):
        if input_dim == 0:
            return nn.Identity(), 0
        if not hidden_dims:
            return nn.Identity(), input_dim
        if len(hidden_dims) == 1:
            return nn.Sequential(nn.Linear(input_dim, hidden_dims[0]), _activation(activation)), hidden_dims[0]
        return MLP(input_dim, hidden_dims[-1], hidden_dims[:-1], activation), hidden_dims[-1]

    def _build_image_encoder(self, image_shape, channels, kernels, strides, activation, image_latent_dim):
        if image_shape is None:
            return nn.Identity(), 0

        if not (len(channels) == len(kernels) == len(strides)):
            raise ValueError("cnn_channels, cnn_kernel_sizes and cnn_strides must have the same length.")

        layers = []
        in_channels = image_shape[0]
        for out_channels, kernel_size, stride in zip(channels, kernels, strides):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride))
            layers.append(_activation(activation))
            in_channels = out_channels
        layers.append(nn.Flatten())
        conv = nn.Sequential(*layers)

        with torch.no_grad():
            sample = torch.zeros(1, *image_shape)
            conv_out_dim = conv(sample).shape[-1]

        encoder = nn.Sequential(conv, nn.Linear(conv_out_dim, image_latent_dim), _activation(activation))
        return encoder, image_latent_dim
