# Copyright (c) 2026
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.networks import EmpiricalNormalization, MLP, Memory
from .actor_critic_vit_asymmetric import VitFlyImageEncoder, _activation


class ActorCriticVitRecurrentAsymmetric(nn.Module):
    """Asymmetric ViT+LSTM actor with low-dimensional recurrent critic.

    Actor path:
    - policy_state -> state encoder
    - policy_image -> ViT image encoder
    - concat -> LSTM memory -> linear action head

    Critic path:
    - critic observations stay low-dimensional
    - an LSTM memory is added for PPO recurrent compatibility
    - critic MLP keeps the same input size as the flattened critic observation so
      teacher critic weights can still be partially reused
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
        state_hidden_dims=[128],
        image_latent_dim=128,
        rnn_type="lstm",
        rnn_hidden_dim=128,
        rnn_num_layers=1,
        actor_cfg: dict | None = None,
        critic_cfg: dict | None = None,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticVitRecurrentAsymmetric.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()]),
                flush=True,
            )
        super().__init__()

        vit_resize_hw = [60, 90]
        vit_patch_sizes = [7, 3]
        vit_stage_strides = [4, 2]
        vit_paddings = [3, 1]
        vit_embed_dims = [32, 64]
        vit_num_layers = [2, 2]
        vit_reduction_ratios = [8, 4]
        vit_num_heads = [1, 2]
        vit_expansion_factors = [8, 8]
        vit_decoder_dim = 512

        if actor_cfg is not None:
            actor_hidden_dims = actor_cfg["hidden_dims"]
            state_hidden_dims = actor_cfg.get("state_hidden_dims", state_hidden_dims)
            image_latent_dim = actor_cfg.get("image_latent_dim", image_latent_dim)
            vit_resize_hw = actor_cfg.get("vit_resize_hw", vit_resize_hw)
            vit_patch_sizes = actor_cfg.get("vit_patch_sizes", vit_patch_sizes)
            vit_stage_strides = actor_cfg.get("vit_strides", vit_stage_strides)
            vit_paddings = actor_cfg.get("vit_paddings", vit_paddings)
            vit_embed_dims = actor_cfg.get("vit_embed_dims", vit_embed_dims)
            vit_num_layers = actor_cfg.get("vit_num_layers", vit_num_layers)
            vit_reduction_ratios = actor_cfg.get("vit_reduction_ratios", vit_reduction_ratios)
            vit_num_heads = actor_cfg.get("vit_num_heads", vit_num_heads)
            vit_expansion_factors = actor_cfg.get("vit_expansion_factors", vit_expansion_factors)
            vit_decoder_dim = actor_cfg.get("vit_decoder_dim", vit_decoder_dim)
            activation = actor_cfg.get("activation", activation)

        critic_activation = activation
        if critic_cfg is not None:
            critic_hidden_dims = critic_cfg["hidden_dims"]
            critic_activation = critic_cfg.get("activation", critic_activation)

        self.obs_groups = obs_groups
        self.actor_obs_normalization = actor_obs_normalization
        self.critic_obs_normalization = critic_obs_normalization
        self.noise_std_type = noise_std_type

        actor_state_dim, actor_image_shape = self._infer_group_shapes(obs_groups["policy"], obs)
        critic_obs_dim = self._infer_flat_obs_dim(obs_groups["critic"], obs)

        self._policy_state_groups = [name for name in obs_groups["policy"] if len(obs[name].shape[1:]) == 1]
        self._policy_sensor_groups = [name for name in obs_groups["policy"] if len(obs[name].shape[1:]) >= 2]
        self._actor_sensor_ndim = len(actor_image_shape) if actor_image_shape is not None else 0

        self.actor_state_normalizer = (
            EmpiricalNormalization(actor_state_dim) if actor_obs_normalization and actor_state_dim > 0 else nn.Identity()
        )
        self.critic_obs_normalizer = (
            EmpiricalNormalization(critic_obs_dim) if critic_obs_normalization and critic_obs_dim > 0 else nn.Identity()
        )

        self.actor_state_encoder, actor_state_out_dim = self._build_state_encoder(actor_state_dim, state_hidden_dims, activation)
        self.actor_image_encoder = VitFlyImageEncoder(
            image_shape=actor_image_shape,
            image_latent_dim=image_latent_dim,
            resize_hw=vit_resize_hw,
            patch_sizes=vit_patch_sizes,
            strides=vit_stage_strides,
            paddings=vit_paddings,
            embed_dims=vit_embed_dims,
            num_layers=vit_num_layers,
            reduction_ratios=vit_reduction_ratios,
            num_heads=vit_num_heads,
            expansion_factors=vit_expansion_factors,
            decoder_dim=vit_decoder_dim,
            activation=activation,
        )

        actor_encoder_dim = actor_state_out_dim + image_latent_dim
        critic_memory_dim = critic_obs_dim

        self.memory_a = Memory(actor_encoder_dim, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_dim)
        self.memory_c = Memory(critic_obs_dim, type=rnn_type, num_layers=rnn_num_layers, hidden_size=critic_memory_dim)

        # vitfly's ViT+LSTM head is a single linear layer after the LSTM output.
        self.actor = nn.Linear(rnn_hidden_dim, num_actions)
        self.critic = MLP(critic_memory_dim, 1, critic_hidden_dims, critic_activation)

        print(f"Actor VIT-LSTM: {self.memory_a}", flush=True)
        print(f"Actor Head: {self.actor}", flush=True)
        print(f"Critic LSTM: {self.memory_c}", flush=True)
        print(f"Critic MLP: {self.critic}", flush=True)

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

    def reset(self, dones=None, hidden_states=None):
        if hidden_states is None:
            self.memory_a.reset(dones)
            self.memory_c.reset(dones)
        else:
            hid_a, hid_c = hidden_states
            self.memory_a.reset(dones, hid_a)
            self.memory_c.reset(dones, hid_c)

    def detach_hidden_states(self, dones=None):
        self.memory_a.detach_hidden_states(dones)
        self.memory_c.detach_hidden_states(dones)

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
        out_mem = self.memory_c(critic_obs, masks, hidden_states).squeeze(0)
        return self.critic(out_mem)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def get_hidden_states(self):
        return self.memory_a.hidden_states, self.memory_c.hidden_states

    def update_normalization(self, obs):
        if self.actor_obs_normalization:
            actor_state, _ = self._split_groups(obs, self.obs_groups["policy"])
            if actor_state is not None:
                self.actor_state_normalizer.update(actor_state.reshape(-1, actor_state.shape[-1]))
        if self.critic_obs_normalization:
            critic_obs = self.get_critic_obs(obs)
            self.critic_obs_normalizer.update(critic_obs.reshape(-1, critic_obs.shape[-1]))

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict=strict)
        return True

    def get_actor_features(self, obs):
        state, image = self._split_groups(obs, self.obs_groups["policy"])
        features = []
        leading_shape = None
        if state is not None:
            leading_shape = state.shape[:-1]
            flat_state = state.reshape(-1, state.shape[-1])
            flat_state = self.actor_state_normalizer(flat_state)
            features.append(self.actor_state_encoder(flat_state).reshape(*leading_shape, -1))
        if image is not None:
            if leading_shape is None:
                leading_shape = image.shape[:-3]
            flat_image = image.reshape(-1, *image.shape[-3:])
            features.append(self.actor_image_encoder(flat_image).reshape(*leading_shape, -1))
        return torch.cat(features, dim=-1)

    def get_critic_obs(self, obs):
        obs_list = []
        for obs_group in self.obs_groups["critic"]:
            tensor = obs[obs_group]
            if tensor.ndim in (2, 3):
                obs_list.append(tensor)
            elif tensor.ndim in (4, 5):
                obs_list.append(tensor.reshape(*tensor.shape[:-3], -1))
            else:
                obs_list.append(tensor.reshape(*tensor.shape[:-1], -1))
        critic_obs = torch.cat(obs_list, dim=-1)
        flat = critic_obs.reshape(-1, critic_obs.shape[-1])
        flat = self.critic_obs_normalizer(flat)
        return flat.reshape(*critic_obs.shape[:-1], -1)

    def _split_groups(self, obs, group_names):
        state_tensors = []
        image_tensors = []
        for group_name in group_names:
            tensor = obs[group_name]
            if group_name in self._policy_state_groups:
                state_tensors.append(tensor)
            elif group_name in self._policy_sensor_groups:
                image_tensors.append(tensor)
            else:
                state_tensors.append(tensor.reshape(*tensor.shape[:-1], -1))
        state = torch.cat(state_tensors, dim=-1) if state_tensors else None
        if image_tensors:
            sensor_cat_dim = -self._actor_sensor_ndim
            image = torch.cat(image_tensors, dim=sensor_cat_dim)
        else:
            image = None
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

    def _infer_flat_obs_dim(self, group_names, obs):
        dim = 0
        for group_name in group_names:
            shape = obs[group_name].shape[1:]
            if len(shape) == 1:
                dim += shape[0]
            else:
                dim += int(torch.tensor(shape).prod().item())
        return dim

    def _build_state_encoder(self, input_dim, hidden_dims, activation):
        if input_dim == 0:
            return nn.Identity(), 0
        if not hidden_dims:
            return nn.Identity(), input_dim
        if len(hidden_dims) == 1:
            return nn.Sequential(nn.Linear(input_dim, hidden_dims[0]), _activation(activation)), hidden_dims[0]
        return MLP(input_dim, hidden_dims[-1], hidden_dims[:-1], activation), hidden_dims[-1]
