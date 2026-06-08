from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.networks import MLP, EmpiricalNormalization


def _activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "elu":
        return nn.ELU()
    if name == "relu":
        return nn.ReLU()
    if name == "leaky_relu":
        return nn.LeakyReLU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name}")


class NavRlLidarStateEncoder(nn.Module):
    """NavRL-style lidar CNN plus state MLP feature extractor."""

    def __init__(
        self,
        obs_dim: int,
        state_dim: int,
        lidar_start_idx: int,
        lidar_dim: int,
        lidar_shape: tuple[int, int],
        lidar_latent_dim: int,
        feature_hidden_dims: list[int],
        activation: str,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.state_dim = int(state_dim)
        self.lidar_start_idx = int(lidar_start_idx)
        self.lidar_dim = int(lidar_dim)
        self.lidar_hbeams = int(lidar_shape[0])
        self.lidar_vbeams = int(lidar_shape[1])
        if self.lidar_hbeams * self.lidar_vbeams != self.lidar_dim:
            raise ValueError(
                f"lidar_shape={lidar_shape} does not match lidar_dim={lidar_dim}."
            )
        if self.lidar_start_idx + self.lidar_dim > self.obs_dim:
            raise ValueError(
                f"LiDAR slice [{self.lidar_start_idx}:{self.lidar_start_idx + self.lidar_dim}] "
                f"exceeds obs_dim={self.obs_dim}."
            )

        act1 = _activation(activation)
        act2 = _activation(activation)
        act3 = _activation(activation)
        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.lidar_hbeams, self.lidar_vbeams)
            conv = nn.Sequential(
                nn.Conv2d(1, 4, kernel_size=(5, 3), padding=(2, 1)),
                act1,
                nn.Conv2d(4, 16, kernel_size=(5, 3), stride=(2, 1), padding=(2, 1)),
                act2,
                nn.Conv2d(16, 16, kernel_size=(5, 3), stride=(2, 2), padding=(2, 1)),
                act3,
                nn.Flatten(),
            )
            conv_out_dim = int(conv(dummy).shape[-1])
        self.lidar_cnn = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=(5, 3), padding=(2, 1)),
            _activation(activation),
            nn.Conv2d(4, 16, kernel_size=(5, 3), stride=(2, 1), padding=(2, 1)),
            _activation(activation),
            nn.Conv2d(16, 16, kernel_size=(5, 3), stride=(2, 2), padding=(2, 1)),
            _activation(activation),
            nn.Flatten(),
            nn.Linear(conv_out_dim, int(lidar_latent_dim)),
            nn.LayerNorm(int(lidar_latent_dim)),
        )
        self.fusion = self._make_fusion_mlp(int(lidar_latent_dim) + self.state_dim, feature_hidden_dims)
        self.output_dim = int(feature_hidden_dims[-1])

    def _make_fusion_mlp(self, input_dim: int, hidden_dims: list[int]) -> nn.Sequential:
        layers: list[nn.Module] = []
        last_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, int(hidden_dim)))
            layers.append(nn.LeakyReLU())
            layers.append(nn.LayerNorm(int(hidden_dim)))
            last_dim = int(hidden_dim)
        return nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        state = obs[:, : self.state_dim]
        lidar = obs[:, self.lidar_start_idx : self.lidar_start_idx + self.lidar_dim]
        lidar = lidar.view(-1, 1, self.lidar_hbeams, self.lidar_vbeams)
        lidar_feature = self.lidar_cnn(lidar)
        return self.fusion(torch.cat([lidar_feature, state], dim=-1))


class ActorCriticSAC(nn.Module):
    """Tanh-squashed Gaussian actor with twin Q critics for SAC."""

    is_recurrent = False

    def __init__(
        self,
        obs,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_hidden_dims: list[int],
        critic_hidden_dims: list[int],
        feature_hidden_dims: list[int] | None = None,
        activation: str = "elu",
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        use_lidar_cnn: bool = False,
        state_dim: int | None = None,
        lidar_start_idx: int | None = None,
        lidar_dim: int | None = None,
        lidar_shape: list[int] | tuple[int, int] | None = None,
        lidar_latent_dim: int = 128,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
        init_noise_std: float | None = None,
        **kwargs,
    ):
        super().__init__()
        del init_noise_std, kwargs
        self.obs_groups = obs_groups
        self.num_actions = int(num_actions)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.actor_obs_normalization = bool(actor_obs_normalization)
        self.critic_obs_normalization = bool(critic_obs_normalization)
        self.use_lidar_cnn = bool(use_lidar_cnn)

        actor_obs_dim = self._obs_dim(obs, obs_groups["policy"])
        critic_obs_dim = self._obs_dim(obs, obs_groups.get("critic", obs_groups["policy"]))
        self.actor_obs_group = obs_groups["policy"]
        self.critic_obs_group = obs_groups.get("critic", obs_groups["policy"])

        self.actor_obs_normalizer = (
            EmpiricalNormalization(shape=[actor_obs_dim], until=1.0e8) if actor_obs_normalization else nn.Identity()
        )
        self.critic_obs_normalizer = (
            EmpiricalNormalization(shape=[critic_obs_dim], until=1.0e8) if critic_obs_normalization else nn.Identity()
        )

        if self.use_lidar_cnn:
            feature_hidden_dims = [256, 256] if feature_hidden_dims is None else feature_hidden_dims
            if state_dim is None or lidar_start_idx is None or lidar_dim is None or lidar_shape is None:
                raise ValueError(
                    "state_dim, lidar_start_idx, lidar_dim, and lidar_shape are required when use_lidar_cnn=True."
                )
            lidar_shape_tuple = (int(lidar_shape[0]), int(lidar_shape[1]))
            self.actor_encoder = NavRlLidarStateEncoder(
                actor_obs_dim,
                state_dim,
                lidar_start_idx,
                lidar_dim,
                lidar_shape_tuple,
                lidar_latent_dim,
                feature_hidden_dims,
                activation,
            )
            self.q1_encoder = NavRlLidarStateEncoder(
                critic_obs_dim,
                state_dim,
                lidar_start_idx,
                lidar_dim,
                lidar_shape_tuple,
                lidar_latent_dim,
                feature_hidden_dims,
                activation,
            )
            self.q2_encoder = NavRlLidarStateEncoder(
                critic_obs_dim,
                state_dim,
                lidar_start_idx,
                lidar_dim,
                lidar_shape_tuple,
                lidar_latent_dim,
                feature_hidden_dims,
                activation,
            )
            self.target_q1_encoder = NavRlLidarStateEncoder(
                critic_obs_dim,
                state_dim,
                lidar_start_idx,
                lidar_dim,
                lidar_shape_tuple,
                lidar_latent_dim,
                feature_hidden_dims,
                activation,
            )
            self.target_q2_encoder = NavRlLidarStateEncoder(
                critic_obs_dim,
                state_dim,
                lidar_start_idx,
                lidar_dim,
                lidar_shape_tuple,
                lidar_latent_dim,
                feature_hidden_dims,
                activation,
            )
            feature_dim = self.actor_encoder.output_dim
            self.actor = MLP(feature_dim, 2 * self.num_actions, actor_hidden_dims, activation)
            critic_input_dim = feature_dim + self.num_actions
        else:
            self.actor_encoder = nn.Identity()
            self.q1_encoder = nn.Identity()
            self.q2_encoder = nn.Identity()
            self.target_q1_encoder = nn.Identity()
            self.target_q2_encoder = nn.Identity()
            self.actor = MLP(actor_obs_dim, 2 * self.num_actions, actor_hidden_dims, activation)
            critic_input_dim = critic_obs_dim + self.num_actions

        self.q1 = MLP(critic_input_dim, 1, critic_hidden_dims, activation)
        self.q2 = MLP(critic_input_dim, 1, critic_hidden_dims, activation)
        self.target_q1 = MLP(critic_input_dim, 1, critic_hidden_dims, activation)
        self.target_q2 = MLP(critic_input_dim, 1, critic_hidden_dims, activation)
        if self.use_lidar_cnn:
            self.target_q1_encoder.load_state_dict(self.q1_encoder.state_dict())
            self.target_q2_encoder.load_state_dict(self.q2_encoder.state_dict())
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())
        self.target_q1_encoder.requires_grad_(False)
        self.target_q2_encoder.requires_grad_(False)
        self.target_q1.requires_grad_(False)
        self.target_q2.requires_grad_(False)

    @staticmethod
    def _obs_dim(obs, group: list[str]) -> int:
        return int(sum(obs[key].reshape(obs.batch_size[0], -1).shape[-1] for key in group))

    @staticmethod
    def _cat_obs(obs, group: list[str]) -> torch.Tensor:
        return torch.cat([obs[key].reshape(obs.batch_size[0], -1) for key in group], dim=-1)

    def actor_obs(self, obs) -> torch.Tensor:
        return self.actor_obs_normalizer(self._cat_obs(obs, self.actor_obs_group))

    def critic_obs(self, obs) -> torch.Tensor:
        return self.critic_obs_normalizer(self._cat_obs(obs, self.critic_obs_group))

    def _distribution(self, obs) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.actor(self.actor_encoder(self.actor_obs(obs)))
        mean, log_std = torch.chunk(out, 2, dim=-1)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, obs) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self._distribution(obs)
        std = log_std.exp()
        normal = Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)
        log_prob = normal.log_prob(z) - torch.log(1.0 - action.pow(2) + 1.0e-6)
        return action, log_prob.sum(dim=-1, keepdim=True)

    def act(self, obs) -> torch.Tensor:
        action, _ = self.sample(obs)
        return action

    def act_inference(self, obs) -> torch.Tensor:
        mean, _ = self._distribution(obs)
        return torch.tanh(mean)

    @property
    def action_std(self) -> torch.Tensor:
        return torch.ones(self.num_actions, device=next(self.parameters()).device)

    def q_values(self, obs, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        critic_obs = self.critic_obs(obs)
        q1_x = torch.cat([self.q1_encoder(critic_obs), actions], dim=-1)
        q2_x = torch.cat([self.q2_encoder(critic_obs), actions], dim=-1)
        return self.q1(q1_x), self.q2(q2_x)

    def target_q_values(self, obs, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        critic_obs = self.critic_obs(obs)
        q1_x = torch.cat([self.target_q1_encoder(critic_obs), actions], dim=-1)
        q2_x = torch.cat([self.target_q2_encoder(critic_obs), actions], dim=-1)
        return self.target_q1(q1_x), self.target_q2(q2_x)

    def update_normalization(self, obs) -> None:
        if self.actor_obs_normalization:
            self.actor_obs_normalizer.update(self._cat_obs(obs, self.actor_obs_group))
        if self.critic_obs_normalization:
            self.critic_obs_normalizer.update(self._cat_obs(obs, self.critic_obs_group))

    @torch.no_grad()
    def update_targets(self, tau: float) -> None:
        pairs = [
            (self.target_q1_encoder, self.q1_encoder),
            (self.target_q2_encoder, self.q2_encoder),
            (self.target_q1, self.q1),
            (self.target_q2, self.q2),
        ]
        for target, source in pairs:
            for target_param, source_param in zip(target.parameters(), source.parameters()):
                target_param.mul_(1.0 - tau).add_(source_param, alpha=tau)
