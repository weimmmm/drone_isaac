from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass


@configclass
class RslRlActorCfg:
    hidden_dims: list[int] = MISSING
    state_hidden_dims: list[int] | None = None
    cnn_channels: list[int] | None = None
    cnn_kernel_sizes: list[int] | None = None
    cnn_strides: list[int] | None = None
    image_encoder_type: str | None = None
    image_latent_dim: int | None = None
    vit_resize_hw: list[int] | None = None
    vit_patch_sizes: list[int] | None = None
    vit_strides: list[int] | None = None
    vit_paddings: list[int] | None = None
    vit_embed_dims: list[int] | None = None
    vit_num_layers: list[int] | None = None
    vit_reduction_ratios: list[int] | None = None
    vit_num_heads: list[int] | None = None
    vit_expansion_factors: list[int] | None = None
    vit_decoder_dim: int | None = None
    activation: str = MISSING


@configclass
class RslRlCriticCfg:
    hidden_dims: list[int] = MISSING
    activation: str = MISSING


@configclass
class RslRlPpoActorCriticCfg:
    class_name: str = "ActorCritic"
    init_noise_std: float = MISSING
    noise_std_type: Literal["scalar", "log"] = "scalar"
    actor_cfg: RslRlActorCfg | None = None
    critic_cfg: RslRlCriticCfg | None = None
    # Backward-compatible flat fields used by older tasks/configs.
    actor_hidden_dims: list[int] | None = None
    critic_hidden_dims: list[int] | None = None
    static_encoder_hidden_dims: list[int] | None = None
    dynamic_encoder_hidden_dims: list[int] | None = None
    other_encoder_hidden_dims: list[int] | None = None
    static_latent_dim: int | None = None
    dynamic_latent_dim: int | None = None
    other_latent_dim: int | None = None
    static_obs_start_idx: int | None = None
    static_obs_dim: int | None = None
    dynamic_obs_start_idx: int | None = None
    dynamic_obs_dim: int | None = None
    state_hidden_dims: list[int] | None = None
    cnn_channels: list[int] | None = None
    cnn_kernel_sizes: list[int] | None = None
    cnn_strides: list[int] | None = None
    image_encoder_type: str | None = None
    image_latent_dim: int | None = None
    vit_resize_hw: list[int] | None = None
    vit_patch_sizes: list[int] | None = None
    vit_strides: list[int] | None = None
    vit_paddings: list[int] | None = None
    vit_embed_dims: list[int] | None = None
    vit_num_layers: list[int] | None = None
    vit_reduction_ratios: list[int] | None = None
    vit_num_heads: list[int] | None = None
    vit_expansion_factors: list[int] | None = None
    vit_decoder_dim: int | None = None
    activation: str | None = None
    rnn_type: str | None = None
    rnn_hidden_dim: int | None = None
    rnn_num_layers: int | None = None
    actor_obs_normalization: bool | None = None
    critic_obs_normalization: bool | None = None


@configclass
class RslRlPpoAlgorithmCfg:
    class_name: str = "PPO"
    num_learning_epochs: int = MISSING
    num_mini_batches: int = MISSING
    learning_rate: float = MISSING
    schedule: str = MISSING
    gamma: float = MISSING
    lam: float = MISSING
    entropy_coef: float = MISSING
    desired_kl: float = MISSING
    max_grad_norm: float = MISSING
    value_loss_coef: float = MISSING
    use_clipped_value_loss: bool = MISSING
    clip_param: float = MISSING
    normalize_advantage_per_mini_batch: bool = False
    symmetry_cfg: dict | None = None
    rnd_cfg: dict | None = None


@configclass
class RslRlSacActorCriticCfg:
    class_name: str = "ActorCriticSAC"
    actor_hidden_dims: list[int] = MISSING
    critic_hidden_dims: list[int] = MISSING
    feature_hidden_dims: list[int] | None = None
    activation: str = "elu"
    actor_obs_normalization: bool | None = None
    critic_obs_normalization: bool | None = None
    use_lidar_cnn: bool = False
    state_dim: int | None = None
    lidar_start_idx: int | None = None
    lidar_dim: int | None = None
    lidar_shape: list[int] | None = None
    lidar_latent_dim: int = 128
    log_std_min: float = -20.0
    log_std_max: float = 2.0


@configclass
class RslRlSacAlgorithmCfg:
    class_name: str = "SAC"
    learning_rate: float = 1.0e-4
    critic_learning_rate: float | None = 5.0e-4
    alpha_learning_rate: float | None = 1.0e-4
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 1024
    target_entropy: float | None = None
    init_alpha: float = 0.2
    min_alpha: float = 0.02
    autotune_alpha: bool = True
    actor_update_interval: int = 2
    max_grad_norm: float | None = 1.0


@configclass
class RslRlOffPolicyRunnerCfg:
    seed: int = 42
    device: str = "cuda:0"
    num_steps_per_env: int = MISSING
    max_iterations: int = MISSING
    empirical_normalization: bool = False
    obs_groups: dict[str, list[str]] = {"policy": ["policy"], "critic": ["policy"]}
    policy: RslRlSacActorCriticCfg = MISSING
    algorithm: RslRlSacAlgorithmCfg = MISSING
    clip_actions: float | None = 1.0
    save_interval: int = MISSING
    experiment_name: str = MISSING
    run_name: str = ""
    logger: Literal["tensorboard", "neptune", "wandb"] = "tensorboard"
    resume: bool = False
    load_run: str = ".*"
    load_checkpoint: str = "model_.*.pt"
    replay_buffer_size: int = 1_000_000
    replay_buffer_device: str = "cpu"
    replay_buffer_sample_interval: int = 1
    warmup_iterations: int | None = None
    random_steps: int = 1_000_000
    learning_starts: int = 1_000_000
    gradient_steps_per_iteration: int = 100
    random_forward_action_mean: float = 0.20
    random_forward_action_std: float = 0.60
    random_lateral_action_std: float = 0.60
    random_vertical_action_std: float = 0.10
    warmup_action_mode: Literal["random", "target"] = "random"
    warmup_target_action_scale: float = 0.60
    warmup_lateral_noise_std: float = 0.15
    warmup_vertical_gain: float = 0.60
    warmup_vertical_noise_std: float = 0.03
    warmup_min_height: float = 0.80


@configclass
class RslRlOnPolicyRunnerCfg:
    seed: int = 42
    device: str = "cuda:0"
    num_steps_per_env: int = MISSING
    max_iterations: int = MISSING
    empirical_normalization: bool = MISSING
    obs_groups: dict[str, list[str]] = {"policy": ["policy"], "critic": ["policy"]}
    policy: RslRlPpoActorCriticCfg = MISSING
    algorithm: RslRlPpoAlgorithmCfg = MISSING
    clip_actions: float | None = None
    save_interval: int = MISSING
    experiment_name: str = MISSING
    run_name: str = ""
    logger: Literal["tensorboard", "neptune", "wandb"] = "tensorboard"
    neptune_project: str = "isaaclab"
    wandb_project: str = "isaaclab"
    resume: bool = False
    load_run: str = ".*"
    load_checkpoint: str = "model_.*.pt"
    distill_loss_coef_init: float = 0.0
    distill_loss_coef_final: float = 0.0
    distill_loss_anneal_steps: int = 1
    distill_learning_rate: float | None = None
    distill_num_learning_epochs: int = 1
    distill_num_mini_batches: int = 4
    distill_loss_type: Literal["mse", "huber"] = "mse"
