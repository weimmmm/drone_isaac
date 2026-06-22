# Copyright (c) 2026 Alex Jauregui & Erik Eguskiza.
# PPO Configuration for Quadcopter Obstacles v5 - Directional Observations

from isaaclab.utils import configclass
from local_rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class QuadcopterObstaclesPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO config for obstacles with directional observations."""
    
    num_steps_per_env = 48
    max_iterations = 8000
    save_interval = 100
    experiment_name = "quadcopter_obstacles_student"
    run_name = "5090_quadcopter_obstacles_student_safe_multihead"
    logger = "wandb"
    wandb_project = "End_to_End_navigation"
    empirical_normalization = True
    
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCriticMultiHeadObs",
        init_noise_std=0.5,
        noise_std_type="log",
        static_obs_start_idx=15,
        static_obs_dim=32,
        dynamic_obs_start_idx=47,
        dynamic_obs_dim=50,
        static_encoder_hidden_dims=[256, 512],
        dynamic_encoder_hidden_dims=[256, 512],
        other_encoder_hidden_dims=[256, 512],
        static_latent_dim=512,
        dynamic_latent_dim=512,
        other_latent_dim=512,
        actor_hidden_dims=[512, 512, 256],
        critic_hidden_dims=[512, 512, 256],
        activation="elu",
    )
    
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=4,
        num_mini_batches=4,
        learning_rate=1.5e-4,
        schedule="adaptive",
        gamma=0.995,
        lam=0.95,
        desired_kl=0.008,
        max_grad_norm=1.0,
    )
