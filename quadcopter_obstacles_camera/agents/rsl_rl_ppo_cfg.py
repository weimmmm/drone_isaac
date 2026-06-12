# Copyright (c) 2026 Alex Jauregui & Erik Eguskiza.
# PPO Configuration for Quadcopter Obstacles v5 - Directional Observations

from isaaclab.utils import configclass
from local_rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class QuadcopterObstaclesPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO config for FPV depth-image observations."""
    
    num_steps_per_env = 48
    max_iterations = 8000
    save_interval = 100
    experiment_name = "quadcopter_obstacles_camera"
    run_name = "5090_quadcopter_obstacles_camera_depth_cnn"
    logger = "wandb"
    wandb_project = "End_to_End_navigation"
    empirical_normalization = True
    obs_groups = {"policy": ["policy_state", "policy_image"], "critic": ["policy_state", "policy_image"]}
    
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCriticCnnRecurrent",
        init_noise_std=0.5,
        noise_std_type="log",
        state_hidden_dims=[128, 128],
        cnn_channels=[16, 32, 64],
        cnn_kernel_sizes=[5, 3, 3],
        cnn_strides=[2, 2, 2],
        image_latent_dim=128,
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
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
