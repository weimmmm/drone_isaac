# Copyright (c) 2026 Alex Jauregui & Erik Eguskiza.
# PPO Configuration for Quadcopter Obstacles v5 - Directional Observations

from isaaclab.utils import configclass
from local_rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class QuadcopterObstaclesPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO config for obstacles with directional observations."""
    
    num_steps_per_env = 32
    max_iterations = 5000  # Más iteraciones para que converja bien
    save_interval = 300
    experiment_name = "quadcopter_camera_depth_cnn_lstm_v1"
    empirical_normalization = True
    obs_groups = {"policy": ["state", "depth"], "critic": ["state", "depth"]}
    
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCriticCnnRecurrent",
        init_noise_std=0.03,
        actor_hidden_dims=[128, 64],
        critic_hidden_dims=[128, 64],
        state_hidden_dims=[64],
        cnn_channels=[16, 32, 64],
        cnn_kernel_sizes=[5, 3, 3],
        cnn_strides=[2, 2, 2],
        image_latent_dim=128,
        rnn_type="lstm",
        rnn_hidden_dim=128,
        rnn_num_layers=1,
        activation="elu",
    )
    
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0002,
        num_learning_epochs=5,
        num_mini_batches=1,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
