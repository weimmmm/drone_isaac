# Copyright (c) 2026 Alex Jauregui & Erik Eguskiza.
# PPO Configuration for Quadcopter Obstacles v5 - Directional Observations

from isaaclab.utils import configclass
from local_rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class QuadcopterObstaclesPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO config for obstacles with directional observations."""
    
    num_steps_per_env = 96
    max_iterations = 8000
    save_interval = 100
    experiment_name = "quadcopter_obstacles_v5_lstm"
    empirical_normalization = True
    
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}

    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCriticRecurrent",
        init_noise_std=0.3,
        noise_std_type="log",
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
        rnn_type="lstm",
        rnn_hidden_dim=128,
        rnn_num_layers=1,
        activation="elu",
    )
    
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0005,
        num_learning_epochs=5,
        num_mini_batches=16,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
