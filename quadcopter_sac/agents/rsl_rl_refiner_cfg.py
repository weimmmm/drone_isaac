from isaaclab.utils import configclass

from local_rsl_rl import (
    RslRlActorCfg,
    RslRlCriticCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class QuadcopterObstaclesRefinerPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO config for standalone depth student with annealed teacher distillation."""

    num_steps_per_env = 48
    max_iterations = 6000
    save_interval = 50
    experiment_name = "quadcopter_obstacles_refiner"
    empirical_normalization = True
    distill_loss_coef_init = 1.0
    distill_loss_coef_final = 0.0
    distill_loss_anneal_steps = 150_000_000
    distill_learning_rate = 5.0e-5
    distill_num_learning_epochs = 1
    distill_num_mini_batches = 4
    distill_loss_type = "mse"
    obs_groups = {
        "policy": ["policy_state", "policy_image"],
        "critic": ["critic_base_state", "critic_privileged"],
    }

    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCriticCnnAsymmetric",
        init_noise_std=0.2,
        noise_std_type="log",
        actor_cfg=RslRlActorCfg(
            hidden_dims=[256, 256, 128],
            state_hidden_dims=[128, 128],
            cnn_channels=[16, 32, 64],
            cnn_kernel_sizes=[5, 3, 3],
            cnn_strides=[2, 2, 2],
            image_latent_dim=128,
            activation="elu",
        ),
        critic_cfg=RslRlCriticCfg(
            hidden_dims=[256, 256, 128],
            activation="elu",
        ),
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0005,
        num_learning_epochs=4,
        num_mini_batches=4,
        learning_rate=1.5e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.008,
        max_grad_norm=1.0,
    )
