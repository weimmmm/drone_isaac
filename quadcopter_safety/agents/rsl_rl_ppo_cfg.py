from isaaclab.utils import configclass

from local_rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class QuadcopterSafetyPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 3000
    save_interval = 200
    experiment_name = "quadcopter_safety_policy_v1"
    empirical_normalization = True
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}

    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCriticRecurrent",
        init_noise_std=0.5,
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
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=1,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
