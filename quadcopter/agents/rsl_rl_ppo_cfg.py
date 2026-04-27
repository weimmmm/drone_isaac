# Copyright (c) 2026 Alex Jauregui & Erik Eguskiza.
# All rights reserved.

from isaaclab.utils import configclass
from local_rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class QuadcopterPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """
    Configuration for the PPO (Proximal Policy Optimization) runner tailored for a base quadcopter.
    
    This configuration is designed to be compatible with transfer learning workflows. 
    The network architecture matches downstream tasks (like localization) to facilitate weight loading.
    """
    
    # Number of steps per environment per update. 
    # Slightly longer rollout helps velocity-tracking tasks learn smoother command transitions.
    num_steps_per_env = 32
    
    # Maximum number of iterations for training.
    # Increased for more stable convergence under velocity-tracking commands.
    max_iterations = 1000
    
    # Interval for saving model checkpoints.
    save_interval = 50
    
    # Experiment identifier.
    experiment_name = "quadcopter_base"
    
    # CRITICAL: Enables empirical normalization of observations.
    # In physics simulations, raw observation values (velocities, positions) vary wildly in scale.
    # Normalizing them (mean 0, std 1) is essential for the PPO optimizer to converge stably.
    empirical_normalization = True
    
    policy = RslRlPpoActorCriticCfg(
        # Lower initial exploration noise to reduce policy collapse in late training.
        init_noise_std=0.3,
        
        # Architecture for the Actor network (Policy).
        # [256, 256, 128] matches the architecture used in the localization task
        # to allow seamless transfer learning / fine-tuning later.
        actor_hidden_dims=[256, 256, 128],
        
        # Architecture for the Critic network (Value function).
        critic_hidden_dims=[256, 256, 128],
        
        # Activation function. ELU is often preferred in RL over ReLU to avoid "dead neurons" 
        # and handle negative values better in continuous control.
        activation="elu",
    )
    
    algorithm = RslRlPpoAlgorithmCfg(
        # Coefficient for the value function loss in the total loss objective.
        value_loss_coef=1.0,
        
        # Use clipped value loss to prevent the value function from changing too drastically 
        # in a single update, improving stability.
        use_clipped_value_loss=True,
        
        # PPO clipping parameter (epsilon). Limits the policy update step size.
        clip_param=0.2,
        
        # Lower entropy to avoid over-exploration after policy has started converging.
        entropy_coef=0.001,
        
        # Number of epochs to optimize the surrogate loss using the collected mini-batches.
        num_learning_epochs=5,
        
        # Number of mini-batches per epoch.
        num_mini_batches=4,
        
        # Base learning rate. 3.0e-4 is a standard starting point for Adam in continuous PPO.
        learning_rate=3.0e-4,
        
        # Learning rate schedule. "adaptive" adjusts the LR based on the KL divergence 
        # to ensure updates aren't too large (unsafe) or too small (slow).
        schedule="adaptive",
        
        # Discount factor (Gamma). 0.99 prioritizes long-term rewards.
        gamma=0.99,
        
        # GAE (Generalized Advantage Estimation) parameter (Lambda).
        # 0.95 balances bias and variance in advantage estimation.
        lam=0.95,
        
        # Target KL divergence for the adaptive learning rate scheduler.
        # If KL exceeds this, LR decreases; if it's lower, LR increases.
        desired_kl=0.01,
        
        # Gradient clipping norm to prevent exploding gradients during backpropagation.
        max_grad_norm=1.0,
    )
