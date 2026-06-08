from isaaclab.utils import configclass

from local_rsl_rl import (
    RslRlOffPolicyRunnerCfg,
    RslRlSacActorCriticCfg,
    RslRlSacAlgorithmCfg,
)

@configclass
class QuadcopterObstaclesSACRunnerCfg(RslRlOffPolicyRunnerCfg):
    """用于紧凑型激光雷达避障Teacher策略的标准SAC配置。"""

    # ==========================
    # 1. Runner 运行器配置 (控制数据收集与训练节奏)
    # ==========================
    # 每轮 rollout 走 512 个环境步，缩短采样-更新闭环，避免单轮过长。
    num_steps_per_env = 512
    
    # 训练的最大迭代次数。总环境步数 = max_iterations * num_envs * num_steps_per_env。
    max_iterations = 5000 
    
    # 每隔多少次迭代保存一次模型权重 (checkpoint)。
    save_interval = 50 
    
    # 实验名称，通常用于 TensorBoard/WandB 记录和模型保存的文件夹命名。
    experiment_name = "quadcopter_sac" 
    
    # 是否开启经验均值/方差的滑动平均来归一化观测值，这对于无人机动力学状态收敛很有帮助。
    empirical_normalization = True 
    
    # 动作截断范围，将网络输出的动作限制在 [-1.0, 1.0] 内。
    clip_actions = 1.0 
    
    # 经验回放池 (Replay Buffer) 的容量。1000万步的容量极大，说明你非常看重过往经验的复用。
    replay_buffer_size = 15_000_000 
    
    # 经验回放池存放的设备。由于 Buffer 极大，为了防止 GPU 显存溢出 (OOM)，放在 CPU 内存中。
    # 训练采样时再将 batch 数据搬运到 GPU。
    replay_buffer_device = "cpu" 

    replay_buffer_sample_interval = 16
    
    # 在训练开始前，先跑多少个 rollout iteration 的 warmup 动作。
    # 实际 warmup 环境步数 = warmup_iterations * num_envs * num_steps_per_env。
    warmup_iterations = 20

    # 在训练开始前，执行 warmup 动作的步数；若设置 warmup_iterations，则由 runner 自动覆盖。
    random_steps = 1_000_000 
    
    # 经过多少环境步后才开始进行神经网络的梯度更新；若设置 warmup_iterations，则由 runner 自动覆盖。
    learning_starts = 1_000_000 
    
    # 在完成一次 rollout 数据收集后，对 SAC 网络进行多少次梯度下降更新。
    gradient_steps_per_iteration = 100 
    random_forward_action_mean = 0.20
    random_forward_action_std = 0.60
    random_lateral_action_std = 0.60
    random_vertical_action_std = 0.10
    
    # 观测字典的分组配置。指定 Actor (policy) 和 Critic 使用的观测数据组。
    # 某些非对称 Actor-Critic 架构中，Critic 会使用额外的 privileged_obs。
    obs_groups = {
        "policy": ["policy"],
        "critic": ["policy"],
    }

    # ==========================
    # 2. Policy 策略网络配置 (Actor-Critic 架构)
    # ==========================
    policy = RslRlSacActorCriticCfg(
        # Actor 网络 (策略网络) 的独有隐藏层维度。
        actor_hidden_dims=[256], 
        
        # Critic 网络 (Q网络) 的独有隐藏层维度。
        critic_hidden_dims=[256, 256], 
        
        # 特征提取网络 (Shared/Feature Extractor) 的隐藏层维度，先经过这层再分发给 Actor 和 Critic。
        feature_hidden_dims=[256, 256], 
        
        # 激活函数，ELU 在连续控制任务中由于平滑特性，通常表现优于 ReLU。
        activation="elu", 
        
        # 环境观测已经按 NavRL 做了物理量归一化，避免对 LiDAR 占据值再做二次 running mean/std。
        actor_obs_normalization=False, 
        
        # Critic 同样直接使用 NavRL 语义的归一化观测。
        critic_obs_normalization=False, 
        
        # 启用 NavRL 风格 2D CNN 作为 3D LiDAR 的特征提取器。
        use_lidar_cnn=True, 
        
        # NavRL 风格 MLP 状态：8维目标方向坐标系状态 + 5个动态障碍物 * 10维。
        state_dim=58, 
        
        # 拼接后的总观测向量中，Lidar 数据的起始索引。
        lidar_start_idx=58, 
        
        # NavRL 3D LiDAR 展平维度：36 个水平束 x 4 个垂直束。
        lidar_dim=144, 
        
        # Lidar 数据的原始形状 (hbeams, vbeams)，适配 NavRL 的 2D CNN。
        lidar_shape=[36, 4], 
        
        # Lidar 经过 CNN 提取特征后，输出的隐向量维度大小。
        lidar_latent_dim=128, 
        
        # 高斯策略输出的对数标准差下界 (防止动作分布变得过于尖锐导致数值不稳定)。
        log_std_min=-20.0, 
        
        # 高斯策略输出的对数标准差上界 (防止探索空间变得过大导致发散)。
        log_std_max=2.0, 
    )

    # ==========================
    # 3. Algorithm 算法配置 (SAC 超参数)
    # ==========================
    algorithm = RslRlSacAlgorithmCfg(
        # Actor 网络的学习率。
        learning_rate=3.0e-4, 
        
        # Critic 网络的学习率 (通常与 Actor 保持一致或略大)。
        critic_learning_rate=3.0e-4, 
        
        # 温度系数 Alpha ($\alpha$) 的学习率，用于自动调节探索力度。
        alpha_learning_rate=3.0e-4, 
        
        # 折扣因子 Gamma ($\gamma$)，0.99 表示目光相对长远，适合无人机导航这种有延迟奖励的任务。
        gamma=0.99, 
        
        # 目标网络软更新系数 Tau ($\tau$)。每步用 0.5% 的当前网络权重去平滑更新目标网络。
        tau=0.005, 
        
        # 每次梯度更新时从 Replay Buffer 中采样的 Batch Size。
        batch_size=256, 
        
        # 目标熵 (Target Entropy)。设为 None 时，算法内部通常会自动将其设为 `-action_dim`。
        target_entropy=None, 
        
        # 温度系数 Alpha 的初始值。控制初始阶段对最大化熵 (即动作随机性/探索) 的权重。
        init_alpha=0.2, 
        
        # Alpha 允许的最小值。即使在后期也不让探索变为 0，防止策略坍缩为确定性。
        min_alpha=0.02, 
        
        # 是否启用 Alpha 自动调节 (基于对偶梯度下降优化温度系数)，SAC 的标配，极大减少调参负担。
        autotune_alpha=True, 

        # Actor 和 Critic 同频更新；当前任务里 critic 过度更新会把 Q 值推高并带偏策略。
        actor_update_interval=1,
        
        # 梯度裁剪阈值，防止网络更新步子过大引发梯度爆炸。
        max_grad_norm=1.0, 
    )
