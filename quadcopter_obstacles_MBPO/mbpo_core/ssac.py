# 深拷贝工具，用于复制目标网络
import copy
# 数学计算库
import math
# 随机工具
import random

# PyTorch核心库
import torch
# PyTorch神经网络模块
from torch import nn

# 导入配置基类、可选参数
from .config import BaseConfig, Configurable, Optional
# 导入默认配置：策略学习率、优化器
from .defaults import ACTOR_LR, OPTIMIZER
# 导入日志工具
from .log import default_log as log
# 导入策略基类、高斯策略（带输出压缩）
from .policy import BasePolicy, SquashedGaussianPolicy
# 导入PyTorch工具：设备、模型、MLP、EMA更新、冻结网络
from .torch_util import device, Module, mlp, update_ema, freeze_module
# 导入通用工具：均值计算
from .util import pythonic_mean


class LidarStateEncoder(Module):
    class Config(BaseConfig):
        lidar_start_idx = 11
        lidar_dim = 35
        state_hidden_dim = 128
        lidar_feature_dim = 64
        fused_dim = 256
        lidar_channels = (8, 16)

    def __init__(self, config, state_dim):
        Configurable.__init__(self, config)
        Module.__init__(self)
        self.state_dim = state_dim
        self.lidar_end_idx = self.lidar_start_idx + self.lidar_dim
        state_input_dim = state_dim - self.lidar_dim
        c1, c2 = self.lidar_channels

        self.state_net = mlp(
            [state_input_dim, self.state_hidden_dim, self.state_hidden_dim],
            output_activation="relu",
        )
        self.lidar_net = nn.Sequential(
            nn.Conv1d(1, c1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(c1, c2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(c2 * self.lidar_dim, self.lidar_feature_dim),
            nn.LayerNorm(self.lidar_feature_dim),
            nn.ReLU(),
        )
        self.fusion = mlp(
            [self.state_hidden_dim + self.lidar_feature_dim, self.fused_dim, self.fused_dim],
            output_activation="relu",
        )
        self.to(device)

    def forward(self, states):
        lidar = states[:, self.lidar_start_idx:self.lidar_end_idx].unsqueeze(1)
        state_wo_lidar = torch.cat(
            [states[:, :self.lidar_start_idx], states[:, self.lidar_end_idx:]], dim=1
        )
        state_feat = self.state_net(state_wo_lidar)
        lidar_feat = self.lidar_net(lidar)
        return self.fusion(torch.cat([state_feat, lidar_feat], dim=1))


class StateLidarPolicyNet(Module):
    def __init__(self, encoder_cfg, state_dim, action_dim, hidden_dim, hidden_layers):
        Module.__init__(self)
        self.encoder = LidarStateEncoder(encoder_cfg, state_dim)
        head_dims = [self.encoder.fused_dim, *([hidden_dim] * max(hidden_layers - 1, 0)), action_dim * 2]
        self.head = mlp(head_dims)

    def forward(self, states):
        return self.head(self.encoder(states))


class StateLidarQNet(Module):
    def __init__(self, encoder_cfg, state_dim, action_dim, hidden_dim, hidden_layers):
        Module.__init__(self)
        self.encoder = LidarStateEncoder(encoder_cfg, state_dim)
        self.q = mlp(
            [self.encoder.fused_dim + action_dim, *([hidden_dim] * hidden_layers), 1],
            squeeze_output=True,
        )

    def forward(self, state, action):
        features = self.encoder(state)
        return self.q(torch.cat([features, action], dim=1))


# ====================== 评论家集成网络 ======================
# 集成多个评论家网络，解决SAC中的过估计问题
class CriticEnsemble(Configurable, Module):
    # 输入：
    #   无
    # 输出：
    #   一个评论家集成网络对象
    # 参数：
    #   无
    # 作用：
    #   维护多个 Q 网络，供 SAC/SSAC 计算最小 Q、平均 Q 或随机 Q。
    # 评论家配置类
    class Config(BaseConfig):
        # 输入：
        #   无
        # 输出：
        #   评论家配置对象
        # 参数：
        #   n_critics, hidden_layers, hidden_dim, learning_rate
        # 作用：
        #   定义评论家网络数量、结构和学习率。
        n_critics = 2           # 评论家数量（默认2个，Double Q-learning）
        hidden_layers = 2       # 隐藏层层数
        hidden_dim = 256        # 隐藏层维度
        learning_rate = 3e-4    # 学习率
        use_lidar_cnn = True
        encoder_cfg = LidarStateEncoder.Config()

    # 初始化：配置、状态维度、动作维度
    def __init__(self, config, state_dim, action_dim):
        # 输入：
        #   config: 评论家配置
        #   state_dim: 状态维度
        #   action_dim: 动作维度
        # 输出：
        #   初始化完成的评论家集成
        # 参数：
        #   dims: [state_dim + action_dim, ..., 1]
        # 作用：
        #   构建多个 Q 网络以及对应优化器。
        Configurable.__init__(self, config)
        Module.__init__(self)
        if self.use_lidar_cnn:
            self.qs = torch.nn.ModuleList([
                StateLidarQNet(self.encoder_cfg, state_dim, action_dim, self.hidden_dim, self.hidden_layers)
                for _ in range(self.n_critics)
            ])
        else:
            dims = [state_dim + action_dim, *([self.hidden_dim] * self.hidden_layers), 1]
            self.qs = torch.nn.ModuleList([
                mlp(dims, squeeze_output=True) for _ in range(self.n_critics)
            ])
        # 初始化Adam优化器
        self.optimizer = torch.optim.Adam(self.qs.parameters(), lr=self.learning_rate)

    # 返回所有评论家的Q值
    def all(self, state, action):
        # 输入：
        #   state: [B, state_dim]
        #   action: [B, action_dim]
        # 输出：
        #   一个长度为 n_critics 的 Q 值列表，每个元素形状为 [B]
        # 参数：
        #   state/action: 状态动作批次
        # 作用：
        #   计算所有评论家对同一批状态动作的 Q 估计。
        if self.use_lidar_cnn:
            return [q(state, action) for q in self.qs]
        sa = torch.cat([state, action], 1)  # 拼接状态+动作
        return [q(sa) for q in self.qs]

    # 返回最小Q值（防止过估计，Double Q-learning核心）
    def min(self, state, action):
        # 输入：
        #   state: [B, state_dim]
        #   action: [B, action_dim]
        # 输出：
        #   [B]
        # 参数：
        #   state/action: 状态动作批次
        # 作用：
        #   返回所有评论家中最小的 Q 值，用于抑制过估计。
        return torch.min(*self.all(state, action))

    # 返回所有评论家Q值的均值
    def mean(self, state, action):
        # 输入：
        #   state: [B, state_dim]
        #   action: [B, action_dim]
        # 输出：
        #   [B]
        # 参数：
        #   state/action: 状态动作批次
        # 作用：
        #   返回所有评论家 Q 值的平均值，多用于诊断和日志。
        return pythonic_mean(self.all(state, action))

    # 随机选择一个评论家（用于策略训练，减少偏差）
    def random_choice(self, state, action):
        # 输入：
        #   state: [B, state_dim]
        #   action: [B, action_dim]
        # 输出：
        #   [B]
        # 参数：
        #   state/action: 状态动作批次
        # 作用：
        #   随机选一个评论家输出 Q 值，给 actor 更新使用。
        q = random.choice(self.qs)
        if self.use_lidar_cnn:
            return q(state, action)
        sa = torch.cat([state, action], 1)
        return q(sa)


# ====================== SSAC：安全SAC算法核心 ======================
# 继承策略基类 + PyTorch模型类
class SSAC(BasePolicy, Module):
    # 输入：
    #   无
    # 输出：
    #   一个安全版 SAC 求解器对象
    # 参数：
    #   无
    # 作用：
    #   管理 actor、critic、target critic、alpha，并把安全 violation 纳入价值学习。
    # SSAC算法配置类（所有超参数）
    class Config(BaseConfig):
        # 输入：
        #   无
        # 输出：
        #   SSAC 配置对象
        # 参数：
        #   discount, init_alpha, autotune_alpha, target_entropy,
        #   use_log_alpha_loss, deterministic_backup, critic_update_multiplier,
        #   actor_lr, critic_cfg, tau, batch_size, hidden_dim,
        #   hidden_layers, update_violation_cost
        # 作用：
        #   定义安全 SAC 训练用到的全部超参数。
        discount = 0.99                # 折扣因子γ
        init_alpha = 1.0               # 初始温度系数α
        autotune_alpha = True          # 是否自动调整α
        target_entropy = Optional(float)  # 目标熵（可选）
        use_log_alpha_loss = True      # 使用对数α计算损失
        deterministic_backup = False   # 是否使用确定性目标值
        critic_update_multiplier = 1   # 评论家更新倍数
        actor_lr = ACTOR_LR            # 策略学习率
        critic_cfg = CriticEnsemble.Config()  # 评论家配置
        tau = 0.005                    # 目标网络更新系数τ
        batch_size = 256               # 批次大小
        hidden_dim = 256              # 策略隐藏层维度
        hidden_layers = 2              # 策略隐藏层层数
        update_violation_cost = True   # 自动更新安全违规代价
        use_lidar_cnn = True
        encoder_cfg = LidarStateEncoder.Config()

    # 初始化：配置、状态维度、动作维度、视野长度、优化器
    def __init__(self, config, state_dim, action_dim, horizon,
                 optimizer_factory=OPTIMIZER):
        # 输入：
        #   config: SSAC 配置对象
        #   state_dim: 状态维度
        #   action_dim: 动作维度
        #   horizon: 模型 rollout 视野长度
        #   optimizer_factory: 优化器构造器
        # 输出：
        #   初始化完成的 SSAC 实例
        # 参数：
        #   violation_cost: 违规惩罚
        #   log_alpha: 温度系数的对数形式
        # 作用：
        #   创建策略网络、评论家网络、目标网络和 alpha 优化逻辑。
        Configurable.__init__(self, config)
        Module.__init__(self)
        self.horizon = horizon         # 策略视野
        self.violation_cost = 0.0      # 安全违规代价（惩罚值）

        if self.use_lidar_cnn:
            self.actor = SquashedGaussianPolicy(
                StateLidarPolicyNet(self.encoder_cfg, state_dim, action_dim, self.hidden_dim, self.hidden_layers)
            )
            self.critic_cfg.use_lidar_cnn = True
            self.critic_cfg.encoder_cfg = self.encoder_cfg
        else:
            self.actor = SquashedGaussianPolicy(mlp(
                [state_dim, *([self.hidden_dim] * self.hidden_layers), action_dim*2]
            ))
        # 评论家集成网络
        self.critic = CriticEnsemble(self.critic_cfg, state_dim, action_dim)
        # 目标评论家网络（延迟更新，稳定训练）
        self.critic_target = copy.deepcopy(self.critic)
        # 冻结目标网络参数（不参与梯度更新）
        freeze_module(self.critic_target)

        # 策略优化器
        self.actor_optimizer = optimizer_factory(self.actor.parameters(), lr=self.actor_lr)

        # 温度系数α（对数形式，保证输出为正）
        log_alpha = torch.tensor(math.log(self.init_alpha), device=device, requires_grad=True)
        self.log_alpha = log_alpha
        # 自动调整α：创建优化器
        if self.autotune_alpha:
            self.alpha_optimizer = optimizer_factory([self.log_alpha], lr=self.actor_lr)
        # 无目标熵则默认设置为 -动作维度
        if self.target_entropy is None or isinstance(self.target_entropy, Optional):
            self.target_entropy = -float(action_dim)

        # 损失函数：均方误差（MSE）
        self.criterion = nn.MSELoss()

        # 注册缓冲区：总更新次数（持久化，不参与梯度）
        self.register_buffer('total_updates', torch.zeros([]))

    # 动作选择接口：输入状态，输出动作
    def act(self, states, eval):
        # 输入：
        #   states: [B, state_dim]
        #   eval: 是否评估模式
        # 输出：
        #   [B, action_dim]
        # 参数：
        #   states: 状态批次
        #   eval: True 时倾向使用确定性动作
        # 作用：
        #   统一对外暴露动作选择接口。
        return self.actor.act(states, eval)

    # 属性：获取α真实值（对数指数化）
    @property
    def alpha(self):
        # 输入：
        #   无
        # 输出：
        #   标量 alpha
        # 参数：
        #   无
        # 作用：
        #   将 log_alpha 指数化，得到实际温度系数。
        return self.log_alpha.exp()

    # 属性：违规状态的价值（惩罚值的折扣和）
    @property
    def violation_value(self):
        # 输入：
        #   无
        # 输出：
        #   标量 violation value
        # 参数：
        #   无
        # 作用：
        #   将单步违规代价转成长期价值上的终止惩罚。
        return -self.violation_cost / (1. - self.discount)

    # 更新奖励范围 + 自动计算违规代价
    def update_r_bounds(self, r_min, r_max):
        # 输入：
        #   r_min: 观测到的最小奖励
        #   r_max: 观测到的最大奖励
        # 输出：
        #   无
        # 参数：
        #   r_min/r_max: 奖励范围
        # 作用：
        #   同步奖励上下界，并据此自动推导 violation 惩罚尺度。
        self.r_min, self.r_max = r_min, r_max
        # 自动更新违规惩罚：根据奖励范围和视野动态计算
        if self.update_violation_cost:
            self.violation_cost = (r_max - r_min) / self.discount**self.horizon - r_max
        log.message(f'r bounds: [{r_min, r_max}], C = {self.violation_cost}')

    # 旧版评论家损失（已被下方重载方法替代）
    def critic_loss(self, obs, action, next_obs, reward, done):
        # 输入：
        #   obs, action, next_obs, reward, done
        # 输出：
        #   标量 critic loss
        # 参数：
        #   同 SAC 标准目标
        # 作用：
        #   旧接口，当前已被下方带 violation 的版本覆盖。
        reward = reward.clamp(self.r_min, self.r_max)
        target = super().compute_target(next_obs, reward, done)
        if done.any():
            target[done] = self.terminal_value
        return self.critic_loss_given_target(obs, action, target)

    # 计算评论家目标Q值（贝尔曼方程）
    def compute_target(self, next_obs, reward, done, violation):
        # 输入：
        #   next_obs: [B, state_dim]
        #   reward: [B]
        #   done: [B]
        #   violation: [B]
        # 输出：
        #   目标 Q 值 [B]
        # 参数：
        #   violation: 是否发生安全违规
        # 作用：
        #   计算 Bellman target，并将违规样本强制设为惩罚价值。
        with torch.no_grad():  # 目标值计算，禁用梯度
            next_action, log_prob = self.actor.sample_and_log_prob(next_obs)
            next_value = self.critic_target.min(next_obs, next_action)  # 目标Q最小值
            # 随机策略：Q = 奖励 + 折扣*(next_Q - α*log_prob)
            if not self.deterministic_backup:
                next_value = next_value - self.alpha.detach() * log_prob
            # 贝尔曼目标
            q = reward + self.discount * (1. - done.float()) * next_value
            # 安全约束：违规状态直接赋值惩罚价值
            q[violation] = 0.2 * self.violation_value # 将惩罚缩成0.2倍
            return q

    # 给定目标值，计算评论家损失（MSE）
    def critic_loss_given_target(self, obs, action, target):
        # 输入：
        #   obs: [B, state_dim]
        #   action: [B, action_dim]
        #   target: [B]
        # 输出：
        #   标量 critic loss
        # 参数：
        #   target: 目标 Q 值
        # 作用：
        #   对所有评论家计算 MSE，并求平均作为总 critic 损失。
        qs = self.critic.all(obs, action)
        # 所有评论家损失的均值
        return pythonic_mean([self.criterion(q, target) for q in qs])

    # 评论家损失（完整版，包含安全违规）
    def critic_loss(self, obs, action, next_obs, reward, done, violation):
        # 输入：
        #   obs, action, next_obs, reward, done, violation
        # 输出：
        #   标量 critic loss
        # 参数：
        #   violation: 安全违规标记
        # 作用：
        #   先构造包含安全约束的目标值，再计算评论家损失。
        target = self.compute_target(next_obs, reward, done, violation)
        return self.critic_loss_given_target(obs, action, target)

    # 更新评论家网络 + 目标网络EMA更新
    def update_critic(self, *critic_loss_args):
        # 输入：
        #   critic_loss_args: critic_loss 所需的一整组 batch
        # 输出：
        #   detach 后的 critic loss
        # 参数：
        #   critic_loss_args: (obs, action, next_obs, reward, done, violation)
        # 作用：
        #   完成一次 critic 反向传播，并对 target critic 做 EMA 软更新。
        critic_loss = self.critic_loss(*critic_loss_args)
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()
        # 软更新目标网络：target = τ*online + (1-τ)*target
        update_ema(self.critic_target, self.critic, self.tau)
        return critic_loss.detach()

    # 策略损失 + α损失
    def actor_loss(self, obs, include_alpha=True):
        # 输入：
        #   obs: [B, state_dim]
        #   include_alpha: 是否同时返回 alpha loss
        # 输出：
        #   [actor_loss] 或 [actor_loss, alpha_loss]
        # 参数：
        #   include_alpha: 是否自动调节温度系数
        # 作用：
        #   计算策略优化目标，以及可选的温度系数优化目标。
        action, log_prob = self.actor.sample_and_log_prob(obs)
        # 随机选一个评论家计算Q值
        actor_Q = self.critic.random_choice(obs, action)
        alpha = self.alpha
        # 策略损失：最大化 (Q - α*log_prob) → 等价于最小化 (α*log_prob - Q)
        actor_loss = torch.mean(alpha.detach() * log_prob - actor_Q)
        # 自动调整α：最大化熵，接近目标熵
        if include_alpha:
            multiplier = self.log_alpha if self.use_log_alpha_loss else alpha
            alpha_loss = -multiplier * torch.mean(log_prob.detach() + self.target_entropy)
            return [actor_loss, alpha_loss]
        else:
            return [actor_loss]

    # 更新策略网络 + α
    def update_actor_and_alpha(self, obs):
        # 输入：
        #   obs: [B, state_dim]
        # 输出：
        #   无
        # 参数：
        #   obs: 用于 actor 更新的一批观测
        # 作用：
        #   分别对 actor 和 alpha 执行一次梯度更新。
        losses = self.actor_loss(obs, include_alpha=self.autotune_alpha)
        optimizers = [self.actor_optimizer, self.alpha_optimizer] if self.autotune_alpha else \
                     [self.actor_optimizer]
        # 分别反向传播更新
        assert len(losses) == len(optimizers)
        for loss, optimizer in zip(losses, optimizers):
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # 完整更新一步：评论家更新N次 + 策略更新1次
    def update(self, replay_buffer):
        # 输入：
        #   replay_buffer: 经验回放缓冲区
        # 输出：
        #   无
        # 参数：
        #   replay_buffer: 提供 batch 采样接口
        # 作用：
        #   执行一轮完整的 SSAC 更新流程：多次 critic 更新，再更新 actor 与 alpha。
        assert self.critic_update_multiplier >= 1
        # 评论家更新多次（提升收敛稳定性）
        for _ in range(self.critic_update_multiplier):
            samples = replay_buffer.sample(self.batch_size)
            self.update_critic(*samples)
        # 更新策略和α
        self.update_actor_and_alpha(samples[0])
        self.total_updates += 1
