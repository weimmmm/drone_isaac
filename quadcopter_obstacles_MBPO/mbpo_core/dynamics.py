# 导入抽象基类，用于定义接口
from abc import ABC, abstractmethod
# 随机数工具
import random

# PyTorch核心库
import torch
# PyTorch神经网络模块
import torch.nn as nn
# PyTorch函数式接口
import torch.nn.functional as F
# 进度条库
from tqdm import trange

# 导入默认配置
from . import defaults
# 导入配置基类
from .config import BaseConfig, Configurable
# 导入数据归一化工具
from .normalization import Normalizer
# 导入训练工具
from .train import epochal_training
# 导入PyTorch工具函数
from .torch_util import device, Module, mlp


# 基础模型抽象类（所有环境模型必须继承并实现sample方法）
class BaseModel(ABC):
    # 输入：
    #   无
    # 输出：
    #   一个动力学模型抽象基类
    # 参数：
    #   无
    # 作用：
    #   统一约束动力学模型接口，子类必须实现 sample(states, actions)。
    # 抽象方法：必须实现，输入状态和动作，输出下一状态和奖励的采样值
    @abstractmethod
    def sample(self, states, actions):
        """
        输入：
            states: 当前状态张量
            actions: 当前动作张量
        输出：
            (next_states, rewards)
        参数：
            states: 状态输入
            actions: 动作输入
        作用：
            给定当前状态 s 和动作 a，返回采样得到的下一状态 s' 与奖励 r。

        Returns a sample of (s', r) given (s, a)
        给定当前状态s和动作a，返回下一状态s'和奖励r的采样结果
        """
        pass


# 批处理线性层：用于高效实现集成模型的矩阵乘法（核心优化层）
class BatchedLinear(nn.Module):
    """For efficient MLP ensembles with batched matrix multiplies
    为高效的多层感知机集成模型提供批处理矩阵乘法
    """
    # 输入：
    #   ensemble_size: 集成模型数量
    #   in_features: 输入维度
    #   out_features: 输出维度
    #   bias: 是否使用偏置
    # 输出：
    #   一个支持 ensemble 并行计算的线性层
    # 参数：
    #   weight: [ensemble_size, out_features, in_features]
    #   bias: [ensemble_size, out_features]
    # 作用：
    #   用一个模块同时存储多个线性层参数，供 ensemble 并行前向使用。
    # 初始化：集成数量、输入维度、输出维度、是否使用偏置
    def __init__(self, ensemble_size, in_features, out_features, bias=True):
        super().__init__()
        self.ensemble_size = ensemble_size  # 集成模型数量
        self.in_features = in_features      # 输入特征维度
        self.out_features = out_features    # 输出特征维度
        # 权重参数：形状 [集成数, 输出维度, 输入维度]
        self.weight = nn.Parameter(torch.empty(ensemble_size, out_features, in_features))
        if bias:
            # 偏置参数：形状 [集成数, 输出维度]
            self.bias = nn.Parameter(torch.empty(ensemble_size, out_features))
        else:
            # 不使用偏置则注册为空参数
            self.register_parameter('bias', None)
        # 初始化权重和偏置
        self.reset_parameters()

    # 权重初始化：复用PyTorch标准Linear层的初始化逻辑
    def reset_parameters(self):
        # 输入：
        #   无
        # 输出：
        #   无
        # 参数：
        #   无
        # 作用：
        #   逐个初始化 ensemble 中每个线性层的参数。
        has_bias = self.bias is not None
        # 创建标准线性层用于初始化
        l = nn.Linear(self.in_features, self.out_features, bias=has_bias)
        for i in range(self.ensemble_size):
            l.reset_parameters()  # 重置标准层参数
            self.weight.data[i].copy_(l.weight.data)  # 复制到集成权重
            if has_bias:
                self.bias.data[i].copy_(l.bias.data)  # 复制到集成偏置

    # 前向传播：批处理矩阵乘法
    def forward(self, input):
        # 输入：
        #   input: [ensemble_size, batch_size, in_features]
        # 输出：
        #   [ensemble_size, batch_size, out_features]
        # 参数：
        #   input: 每个子模型对应的一批输入
        # 作用：
        #   使用 batched matrix multiply 一次完成多个子模型的线性变换。
        # 输入必须是3维张量：[集成数, 批量大小, 输入维度]
        assert len(input.shape) == 3
        assert input.shape[0] == self.ensemble_size
        # 批量矩阵乘法 + 偏置
        return torch.bmm(input, self.weight.transpose(1, 2)) + self.bias.unsqueeze(1)


# 批处理高斯集成模型：继承配置类、模型类、基础模型类
class BatchedGaussianEnsemble(Configurable, Module, BaseModel):
    # 输入：
    #   无
    # 输出：
    #   一个学习 (state, action) -> (next_state, reward) 的高斯集成模型类
    # 参数：
    #   无
    # 作用：
    #   Safe-MBPO 的动力学模型主体，支持训练、采样、以及 ensemble 均值预测。
    # 模型配置类：所有超参数
    class Config(BaseConfig):
        # 输入：
        #   无
        # 输出：
        #   配置对象
        # 参数：
        #   ensemble_size, hidden_dim, trunk_layers, head_hidden_layers,
        #   activation, init_min_log_var, init_max_log_var,
        #   log_var_bound_weight, batch_size, learning_rate
        # 作用：
        #   定义动力学模型的结构与训练超参数。
        ensemble_size = 5          # 集成模型数量（默认5个）
        hidden_dim = 200           # 隐藏层维度
        trunk_layers = 2           # 共享主干网络层数
        head_hidden_layers = 1     # 输出头隐藏层数
        activation = 'relu'        # 激活函数
        init_min_log_var = -10.0   # 初始最小对数方差
        init_max_log_var = 1.0     # 初始最大对数方差
        log_var_bound_weight = 0.01  # 方差边界约束权重
        batch_size = 256           # 训练批次大小
        learning_rate = 1e-3       # 学习率

    # 初始化模型
    def __init__(self, config, state_dim, action_dim,
                 device=device, optimizer_factory=defaults.OPTIMIZER):
        # 输入：
        #   config: 配置对象
        #   state_dim: 状态维度
        #   action_dim: 动作维度
        #   device: 运行设备
        #   optimizer_factory: 优化器构造函数
        # 输出：
        #   初始化完成的动力学 ensemble
        # 参数：
        #   input_dim = state_dim + action_dim
        #   output_dim = state_dim + 1
        # 作用：
        #   构建 trunk、均值头、方差头、归一化器和优化器。
        Configurable.__init__(self, config)  # 初始化配置
        Module.__init__(self)               # 初始化PyTorch模型

        self.state_dim = state_dim    # 状态维度
        self.action_dim = action_dim  # 动作维度
        input_dim = state_dim + action_dim  # 模型输入维度 = 状态+动作
        output_dim = state_dim + 1          # 模型输出维度 = 下一状态+奖励

        # 可学习的对数方差上下界（约束方差范围，保证数值稳定）
        self.min_log_var = nn.Parameter(torch.full([output_dim], self.init_min_log_var, device=device))
        self.max_log_var = nn.Parameter(torch.full([output_dim], self.init_max_log_var, device=device))
        # 状态归一化器：提升模型训练稳定性
        self.state_normalizer = Normalizer(state_dim)

        # 构建网络层工厂：使用批处理线性层
        layer_factory = lambda n_in, n_out: BatchedLinear(self.ensemble_size, n_in, n_out)
        # 主干网络维度：输入 -> 隐藏层*N
        trunk_dims = [input_dim] + [self.hidden_dim] * self.trunk_layers
        # 输出头维度：隐藏层*N -> 输出
        head_dims = [self.hidden_dim] * (self.head_hidden_layers + 1) + [output_dim]
        
        # 共享主干网络：所有集成模型共享特征提取
        self.trunk = mlp(trunk_dims, layer_factory=layer_factory, activation=self.activation,
                         output_activation=self.activation)
        # 均值输出头：预测状态增量+奖励
        self.diff_head = mlp(head_dims, layer_factory=layer_factory, activation=self.activation)
        # 对数方差输出头：预测分布不确定性
        self.log_var_head = mlp(head_dims, layer_factory=layer_factory, activation=self.activation)
        
        # 模型移至指定设备（CPU/GPU）
        self.to(device)
        # 初始化优化器：更新所有网络参数+方差边界参数
        self.optimizer = optimizer_factory([
            *self.trunk.parameters(),
            *self.diff_head.parameters(),
            *self.log_var_head.parameters(),
            self.min_log_var, self.max_log_var
        ], lr=self.learning_rate)

    # 属性：总批量大小 = 集成数 × 单模型批次
    @property
    def total_batch_size(self):
        # 输入：
        #   无
        # 输出：
        #   ensemble 总 batch 大小
        # 参数：
        #   无
        # 作用：
        #   返回一次训练中所有子模型合计处理的样本数。
        return self.ensemble_size * self.batch_size

    # 单模型前向传播：只使用集成中的某一个模型
    def _forward1(self, states, actions, index):
        # 输入：
        #   states: [B, state_dim]
        #   actions: [B, action_dim]
        #   index: 子模型编号
        # 输出：
        #   means: [B, state_dim + 1]
        #   log_vars: [B, state_dim + 1]
        # 参数：
        #   states/actions: 一批状态动作
        #   index: 指定使用第几个集成成员
        # 作用：
        #   使用单个子模型预测下一状态和奖励的均值与对数方差。
        normalized_states = self.state_normalizer(states)  # 归一化状态
        inputs = torch.cat([normalized_states, actions], dim=-1)  # 拼接状态+动作
        batch_size = inputs.shape[0]
        # 单模型前向（非批处理）
        shared_hidden = unbatched_forward(self.trunk, inputs, index)
        diffs = unbatched_forward(self.diff_head, shared_hidden, index)
        # 均值 = 当前状态 + 预测增量 + 奖励（拼接0向量）
        means = diffs + torch.cat([states, torch.zeros([batch_size, 1], device=device)], dim=1)
        # 预测对数方差
        log_vars = unbatched_forward(self.log_var_head, shared_hidden, index)
        # 约束对数方差在[min, max]范围内
        log_vars = self.max_log_var - F.softplus(self.max_log_var - log_vars)
        log_vars = self.min_log_var + F.softplus(log_vars - self.min_log_var)
        return means, log_vars

    # 全集成模型前向传播：所有模型一起计算
    def _forward_all(self, states, actions):
        # 输入：
        #   states: [E, B, state_dim]
        #   actions: [E, B, action_dim]
        # 输出：
        #   means: [E, B, state_dim + 1]
        #   log_vars: [E, B, state_dim + 1]
        # 参数：
        #   E: ensemble_size
        #   B: batch 大小
        # 作用：
        #   让全部子模型同时前向，便于并行训练或分析 ensemble 分歧。
        normalized_states = self.state_normalizer(states)
        inputs = torch.cat([normalized_states, actions], dim=-1)
        batch_size = inputs.shape[1]
        # 批处理前向
        shared_hidden = self.trunk(inputs)
        diffs = self.diff_head(shared_hidden)
        # 计算均值
        means = diffs + torch.cat([states, torch.zeros([self.ensemble_size, batch_size, 1], device=device)], dim=-1)
        # 计算并约束对数方差
        log_vars = self.log_var_head(shared_hidden)
        log_vars = self.max_log_var - F.softplus(self.max_log_var - log_vars)
        log_vars = self.min_log_var + F.softplus(log_vars - self.min_log_var)
        return means, log_vars

    # 重构批次：将数据按集成模型数量分组
    def _rebatch(self, x):
        # 输入：
        #   x: [total_batch_size, ...]
        # 输出：
        #   [ensemble_size, batch_size, ...]
        # 参数：
        #   x: 平铺的一批样本
        # 作用：
        #   将总 batch 重塑成每个集成成员各自一份的三维张量。
        total_batch_size = len(x)
        assert total_batch_size % self.ensemble_size == 0, f'{total_batch_size} not divisible by {self.ensemble_size}'
        batch_size = total_batch_size // self.ensemble_size
        remaining_dims = tuple(x.shape[1:])
        # 重塑为 [集成数, 单模型批次, 特征维度]
        return x.reshape(self.ensemble_size, batch_size, *remaining_dims)

    # 计算模型损失：高斯负对数似然损失 + 方差约束损失
    def compute_loss(self, states, actions, targets):
        # 输入：
        #   states: [N, state_dim]
        #   actions: [N, action_dim]
        #   targets: [N, state_dim + 1]
        # 输出：
        #   标量损失值
        # 参数：
        #   targets: 由 [next_states, rewards] 拼接得到
        # 作用：
        #   计算高斯预测的似然损失，并限制方差上下界避免数值不稳定。
        inputs = [states, actions, targets]
        total_batch_size = len(targets)
        # 确保批次能被集成数整除
        remainder = total_batch_size % self.ensemble_size
        if remainder != 0:
            nearest = total_batch_size - remainder
            inputs = [x[:nearest] for x in inputs]

        # 重构批次
        states, actions, targets = [self._rebatch(x) for x in inputs]
        # 前向传播得到均值和方差
        means, log_vars = self._forward_all(states, actions)
        inv_vars = torch.exp(-log_vars)  # 方差的倒数
        # 高斯负对数似然核心项
        squared_errors = torch.sum((targets - means)**2 * inv_vars, dim=-1)
        log_dets = torch.sum(log_vars, dim=-1)
        mle_loss = torch.mean(squared_errors + log_dets)
        # 总损失 = 似然损失 + 方差边界约束
        return mle_loss + self.log_var_bound_weight * (self.max_log_var.sum() - self.min_log_var.sum())

    # 模型训练：用经验数据拟合环境模型
    def fit(self, buffer, steps=None, epochs=None, progress_bar=False, **kwargs):
        # 输入：
        #   buffer: 经验池
        #   steps: 按步训练的步数
        #   epochs: 按轮训练的轮数
        #   progress_bar: 是否显示进度条
        # 输出：
        #   loss 列表
        # 参数：
        #   steps 与 epochs 二选一
        # 作用：
        #   从 buffer 中取出监督数据，训练 dynamics ensemble。
        n = len(buffer)
        # 从缓冲区获取训练数据
        states, actions, next_states, rewards = buffer.get()[:4]
        # 拟合状态归一化器
        self.state_normalizer.fit(states)
        # 训练目标 = 下一状态 + 奖励（拼接为一个张量）
        targets = torch.cat([next_states, rewards.unsqueeze(1)], dim=1)

        if steps is not None:
            # 按固定步数训练
            assert epochs is None, 'Cannot pass both steps and epochs'
            losses = []
            for _ in (trange if progress_bar else range)(steps):
                # 随机采样批次数据
                indices = torch.randint(n, [self.total_batch_size], device=device)
                # 计算损失
                loss = self.compute_loss(states[indices], actions[indices], targets[indices])
                losses.append(loss.item())
                # 反向传播更新参数
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            return losses
        elif epochs is not None:
            # 按轮数训练（轮数×集成数，保证每个模型都看到足够数据）
            adjusted_epochs = self.ensemble_size * epochs
            return epochal_training(self.compute_loss, self.optimizer, [states, actions, targets],
                                    epochs=adjusted_epochs,
                                    batch_size=self.total_batch_size, **kwargs)
        else:
            raise ValueError('Must pass steps or epochs')

    # 模型采样：随机选一个集成模型，预测下一状态和奖励
    def sample(self, states, actions):
        # 输入：
        #   states: [B, state_dim]
        #   actions: [B, action_dim]
        # 输出：
        #   next_states: [B, state_dim]
        #   rewards: [B]
        # 参数：
        #   states/actions: 当前状态动作批次
        # 作用：
        #   随机选择一个子模型，并从其高斯分布中采样 (s', r)。
        index = random.randrange(self.ensemble_size)  # 随机选一个模型
        means, log_vars = self._forward1(states, actions, index)
        stds = torch.exp(log_vars).sqrt()  # 标准差
        # 重参数化采样：均值 + 噪声×标准差
        samples = means + stds * torch.randn_like(means)
        # 拆分输出：下一状态（前N维）、奖励（最后1维）
        return samples[:,:-1], samples[:,-1]

    # 获取所有集成模型的均值预测
    def means(self, states, actions):
        # 输入：
        #   states: [B, state_dim]
        #   actions: [B, action_dim]
        # 输出：
        #   next_state_means: [E, B, state_dim]
        #   reward_means: [E, B]
        # 参数：
        #   E: ensemble_size
        # 作用：
        #   返回所有子模型对同一批输入的均值预测，用于评估分歧或取平均。
        # 复制数据给所有集成模型
        states = states.repeat(self.ensemble_size, 1, 1)
        actions = actions.repeat(self.ensemble_size, 1, 1)
        means, _ = self._forward_all(states, actions)
        # 拆分状态和奖励均值
        return means[:,:,:-1], means[:,:,-1]

    # 获取所有集成模型的均值平均（最终预测）
    def mean(self, states, actions):
        # 输入：
        #   states: [B, state_dim]
        #   actions: [B, action_dim]
        # 输出：
        #   next_state_mean: [B, state_dim]
        #   reward_mean: [B]
        # 参数：
        #   states/actions: 当前状态动作批次
        # 作用：
        #   对所有子模型的均值预测再取平均，得到 ensemble 的总体预测。
        next_state_means, reward_means = self.means(states, actions)
        return next_state_means.mean(dim=0), reward_means.mean(dim=0)


# 专用前向函数：针对包含BatchedLinear的Sequential，只调用其中一个模型
def unbatched_forward(batched_sequential, input, index):
    # 输入：
    #   batched_sequential: 含 BatchedLinear 的网络
    #   input: [B, dim]
    #   index: 子模型编号
    # 输出：
    #   单个子模型的前向结果
    # 参数：
    #   batched_sequential: 顺序网络
    #   input: 输入张量
    #   index: 选择第几个 ensemble 成员
    # 作用：
    #   从 batched 网络中抽取某一个子模型执行前向，供 _forward1 使用。
    for layer in batched_sequential:
        # 如果是批处理线性层，只取指定index的模型参数
        if isinstance(layer, BatchedLinear):
            input = F.linear(input, layer.weight[index], layer.bias[index])
        else:
            # 普通层直接前向
            input = layer(input)
    return input
