# Copyright (c) 2026
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

"""ViT-based asymmetric actor-critic used by the depth-image student policy.

整体思路：
1. actor 接收两路 policy 观测：低维状态 + 深度图。
2. 低维状态走一个小 MLP；深度图走 vitfly 风格的分层 ViT 编码器。
3. 两路特征拼接后送入 actor MLP，输出动作均值。
4. critic 不看图像，只看低维 privileged critic 观测。

这里实现的是“把 ViT 当成视觉 backbone 接到 RL actor 前面”，
而不是直接复刻 vitfly 里最终的控制头。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal  # 用于构建连续动作空间的的正态分布

# rsl_rl 是常用的机器人强化学习库，这里引入了经验归一化层和多层感知机模块
from rsl_rl.networks import EmpiricalNormalization, MLP


def _activation(name: str) -> nn.Module:
    """Map config strings to torch activation modules."""
    # 字典映射：将配置文件中的字符串转换为 PyTorch 的激活函数实例
    activations = {
        "elu": nn.ELU,
        "relu": nn.ReLU,
        "selu": nn.SELU,
        "crelu": nn.ReLU,  # 注意：这里 crelu 映射到了 ReLU，可能是原作者的妥协或特定设定
        "lrelu": nn.LeakyReLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
    }
    if name not in activations:
        raise ValueError(f"Unsupported activation: {name}")
    return activations[name]()


class OverlapPatchMerging(nn.Module):
    """Convert an image feature map into overlapped patch tokens.
    
    输入:  `[B, C, H, W]`
    输出:
    - token 序列 `[B, N, C_out]`
    - token 对应的空间尺寸 `(H_out, W_out)`

    这一步对应 vitfly / SegFormer 风格的 patch embedding，
    用带重叠的卷积而不是非重叠切块。
    """

    def __init__(self, in_channels: int, out_channels: int, patch_size: int, stride: int, padding: int):
        super().__init__()
        # 使用步长小于卷积核大小的 Conv2d 来实现 Overlap (重叠) 切块，保留更多局部连续性信息
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=patch_size, stride=stride, padding=padding)
        # 归一化层，作用于输出的通道维度
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        # x shape: [Batch, in_channels, H, W]
        x = self.proj(x)
        # 获取卷积后新的特征图尺寸
        _, _, height, width = x.shape
        # flatten(2) 将 H 和 W 展平，维度变成 [B, C_out, H*W]
        # transpose(1, 2) 交换维度，变成 Transformer 标准输入格式: [B, N, C_out] (N = H*W)
        x = x.flatten(2).transpose(1, 2)
        # 对 token 的特征维度进行 LayerNorm
        x = self.norm(x)
        return x, height, width


class EfficientSelfAttention(nn.Module):
    """Self-attention with spatial reduction on key/value tokens.

    为了避免图像 token 太多导致注意力开销过大，
    这里先对 key/value 分支做一次空间降采样，再和原始 query 做注意力。
    这是 vitfly 原始实现里比较重要的“省算力”设计。
    """

    def __init__(self, channels: int, reduction_ratio: int, num_heads: int):
        super().__init__()
        # 确保通道数可以被头数整除
        if channels % num_heads != 0:
            raise ValueError(f"channels={channels} must be divisible by num_heads={num_heads}.")
        self.num_heads = num_heads
        
        # 降采样器：使用 kernel_size = stride = reduction_ratio 的卷积，缩小特征图的空间尺寸
        self.reducer = nn.Conv2d(channels, channels, kernel_size=reduction_ratio, stride=reduction_ratio)
        self.reducer_norm = nn.LayerNorm(channels)
        
        # Key 和 Value 的线性映射层，输出维度是 channels * 2 (各占一半)
        self.key_value = nn.Linear(channels, channels * 2)
        # Query 的线性映射层，Query 保持原有分辨率（不降采样）
        self.query = nn.Linear(channels, channels)
        # Self-Attention 最后的输出映射层
        self.proj = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        batch, num_tokens, channels = x.shape

        # 把 token 序列 [B, N, C] 重新恢复成特征图 [B, C, H, W]，便于进行卷积降采样
        reduced = x.transpose(1, 2).reshape(batch, channels, height, width)
        reduced = self.reducer(reduced) # 空间降采样
        # 降采样后再展平成 token 序列 [B, N_reduced, C]
        reduced = reduced.reshape(batch, channels, -1).transpose(1, 2).contiguous()
        reduced = self.reducer_norm(reduced)

        # 生成 Key 和 Value：使用降采样后的 token
        key_value = self.key_value(reduced)
        # 重塑维度以拆分多头和 K/V：[B, N_reduced, 2, num_heads, head_dim] -> [2, B, num_heads, N_reduced, head_dim]
        key_value = key_value.reshape(batch, -1, 2, self.num_heads, channels // self.num_heads).permute(2, 0, 3, 1, 4)
        key, value = key_value[0], key_value[1] # 分离出 Key 和 Value
        
        # 生成 Query：使用原始的高分辨率 token x
        # 重塑：[B, num_tokens, num_heads, head_dim] -> [B, num_heads, num_tokens, head_dim]
        query = self.query(x).reshape(batch, num_tokens, self.num_heads, channels // self.num_heads).permute(0, 2, 1, 3)

        # 注意力缩放因子 1 / sqrt(d_k)
        scale = (channels // self.num_heads) ** 0.5
        # 计算 Attention 分数: (Q * K^T) / scale -> Softmax
        attention = torch.softmax((query @ key.transpose(-2, -1)) / scale, dim=-1)
        # 乘以 Value，然后恢复原本的 token 形状 [B, num_tokens, channels]
        attended = (attention @ value).transpose(1, 2).reshape(batch, num_tokens, channels)
        # 经过最终的线性映射输出
        return self.proj(attended)


class MixFFN(nn.Module):
    """Feed-forward block with depthwise convolution in the middle.

    这一步先做通道扩展，再把 token 恢复成 feature map，
    用 depthwise conv 注入局部空间归纳偏置，最后再投回原通道数。
    """

    def __init__(self, channels: int, expansion_factor: int):
        super().__init__()
        expanded_channels = channels * expansion_factor # 隐藏层通常比输入通道大
        self.fc1 = nn.Linear(channels, expanded_channels) # 第一层全连接，扩展维度
        # 深度可分离卷积 (Depthwise Conv)：groups=expanded_channels 表示每个通道独立卷积，极大减少参数量
        # 这里的 3x3 卷积用来捕捉局部空间信息，弥补纯 Transformer 缺乏空间归纳偏置的缺点
        self.depthwise = nn.Conv2d(expanded_channels, expanded_channels, kernel_size=3, padding=1, groups=expanded_channels)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(expanded_channels, channels) # 第二层全连接，降回原维度

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        x = self.fc1(x) # [B, N, expanded_channels]
        batch, _, channels = x.shape
        # [B, N, C] -> [B, C, N] -> [B, C, H, W] 恢复为二维特征图
        x = x.transpose(1, 2).reshape(batch, channels, height, width)
        x = self.depthwise(x) # 应用 3x3 空间卷积
        x = self.act(x) # 激活函数
        # 再次展平为 token 序列 [B, N, C]
        x = x.flatten(2).transpose(1, 2)
        return self.fc2(x) # 映射回原始 channels 维度


class MixTransformerEncoderLayer(nn.Module):
    """One hierarchical ViT stage.

    一个 stage 先做 overlapped patch merging，
    再叠加若干层 attention + MixFFN + LayerNorm。
    输出仍然恢复成 `[B, C, H, W]`，便于后续 stage 继续按 feature map 处理。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: int,
        stride: int,
        padding: int,
        n_layers: int,
        reduction_ratio: int,
        num_heads: int,
        expansion_factor: int,
    ):
        super().__init__()
        # 首先进行 Patch Merging（降采样，增加通道数）
        self.patch_merge = OverlapPatchMerging(in_channels, out_channels, patch_size, stride, padding)
        # 构建多个 Transformer Block (包含 Attention, FFN, Normalization)
        self.attn = nn.ModuleList([EfficientSelfAttention(out_channels, reduction_ratio, num_heads) for _ in range(n_layers)])
        self.ffn = nn.ModuleList([MixFFN(out_channels, expansion_factor) for _ in range(n_layers)])
        self.norm = nn.ModuleList([nn.LayerNorm(out_channels) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, _, _ = x.shape
        # x 被降采样并转换为 token 序列，同时获取当前的 height, width
        x, height, width = self.patch_merge(x)
        
        # 遍历叠加的 n_layers 层 Attention 和 FFN
        for attn, ffn, norm in zip(self.attn, self.ffn, self.norm):
            # 这里采用 pre-activation 风格的残差堆叠。
            # 注意：代码里其实是 Post-Norm 风格（先加，后Norm）。通常 Pre-Norm 会写成 x = x + attn(norm(x))。
            # 以代码实际逻辑为准：Attention -> Residual -> FFN -> Residual -> Norm。
            x = x + attn(x, height, width)
            x = x + ffn(x, height, width)
            x = norm(x)
            
        # 将 [B, H*W, C] 重新整形并交换维度，变回 [B, C, H, W] 交给下一个 Stage
        return x.reshape(batch, height, width, -1).permute(0, 3, 1, 2).contiguous()


class VitFlyImageEncoder(nn.Module):
    """Visual encoder adapted from vitfly for single-channel depth images.

    数据流大致是：
    `[B, 1, H, W]`
      -> resize 到固定输入尺寸
      -> stage1 encoder
      -> stage2 encoder
      -> 多尺度特征融合
      -> flatten + decoder + projection
      -> 输出固定维度的图像 latent
    """

    def __init__(
        self,
        image_shape,
        image_latent_dim: int,
        resize_hw=(60, 90),
        patch_sizes=(7, 3),
        strides=(4, 2),
        paddings=(3, 1),
        embed_dims=(32, 64),
        num_layers=(2, 2),
        reduction_ratios=(8, 4),
        num_heads=(1, 2),
        expansion_factors=(8, 8),
        decoder_dim: int = 512,
        activation: str = "elu",
    ):
        super().__init__()
        # 参数校验...确保所有 Stage 相关的配置列表长度一致
        if image_shape is None:
            raise ValueError("VitFlyImageEncoder requires a valid image_shape.")
        if len(image_shape) != 3:
            raise ValueError(f"VitFlyImageEncoder expects image_shape=(C,H,W), got {image_shape}.")
        if not (
            len(patch_sizes) == len(strides) == len(paddings) == len(embed_dims) == len(num_layers) 
            == len(reduction_ratios) == len(num_heads) == len(expansion_factors)
        ):
            raise ValueError("All ViT stage configuration lists must have the same length.")
        if len(embed_dims) < 2:
            raise ValueError("VitFlyImageEncoder expects at least two encoder stages.")

        self.resize_hw = tuple(int(v) for v in resize_hw)
        self.output_activation = _activation(activation)
        self.encoder_blocks = nn.ModuleList()

        in_channels = int(image_shape[0]) # 深度图一般是单通道，所以 in_channels 通常是 1
        # 按 stage 配置堆叠视觉 Transformer 主干（实例化每个 Stage）
        for stage_idx in range(len(embed_dims)):
            self.encoder_blocks.append(
                MixTransformerEncoderLayer(
                    in_channels=in_channels,
                    out_channels=int(embed_dims[stage_idx]),
                    patch_size=int(patch_sizes[stage_idx]),
                    stride=int(strides[stage_idx]),
                    padding=int(paddings[stage_idx]),
                    n_layers=int(num_layers[stage_idx]),
                    reduction_ratio=int(reduction_ratios[stage_idx]),
                    num_heads=int(num_heads[stage_idx]),
                    expansion_factor=int(expansion_factors[stage_idx]),
                )
            )
            in_channels = int(embed_dims[stage_idx]) # 下一层的输入通道数是这一层的输出通道数

        # 这里沿用 vitfly 的多尺度融合思路：
        # Stage1 的分辨率高（低语义），Stage2 的分辨率低（高语义）。
        # 上采样 Stage1，使尺寸统一。使用双线性插值。
        self.up_sample = nn.Upsample(size=(16, 24), mode="bilinear", align_corners=True)
        # Stage2 走 PixelShuffle 进行上采样。PixelShuffle 会将通道维度转化为空间分辨率 (比如 C=64 -> C'=16, H'=H*2, W'=W*2)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
        # 融合后通过卷积降维提取最终融合特征
        self.down_sample = nn.Conv2d(48, 12, kernel_size=3, padding=1)

        # 用一个 dummy (虚拟) 输入自动推导 flatten 后的特征维度长度
        # 这是个很聪明的做法，可以避免根据卷积、池化手动计算全连接层的输入大小
        with torch.no_grad():
            sample = torch.zeros(1, *image_shape)
            flattened_dim = self._forward_features(sample).shape[-1]

        # 视觉编码器的解码头，将拉平的视觉特征映射到指定的 latent 空间
        self.decoder = nn.Linear(flattened_dim, decoder_dim)
        self.proj = nn.Linear(decoder_dim, image_latent_dim)

    def _resize_if_needed(self, x: torch.Tensor) -> torch.Tensor:
        # 如果传入的环境观测图像分辨率不符合预期 (resize_hw)，则自动 resize
        if x.shape[-2:] != self.resize_hw:
            x = F.interpolate(x, size=self.resize_hw, mode="bilinear", align_corners=False)
        return x

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self._resize_if_needed(x)
        embeds = [x] # 保存各层输出，用于后续多尺度融合
        
        # 依次经过所有 Stage
        for block in self.encoder_blocks:
            embeds.append(block(embeds[-1]))

        # 当前实现假设至少有两个 encoder stage，并融合 stage1 / stage2 特征。
        stage1, stage2 = embeds[1], embeds[2]
        # PixelShuffle 对 stage2 放大，Upsample 对 stage1 修改尺寸，将它们在通道维度拼接 (Concat)
        fused = torch.cat([self.pixel_shuffle(stage2), self.up_sample(stage1)], dim=1)
        # 融合后通过降采样卷积提取局部信息
        fused = self.down_sample(fused)
        # 将 [B, C, H, W] 展平成一维向量 [B, C*H*W]
        return fused.flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._forward_features(x)
        # 通过两层 MLP (decoder + proj) 得到最终的隐向量
        x = self.decoder(x)
        x = self.output_activation(x)
        x = self.proj(x)
        return self.output_activation(x)


class ActorCriticVitAsymmetric(nn.Module):
    """Asymmetric actor-critic with ViT actor encoder and low-dim critic.

    - actor: `policy_state + policy_image`
    - critic: `critic_base_state + critic_privileged`

    命名里的 asymmetric 指的是 actor 和 critic 看到的观测不同：
    actor 需要适应真实部署时可见的状态和图像，
    critic 则可以使用训练时才有的 privileged 低维信息。
    """

    is_recurrent = False # 声明这不是一个包含 RNN (LSTM/GRU) 的策略网络

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        state_hidden_dims=[128],
        image_latent_dim=128,
        actor_cfg: dict | None = None,
        critic_cfg: dict | None = None,
        **kwargs,
    ):
        # ...忽略不期望的 kwargs，防止配置错误...
        if kwargs:
            print(
                "ActorCriticVitAsymmetric.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        # 默认超参数基本对应当前项目里采用的 vitfly 风格两级编码器配置。
        vit_resize_hw = [60, 90]
        vit_patch_sizes = [7, 3]
        vit_stage_strides = [4, 2]
        vit_paddings = [3, 1]
        vit_embed_dims = [32, 64]
        vit_num_layers = [2, 2]
        vit_reduction_ratios = [8, 4]
        vit_num_heads = [1, 2]
        vit_expansion_factors = [8, 8]
        vit_decoder_dim = 512

        # 解析外部传入的 config 字典，覆盖上面的默认值。
        if actor_cfg is not None:
            actor_hidden_dims = actor_cfg["hidden_dims"]
            state_hidden_dims = actor_cfg.get("state_hidden_dims", state_hidden_dims)
            image_latent_dim = actor_cfg.get("image_latent_dim", image_latent_dim)
            vit_resize_hw = actor_cfg.get("vit_resize_hw", vit_resize_hw)
            vit_patch_sizes = actor_cfg.get("vit_patch_sizes", vit_patch_sizes)
            vit_stage_strides = actor_cfg.get("vit_strides", vit_stage_strides)
            vit_paddings = actor_cfg.get("vit_paddings", vit_paddings)
            vit_embed_dims = actor_cfg.get("vit_embed_dims", vit_embed_dims)
            vit_num_layers = actor_cfg.get("vit_num_layers", vit_num_layers)
            vit_reduction_ratios = actor_cfg.get("vit_reduction_ratios", vit_reduction_ratios)
            vit_num_heads = actor_cfg.get("vit_num_heads", vit_num_heads)
            vit_expansion_factors = actor_cfg.get("vit_expansion_factors", vit_expansion_factors)
            vit_decoder_dim = actor_cfg.get("vit_decoder_dim", vit_decoder_dim)
            activation = actor_cfg.get("activation", activation)

        critic_activation = activation
        if critic_cfg is not None:
            critic_hidden_dims = critic_cfg["hidden_dims"]
            critic_activation = critic_cfg.get("activation", critic_activation)

        self.obs_groups = obs_groups
        self.actor_obs_normalization = actor_obs_normalization
        self.critic_obs_normalization = critic_obs_normalization
        self.noise_std_type = noise_std_type

        # 根据传入的首批 observation（字典形式）自动推断各分组的维度/形状
        # 分离出 state 和 image 的维度信息，以动态构建后续网络结构。
        actor_state_dim, actor_image_shape = self._infer_group_shapes(obs_groups["policy"], obs)
        critic_obs_dim = self._infer_flat_obs_dim(obs_groups["critic"], obs)
        
        # 识别 policy 分组里，哪些键是低维向量（ndim=1），哪些是高维传感器图（ndim>=2）
        self._policy_state_groups = [name for name in obs_groups["policy"] if len(obs[name].shape[1:]) == 1]
        self._policy_sensor_groups = [name for name in obs_groups["policy"] if len(obs[name].shape[1:]) >= 2]
        self._actor_sensor_ndim = len(actor_image_shape) if actor_image_shape is not None else 0

        # 初始化 EmpiricalNormalization（运行平均的均值和方差来进行在线归一化）
        self.actor_state_normalizer = (
            EmpiricalNormalization(actor_state_dim) if actor_obs_normalization and actor_state_dim > 0 else nn.Identity()
        )
        self.critic_obs_normalizer = (
            EmpiricalNormalization(critic_obs_dim) if critic_obs_normalization and critic_obs_dim > 0 else nn.Identity()
        )

        # 构建 actor 的低维状态特征提取器
        self.actor_state_encoder, actor_state_out_dim = self._build_state_encoder(actor_state_dim, state_hidden_dims, activation)

        # 构建 actor 的图像编码器（即上面的 VitFlyImageEncoder 实例）
        self.actor_image_encoder = VitFlyImageEncoder(
            image_shape=actor_image_shape,
            image_latent_dim=image_latent_dim,
            resize_hw=vit_resize_hw,
            patch_sizes=vit_patch_sizes,
            strides=vit_stage_strides,
            paddings=vit_paddings,
            embed_dims=vit_embed_dims,
            num_layers=vit_num_layers,
            reduction_ratios=vit_reduction_ratios,
            num_heads=vit_num_heads,
            expansion_factors=vit_expansion_factors,
            decoder_dim=vit_decoder_dim,
            activation=activation,
        )

        # actor 的最终 MLP：输入是（低维状态特征维度 + 图像特征隐向量维度）
        actor_input_dim = actor_state_out_dim + image_latent_dim
        self.actor = MLP(actor_input_dim, num_actions, actor_hidden_dims, activation)
        
        # critic 的 MLP：只接收所有展平拼接在一起的低维 Critic 观测
        self.critic = MLP(critic_obs_dim, 1, critic_hidden_dims, critic_activation)

        print(f"Actor VIT-MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # PPO 连续动作分布的标准差参数，这些参数是可学习的 (nn.Parameter)
        # scalar 表示用标准差的原始值，log 表示使用 log_std 以确保在指数映射后恒为正
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        self.distribution = None
        # 关闭 PyTorch 分布计算的参数校验，以稍微提升前向传播速度
        Normal.set_default_validate_args(False)

    def reset(self, dones=None):
        # 非 recurrent policy，无需维护隐状态 (如 LSTM 的 hidden state)。
        pass

    def forward(self):
        # 强制使用者调用 act() 或 evaluate()，不提供直接的 forward
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        # 计算策略动作分布的熵，在 PPO 中用于鼓励探索 (Entropy Bonus)
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, obs):
        # actor 只输出动作的均值
        mean = self.actor(self.get_actor_features(obs))
        
        # 将全局的可学习 std/log_std 扩张成与 batch_size 相同形状的张量
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        
        # 实例化 Normal 分布，供后续 sample 或求 log_prob 使用
        self.distribution = Normal(mean, std)

    def act(self, obs, **kwargs):
        # 在训练时 (Rollout) 调用：更新正态分布并采样实际执行的动作
        self.update_distribution(obs)
        return self.distribution.sample()

    def act_inference(self, obs):
        # 推理/评估阶段调用：直接返回均值动作（不包含随机噪声采样，也就是 greedy action）
        return self.actor(self.get_actor_features(obs))

    def evaluate(self, obs, **kwargs):
        # PPO 计算 Value 时调用：根据 Critic 观测返回当前 state 的价值 V(s)
        return self.critic(self.get_critic_obs(obs))

    def get_actions_log_prob(self, actions):
        # PPO 算法核心：计算选取某动作对数概率 (log π(a|s))，最后用于计算 Importance Ratio
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs):
        # actor 和 critic 的归一化器分开维护，
        # 因为它们看到的观测空间本来就不同。
        if self.actor_obs_normalization:
            # 取出 Actor 视角的观测
            actor_state, _ = self._split_groups(obs, self.obs_groups["policy"])
            if actor_state is not None:
                self.actor_state_normalizer.update(actor_state)
        if self.critic_obs_normalization:
            # 取出 Critic 视角的观测
            critic_obs = self.get_critic_obs(obs)
            self.critic_obs_normalizer.update(critic_obs)

    def load_state_dict(self, state_dict, strict=True):
        # 重写一下用于兼容加载模型，本质上直接调父类
        super().load_state_dict(state_dict, strict=strict)
        return True

    def get_actor_features(self, obs):
        # policy 观测被拆成两路：
        # state -> state encoder
        # image -> ViT encoder
        # 然后在 feature 级拼接。
        state, image = self._split_groups(obs, self.obs_groups["policy"])
        features = []
        if state is not None:
            # 低维状态归一化并过 MLP
            state = self.actor_state_normalizer(state)
            features.append(self.actor_state_encoder(state))
        if image is not None:
            # 图像经过之前实例化的 Vision Transformer
            features.append(self.actor_image_encoder(image))
            
        # 沿特征维度（dim=-1）将两个潜变量拼接起来作为 Actor 动作网络的输入
        return torch.cat(features, dim=-1)

    def get_critic_obs(self, obs):
        # critic 始终只吃展平后的低维 privileged 观测。
        obs_list = []
        for obs_group in self.obs_groups["critic"]:
            tensor = obs[obs_group]
            if tensor.ndim == 2:
                # 已经是 [Batch, Dim] 就直接加进去
                obs_list.append(tensor)
            else:
                # 如果 Critic 输入里有高维数据，也强行拉平成 1D 特征
                obs_list.append(tensor.flatten(start_dim=1))
        # 拼接所有 Critic 视角的观测并进行归一化
        critic_obs = torch.cat(obs_list, dim=-1)
        return self.critic_obs_normalizer(critic_obs)

    def _split_groups(self, obs, group_names):
        # 按 observation group 的形状把输入自动拆成“低维状态张量”和“高维传感器张量”。
        state_tensors = []
        image_tensors = []
        for group_name in group_names:
            tensor = obs[group_name]
            if group_name in self._policy_state_groups:
                state_tensors.append(tensor)
            elif group_name in self._policy_sensor_groups:
                image_tensors.append(tensor)
            else:
                # 保底处理：如果不明确属于谁，强行作为低维状态被拉平
                state_tensors.append(tensor.flatten(start_dim=1))
                
        # 对同一类观测沿着最后一个维度（通道或特征向量）拼接起来
        state = torch.cat(state_tensors, dim=-1) if state_tensors else None
        if image_tensors:
            # 对图像/传感器分组按通道维拼接。
            # 例: [Batch, C1, H, W] 和 [Batch, C2, H, W] 会拼成 [Batch, C1+C2, H, W]
            sensor_cat_dim = -self._actor_sensor_ndim
            image = torch.cat(image_tensors, dim=sensor_cat_dim)
        else:
            image = None
        return state, image

    def _infer_group_shapes(self, group_names, obs):
        # 根据传入的一个 sample 自动统计 actor 观测里低维状态的总维度，并推断图像的整体 shape。
        state_dim = 0
        image_shape = None
        for group_name in group_names:
            shape = obs[group_name].shape[1:]  # 去掉 Batch 维度
            if len(shape) == 1:
                # 一维就是单纯的向量，直接累加维度
                state_dim += shape[0]
            elif len(shape) in (2, 3):
                # 如果是2维或3维，代表是图像矩阵。累加通道数。
                if image_shape is None:
                    image_shape = list(shape)
                else:
                    image_shape[0] += shape[0] # 通道数叠加
            else:
                # 如果遇到了 4D 或者更高维度的数据，也默认当作低维拉平
                state_dim += int(torch.tensor(shape).prod().item())
        return state_dim, tuple(image_shape) if image_shape is not None else None

    def _infer_flat_obs_dim(self, group_names, obs):
        # 统计 critic 输入展平后的总特征维度大小。
        dim = 0
        for group_name in group_names:
            shape = obs[group_name].shape[1:]
            if len(shape) == 1:
                dim += shape[0]
            else:
                # 强制全部乘起来拉平
                dim += int(torch.tensor(shape).prod().item())
        return dim

    def _build_state_encoder(self, input_dim, hidden_dims, activation):
        # 状态编码支路支持三种情况的工厂函数：
        # 1. 没有状态输入：返回 nn.Identity，不处理
        # 2. 不额外编码，直接透传：当 hidden_dims 为空列表
        # 3. 用 1 层或多层 MLP 做状态特征提取
        if input_dim == 0:
            return nn.Identity(), 0
        if not hidden_dims:
            return nn.Identity(), input_dim
        if len(hidden_dims) == 1:
            # 单层时，仅经过一次 Linear 和 激活函数
            return nn.Sequential(nn.Linear(input_dim, hidden_dims[0]), _activation(activation)), hidden_dims[0]
        # 多层时，使用 rsl_rl 提供的 MLP 类
        return MLP(input_dim, hidden_dims[-1], hidden_dims[:-1], activation), hidden_dims[-1]