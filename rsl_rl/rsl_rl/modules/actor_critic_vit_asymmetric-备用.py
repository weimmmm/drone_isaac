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
from torch.distributions import Normal

from rsl_rl.networks import EmpiricalNormalization, MLP


def _activation(name: str) -> nn.Module:
    """Map config strings to torch activation modules."""
    activations = {
        "elu": nn.ELU,
        "relu": nn.ReLU,
        "selu": nn.SELU,
        "crelu": nn.ReLU,
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
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=patch_size, stride=stride, padding=padding)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        x = self.proj(x)
        _, _, height, width = x.shape
        # `[B, C, H, W] -> [B, N, C]`，方便后续 token-based attention 处理。
        x = x.flatten(2).transpose(1, 2)
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
        if channels % num_heads != 0:
            raise ValueError(f"channels={channels} must be divisible by num_heads={num_heads}.")
        self.num_heads = num_heads
        self.reducer = nn.Conv2d(channels, channels, kernel_size=reduction_ratio, stride=reduction_ratio)
        self.reducer_norm = nn.LayerNorm(channels)
        self.key_value = nn.Linear(channels, channels * 2)
        self.query = nn.Linear(channels, channels)
        self.proj = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        batch, num_tokens, channels = x.shape

        # 把 token 重新恢复成 feature map，便于对 key/value 做空间降采样。
        reduced = x.transpose(1, 2).reshape(batch, channels, height, width)
        reduced = self.reducer(reduced)
        reduced = reduced.reshape(batch, channels, -1).transpose(1, 2).contiguous()
        reduced = self.reducer_norm(reduced)

        # key/value 用降采样后的 token，query 保持原分辨率。
        key_value = self.key_value(reduced)
        key_value = key_value.reshape(batch, -1, 2, self.num_heads, channels // self.num_heads).permute(2, 0, 3, 1, 4)
        key, value = key_value[0], key_value[1]
        query = self.query(x).reshape(batch, num_tokens, self.num_heads, channels // self.num_heads).permute(0, 2, 1, 3)

        scale = (channels // self.num_heads) ** 0.5
        attention = torch.softmax((query @ key.transpose(-2, -1)) / scale, dim=-1)
        attended = (attention @ value).transpose(1, 2).reshape(batch, num_tokens, channels)
        return self.proj(attended)


class MixFFN(nn.Module):
    """Feed-forward block with depthwise convolution in the middle.

    这一步先做通道扩展，再把 token 恢复成 feature map，
    用 depthwise conv 注入局部空间归纳偏置，最后再投回原通道数。
    """

    def __init__(self, channels: int, expansion_factor: int):
        super().__init__()
        expanded_channels = channels * expansion_factor
        self.fc1 = nn.Linear(channels, expanded_channels)
        self.depthwise = nn.Conv2d(expanded_channels, expanded_channels, kernel_size=3, padding=1, groups=channels)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(expanded_channels, channels)

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        x = self.fc1(x)
        batch, _, channels = x.shape
        x = x.transpose(1, 2).reshape(batch, channels, height, width)
        x = self.depthwise(x)
        x = self.act(x)
        x = x.flatten(2).transpose(1, 2)
        return self.fc2(x)


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
        self.patch_merge = OverlapPatchMerging(in_channels, out_channels, patch_size, stride, padding)
        self.attn = nn.ModuleList([EfficientSelfAttention(out_channels, reduction_ratio, num_heads) for _ in range(n_layers)])
        self.ffn = nn.ModuleList([MixFFN(out_channels, expansion_factor) for _ in range(n_layers)])
        self.norm = nn.ModuleList([nn.LayerNorm(out_channels) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, _, _ = x.shape
        x, height, width = self.patch_merge(x)
        for attn, ffn, norm in zip(self.attn, self.ffn, self.norm):
            # 这里采用 pre-activation 风格的残差堆叠。
            x = x + attn(x, height, width)
            x = x + ffn(x, height, width)
            x = norm(x)
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
        if image_shape is None:
            raise ValueError("VitFlyImageEncoder requires a valid image_shape.")
        if len(image_shape) != 3:
            raise ValueError(f"VitFlyImageEncoder expects image_shape=(C,H,W), got {image_shape}.")
        if not (
            len(patch_sizes)
            == len(strides)
            == len(paddings)
            == len(embed_dims)
            == len(num_layers)
            == len(reduction_ratios)
            == len(num_heads)
            == len(expansion_factors)
        ):
            raise ValueError("All ViT stage configuration lists must have the same length.")
        if len(embed_dims) < 2:
            raise ValueError("VitFlyImageEncoder expects at least two encoder stages.")

        self.resize_hw = tuple(int(v) for v in resize_hw)
        self.output_activation = _activation(activation)
        self.encoder_blocks = nn.ModuleList()

        # 按 stage 配置堆叠视觉 Transformer 主干。
        in_channels = int(image_shape[0])
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
            in_channels = int(embed_dims[stage_idx])

        # 这里沿用 vitfly 的多尺度融合思路：
        # stage2 走 pixel shuffle，stage1 走上采样，然后拼接压缩。
        self.up_sample = nn.Upsample(size=(16, 24), mode="bilinear", align_corners=True)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
        self.down_sample = nn.Conv2d(48, 12, kernel_size=3, padding=1)

        # 用一个 dummy 输入自动推导 flatten 后的维度，
        # 这样外部改 image shape 时不需要手算全连接输入大小。
        with torch.no_grad():
            sample = torch.zeros(1, *image_shape)
            flattened_dim = self._forward_features(sample).shape[-1]

        self.decoder = nn.Linear(flattened_dim, decoder_dim)
        self.proj = nn.Linear(decoder_dim, image_latent_dim)

    def _resize_if_needed(self, x: torch.Tensor) -> torch.Tensor:
        # 环境分辨率只要不是 ViT 期望大小，就在这里统一 resize。
        if x.shape[-2:] != self.resize_hw:
            x = F.interpolate(x, size=self.resize_hw, mode="bilinear", align_corners=False)
        return x

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self._resize_if_needed(x)
        embeds = [x]
        for block in self.encoder_blocks:
            embeds.append(block(embeds[-1]))

        # 当前实现假设至少有两个 encoder stage，并融合 stage1/stage2 特征。
        stage1, stage2 = embeds[1], embeds[2]
        fused = torch.cat([self.pixel_shuffle(stage2), self.up_sample(stage1)], dim=1)
        fused = self.down_sample(fused)
        return fused.flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._forward_features(x)
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

    is_recurrent = False

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

        # actor_cfg / critic_cfg 是 runner 从配置文件透传进来的字典，
        # 这里统一解包，便于和旧版 flat config 兼容。
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

        # 根据首批 observation 自动推断各分组的维度/形状，
        # 这样不需要把 state_dim / image_shape 手动写死在配置里。
        actor_state_dim, actor_image_shape = self._infer_group_shapes(obs_groups["policy"], obs)
        critic_obs_dim = self._infer_flat_obs_dim(obs_groups["critic"], obs)
        self._policy_state_groups = [name for name in obs_groups["policy"] if len(obs[name].shape[1:]) == 1]
        self._policy_sensor_groups = [name for name in obs_groups["policy"] if len(obs[name].shape[1:]) >= 2]
        self._actor_sensor_ndim = len(actor_image_shape) if actor_image_shape is not None else 0

        self.actor_state_normalizer = (
            EmpiricalNormalization(actor_state_dim) if actor_obs_normalization and actor_state_dim > 0 else nn.Identity()
        )
        self.critic_obs_normalizer = (
            EmpiricalNormalization(critic_obs_dim) if critic_obs_normalization and critic_obs_dim > 0 else nn.Identity()
        )

        # actor 的低维状态编码支路。
        self.actor_state_encoder, actor_state_out_dim = self._build_state_encoder(actor_state_dim, state_hidden_dims, activation)

        # actor 的图像编码支路，核心就是上面的 VitFlyImageEncoder。
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

        # actor 最终只看拼接后的 latent，critic 则是传统低维 value 网络。
        actor_input_dim = actor_state_out_dim + image_latent_dim
        self.actor = MLP(actor_input_dim, num_actions, actor_hidden_dims, activation)
        self.critic = MLP(critic_obs_dim, 1, critic_hidden_dims, critic_activation)

        print(f"Actor VIT-MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # PPO 连续动作分布的标准差参数。
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        self.distribution = None
        Normal.set_default_validate_args(False)

    def reset(self, dones=None):
        # 非 recurrent policy，无需维护隐状态。
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, obs):
        # actor 只输出动作均值；探索噪声来自单独维护的 std / log_std。
        mean = self.actor(self.get_actor_features(obs))
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        self.distribution = Normal(mean, std)

    def act(self, obs, **kwargs):
        self.update_distribution(obs)
        return self.distribution.sample()

    def act_inference(self, obs):
        # 推理阶段直接返回均值动作，不做随机采样。
        return self.actor(self.get_actor_features(obs))

    def evaluate(self, obs, **kwargs):
        return self.critic(self.get_critic_obs(obs))

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs):
        # actor 和 critic 的归一化器分开维护，
        # 因为它们看到的观测空间本来就不同。
        if self.actor_obs_normalization:
            actor_state, _ = self._split_groups(obs, self.obs_groups["policy"])
            if actor_state is not None:
                self.actor_state_normalizer.update(actor_state)
        if self.critic_obs_normalization:
            critic_obs = self.get_critic_obs(obs)
            self.critic_obs_normalizer.update(critic_obs)

    def load_state_dict(self, state_dict, strict=True):
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
            state = self.actor_state_normalizer(state)
            features.append(self.actor_state_encoder(state))
        if image is not None:
            features.append(self.actor_image_encoder(image))
        return torch.cat(features, dim=-1)

    def get_critic_obs(self, obs):
        # critic 始终只吃展平后的低维 privileged 观测。
        obs_list = []
        for obs_group in self.obs_groups["critic"]:
            tensor = obs[obs_group]
            if tensor.ndim == 2:
                obs_list.append(tensor)
            else:
                obs_list.append(tensor.flatten(start_dim=1))
        critic_obs = torch.cat(obs_list, dim=-1)
        return self.critic_obs_normalizer(critic_obs)

    def _split_groups(self, obs, group_names):
        # 按 observation group 的形状把输入自动拆成“状态张量”和“传感器张量”。
        state_tensors = []
        image_tensors = []
        for group_name in group_names:
            tensor = obs[group_name]
            if group_name in self._policy_state_groups:
                state_tensors.append(tensor)
            elif group_name in self._policy_sensor_groups:
                image_tensors.append(tensor)
            else:
                state_tensors.append(tensor.flatten(start_dim=1))
        state = torch.cat(state_tensors, dim=-1) if state_tensors else None
        if image_tensors:
            # 对图像/传感器分组按通道维拼接。
            sensor_cat_dim = -self._actor_sensor_ndim
            image = torch.cat(image_tensors, dim=sensor_cat_dim)
        else:
            image = None
        return state, image

    def _infer_group_shapes(self, group_names, obs):
        # 统计 actor 观测里低维状态的总维度，并推断图像 shape。
        state_dim = 0
        image_shape = None
        for group_name in group_names:
            shape = obs[group_name].shape[1:]
            if len(shape) == 1:
                state_dim += shape[0]
            elif len(shape) in (2, 3):
                if image_shape is None:
                    image_shape = list(shape)
                else:
                    image_shape[0] += shape[0]
            else:
                state_dim += int(torch.tensor(shape).prod().item())
        return state_dim, tuple(image_shape) if image_shape is not None else None

    def _infer_flat_obs_dim(self, group_names, obs):
        # 统计 critic 输入展平后的总维度。
        dim = 0
        for group_name in group_names:
            shape = obs[group_name].shape[1:]
            if len(shape) == 1:
                dim += shape[0]
            else:
                dim += int(torch.tensor(shape).prod().item())
        return dim

    def _build_state_encoder(self, input_dim, hidden_dims, activation):
        # 状态编码支路支持三种情况：
        # 1. 没有状态输入
        # 2. 不额外编码，直接透传
        # 3. 用 1 层或多层 MLP 做状态特征提取
        if input_dim == 0:
            return nn.Identity(), 0
        if not hidden_dims:
            return nn.Identity(), input_dim
        if len(hidden_dims) == 1:
            return nn.Sequential(nn.Linear(input_dim, hidden_dims[0]), _activation(activation)), hidden_dims[0]
        return MLP(input_dim, hidden_dims[-1], hidden_dims[:-1], activation), hidden_dims[-1]
