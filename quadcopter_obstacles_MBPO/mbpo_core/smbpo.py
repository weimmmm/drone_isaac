# 导入数值计算库numpy
import numpy as np
# 导入深度学习框架PyTorch
import torch
import traceback
# 导入进度条库
from tqdm import trange

# 导入配置基类和可配置基类
from .config import BaseConfig, Configurable
# 导入高斯集成动态模型（环境模型）
from .dynamics import BatchedGaussianEnsemble
# 导入日志工具
from .log import TabularLog, default_log as log
# 导入均匀策略、高斯噪声策略
from .policy import UniformPolicy, GaussianNoisePolicy
# 导入安全样本缓冲池
from .shared import SafetySampleBuffer
# 导入安全SAC算法
from .ssac import SSAC
# 导入PyTorch工具、设备、随机选择、分位数工具
from .torch_util import Module, DummyModuleWrapper, device, random_choice, deciles
# 导入均值计算工具
from .util import pythonic_mean

# 损失值平均窗口大小：计算最近10次损失的平均值
LOSS_AVERAGE_WINDOW = 10

# SMBPO核心类：继承可配置类和PyTorch模块，与环境无关，适配批量Isaac模拟器
class SMBPOCore(Configurable, Module):
    """Environment-agnostic MBPO core wired for the batched Isaac adapter."""

    # 内部配置类：继承基础配置，定义所有超参数
    class Config(BaseConfig):
        # 安全SAC配置
        sac_cfg = SSAC.Config()
        # 高斯集成模型配置
        model_cfg = BatchedGaussianEnsemble.Config()
        # 模型初始训练步数
        model_initial_steps = 10000
        # 模型每次更新步数
        model_steps = 2000
        # 模型更新周期（每多少步更新一次）
        model_update_period = 250
        # 是否只使用 warmup 阶段采集到的真实数据拟合一次动力学模型。
        # 默认关闭：actor 接管后数据分布会变化，动力学模型需要继续吸收真实 rollout。
        warmup_only_model_fit = False
        # 模型开始训练的轮数
        model_start_epoch = 0
        # 模型开始训练的步数
        model_start_steps = 0
        # 求解器（策略）开始训练的步数
        solver_start_steps = 1000
        # warmup 持续的 epoch 数。前 warmup_epochs 个 epoch 只采随机探索数据并拟合动力学模型。
        warmup_epochs = 0
        # 预热阶段动作噪声标准差
        warmup_action_std = 0.35
        # warmup 每个 episode 固定采一个三维速度动作：vx/vy 覆盖 [-1, 1]，vz 覆盖 [-0.5, 0.5]。
        warmup_xy_action_limit = 1.0
        warmup_z_action_limit = 0.5
        # actor 接管训练初期的前向行为探索偏置，帮助真实采样继续进入地图。
        actor_forward_bias_start = 0.5
        # 前向偏置线性退火 epoch 数；到该 epoch 后完全交给 actor 自己控制。
        actor_forward_bias_anneal_epochs = 200
        # actor 真实环境采样动作保持步数。
        # 默认设为 1，和 NavRL 一样基本每个 env step 都重新输出动作；
        # 如果想恢复“持续半秒朝一个方向飞”的探索风格，再手动调大。
        actor_action_hold_steps = 1
        # actor 真实环境采样阶段限制垂直动作，减少高度死亡主导早期数据。
        actor_vertical_action_limit = 0.25
        # 模型推演长度
        horizon = 5
        # 缓冲池最小数据量
        buffer_min = 65536
        # 真实经验缓冲池最大容量
        replay_buffer_max = 10_000_000
        # 虚拟经验缓冲池最大容量
        virt_buffer_max = 700_000
        # 每轮迭代步数
        steps_per_epoch = 1000
        # 推演批次大小
        rollout_batch_size = 256
        # 每步策略更新次数
        solver_updates_per_step = 10
        # 真实数据占比（训练策略时），会作为调度起点。0.9 表示真实:虚拟 = 9:1
        real_fraction = 0.9
        # 真实数据占比调度终点：0.5 表示真实:虚拟 = 1:1
        real_fraction_final = 0.5
        # 用多少个 epoch 从 real_fraction 线性过渡到 real_fraction_final
        real_fraction_schedule_epochs = 3000
        # 最大回合步数
        max_episode_steps = 6000

    # 初始化函数
    def __init__(
        self,
        config,         # 配置参数
        env,            # 环境对象
        state_dim,      # 状态维度
        action_dim,     # 动作维度
        action_space,   # 动作空间
        check_done_fn,  # 检查回合结束的函数
        check_violation_fn,  # 检查安全约束违反的函数
    ):
        # 初始化可配置父类
        Configurable.__init__(self, config)
        # 初始化PyTorch模块父类
        Module.__init__(self)
        # 保存环境
        self.env = env
        # 获取环境数量（批量环境），默认为1
        self.n_envs = getattr(env, "num_envs", 1)
        # 保存状态维度
        self.state_dim = state_dim
        # 保存动作维度
        self.action_dim = action_dim
        # 保存回合结束检查函数
        self.check_done = check_done_fn
        # 保存安全约束检查函数
        self.check_violation = check_violation_fn

        # 初始化安全SAC求解器（策略网络）
        self.solver = SSAC(self.sac_cfg, state_dim, action_dim, self.horizon)
        # 初始化批量高斯集成环境模型
        self.model_ensemble = BatchedGaussianEnsemble(self.model_cfg, state_dim, action_dim)
        # 创建真实经验缓冲池
        self.replay_buffer = self._create_buffer(self.replay_buffer_max)
        # 创建虚拟（模型生成）经验缓冲池
        self.virt_buffer = self._create_buffer(self.virt_buffer_max)
        # 初始化均匀随机策略
        self.uniform_policy = UniformPolicy(action_space)
        # 初始化预热高斯噪声策略
        self.warmup_policy = GaussianNoisePolicy(action_space, std=self.warmup_action_std)

        # 注册持久化张量：采样回合数
        self.register_buffer("episodes_sampled", torch.tensor(0))
        # 注册持久化张量：采样总步数
        self.register_buffer("steps_sampled", torch.tensor(0))
        # 注册持久化张量：环境奖励总和
        self.register_buffer("env_reward_sum", torch.tensor(0.0))
        # 注册持久化张量：环境奖励计数
        self.register_buffer("env_reward_count", torch.tensor(0))
        # 注册持久化张量：成功次数
        self.register_buffer("n_successes", torch.tensor(0))
        # 注册持久化张量：约束违反次数
        self.register_buffer("n_violations", torch.tensor(0))
        # 注册持久化张量：碰撞次数
        self.register_buffer("n_collisions", torch.tensor(0))
        # 注册持久化张量：超时次数
        self.register_buffer("n_timeouts", torch.tensor(0))
        # 注册持久化张量：高度过低终止次数
        self.register_buffer("n_too_low", torch.tensor(0))
        # 注册持久化张量：高度过高终止次数
        self.register_buffer("n_too_high", torch.tensor(0))
        # 注册持久化张量：水平飞出训练区域终止次数
        self.register_buffer("n_out_of_bounds", torch.tensor(0))
        # 动作和进图诊断累计量
        self.register_buffer("action_abs_sum", torch.tensor(0.0))
        self.register_buffer("action_element_count", torch.tensor(0))
        self.register_buffer("vx_cmd_sum", torch.tensor(0.0))
        self.register_buffer("vy_cmd_sum", torch.tensor(0.0))
        self.register_buffer("vz_cmd_sum", torch.tensor(0.0))
        self.register_buffer("cmd_count", torch.tensor(0))
        self.register_buffer("inside_map_count", torch.tensor(0))
        self.register_buffer("inside_map_sample_count", torch.tensor(0))
        self.register_buffer("forward_action_before_bias_sum", torch.tensor(0.0))
        self.register_buffer("forward_action_after_bias_sum", torch.tensor(0.0))
        self.register_buffer("forward_action_count", torch.tensor(0))
        self.register_buffer("forward_bias_sum", torch.tensor(0.0))
        self.register_buffer("forward_bias_count", torch.tensor(0))
        # 注册持久化张量：完成的轮数
        self.register_buffer("epochs_completed", torch.tensor(0))

        # 近期评论家损失列表
        self.recent_critic_losses = []
        # 近期模型损失列表
        self.recent_model_losses = []
        # 最近一次模型拟合的详细统计，供 TensorBoard/日志复用
        self.last_model_update_stats = None
        # 模型 loss 曲线 CSV 日志，首次拟合模型时懒加载创建
        self.model_loss_curve_log = None
        # 近期回合长度列表
        self.recent_episode_lengths = []
        # 当前每个环境的回合长度（批量环境）
        self.current_episode_lengths = torch.zeros(self.n_envs, dtype=torch.long, device=device)
        # 步进生成器
        self.stepper = None
        # 调试日志开关
        self.debug_step_logs = False
        # 模型阶段是否开始
        self.model_phase_started = False
        # 上一次模型更新的步数
        self._last_model_update_t = -1
        # 前几次模型 rollout/update 默认打印定位日志，避免卡在 tqdm 0% 时没有线索。
        self._rollout_update_calls = 0

    def _sync_cuda_if_needed(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def _postprocess_warmup_actions(self, actions):
        xy_limit = float(self.warmup_xy_action_limit)
        z_limit = float(self.warmup_z_action_limit)
        if self.action_dim >= 1:
            actions[:, 0] = torch.empty_like(actions[:, 0]).uniform_(-xy_limit, xy_limit)
        if self.action_dim >= 2:
            actions[:, 1] = torch.empty_like(actions[:, 1]).uniform_(-xy_limit, xy_limit)
        if self.action_dim >= 3:
            actions[:, 2] = torch.empty_like(actions[:, 2]).uniform_(-z_limit, z_limit)
        return actions

    def _current_actor_forward_bias(self):
        start = float(self.actor_forward_bias_start)
        schedule_epochs = max(int(self.actor_forward_bias_anneal_epochs), 0)
        if start <= 0.0:
            return 0.0
        if schedule_epochs <= 0:
            return 0.0
        progress = min(float(self.epochs_completed.item()) / float(schedule_epochs), 1.0)
        return start * (1.0 - progress)

    def _postprocess_actor_env_actions(self, actions):
        before_bias = actions.clone()
        bias = self._current_actor_forward_bias()
        if self.action_dim >= 1 and bias > 0.0:
            actions[:, 0] = actions[:, 0] + bias
        if self.action_dim >= 3 and self.actor_vertical_action_limit >= 0.0:
            actions[:, 2] = actions[:, 2].clamp(
                -float(self.actor_vertical_action_limit),
                float(self.actor_vertical_action_limit),
            )
        actions = actions.clamp(-1.0, 1.0)
        return actions, before_bias, bias

    def _record_step_diagnostics(self, actions, actions_before_bias=None, forward_bias_value=0.0):
        raw_env = getattr(self.env, "env", None)
        cfg = getattr(raw_env, "cfg", None)
        xy_scale = float(getattr(cfg, "cmd_body_vel_xy_max", 1.0))
        z_scale = float(getattr(cfg, "cmd_vel_z_max", 1.0))

        actions_detached = actions.detach()
        self.action_abs_sum += actions_detached.abs().sum()
        self.action_element_count += actions_detached.numel()
        if actions_detached.shape[1] >= 1:
            self.vx_cmd_sum += (actions_detached[:, 0] * xy_scale).sum()
        if actions_detached.shape[1] >= 2:
            self.vy_cmd_sum += (actions_detached[:, 1] * xy_scale).sum()
        if actions_detached.shape[1] >= 3:
            self.vz_cmd_sum += (actions_detached[:, 2] * z_scale).sum()
        self.cmd_count += actions_detached.shape[0]
        if actions_detached.shape[1] >= 1:
            if actions_before_bias is None:
                before_forward = actions_detached[:, 0]
            else:
                before_forward = actions_before_bias.detach()[:, 0]
            self.forward_action_before_bias_sum += before_forward.sum()
            self.forward_action_after_bias_sum += actions_detached[:, 0].sum()
            self.forward_action_count += actions_detached.shape[0]
            self.forward_bias_sum += float(forward_bias_value) * actions_detached.shape[0]
            self.forward_bias_count += actions_detached.shape[0]

        robot = getattr(raw_env, "robot", None)
        if robot is not None:
            root_pos_w = robot.data.root_pos_w
            map_half_extent = float(getattr(cfg, "map_half_extent", 0.0))
            inside_map = torch.all(root_pos_w[:, :2].abs() < map_half_extent, dim=1)
            self.inside_map_count += inside_map.sum()
            self.inside_map_sample_count += inside_map.numel()

    # 提取策略输入的观测值：处理元组/字典/张量格式的观测
    def _extract_policy_obs(self, obs):
        # 如果观测是元组，取第一个元素
        if isinstance(obs, tuple):
            obs = obs[0]
        # 如果观测是字典，取policy键对应的值
        if isinstance(obs, dict):
            obs = obs["policy"]
        # 如果是张量，转移到指定设备并转为浮点型
        if torch.is_tensor(obs):
            return obs.to(device=device, dtype=torch.float)
        # 其他类型转为张量
        return torch.tensor(obs, device=device, dtype=torch.float)

    # 解析环境步进的返回值：提取下一状态、奖励、结束、违反、信息
    def _parse_env_step(self, step_out):
        # 解包环境返回值
        next_state, reward, done, info = step_out
        # 从info中获取约束违反标志，默认为False
        violation = info.get("violation", torch.zeros_like(done, dtype=torch.bool))
        # 提取并处理下一状态
        next_state = self._extract_policy_obs(next_state)
        # 奖励转移设备并转浮点
        reward = reward.to(device=device, dtype=torch.float)
        # 结束标志转移设备
        done = done.to(device=device, dtype=torch.bool)
        # 违反标志转移设备
        violation = violation.to(device=device, dtype=torch.bool)
        # 返回处理后的值
        return next_state, reward, done, violation, info

    # 演员网络属性：直接返回求解器
    @property
    def actor(self):
        return self.solver

    # 创建经验缓冲池：安全样本缓冲池，转移到指定设备
    def _create_buffer(self, capacity):
        buffer = SafetySampleBuffer(
            self.state_dim,
            self.action_dim,
            capacity,
            device=torch.device("cpu"),
        )
        # 用虚拟模块包装，方便PyTorch管理
        return DummyModuleWrapper(buffer)

    # 更新环境模型：训练模型并记录损失
    def update_models(self, model_steps):
        # 打印日志：开始训练模型
        log.message(f"Fitting models @ t = {self.steps_sampled.item()}")
        # 训练集成模型，返回损失列表
        model_losses = self.model_ensemble.fit(self.replay_buffer, steps=model_steps)
        # 保存模型损失
        self.recent_model_losses.extend(model_losses)
        # 计算前N个损失平均值
        start_loss_average = np.mean(model_losses[:LOSS_AVERAGE_WINDOW])
        # 计算后N个损失平均值
        end_loss_average = np.mean(model_losses[-LOSS_AVERAGE_WINDOW:])
        loss_array = np.asarray(model_losses, dtype=np.float64)
        loss_mean = float(np.mean(loss_array))
        loss_min = float(np.min(loss_array))
        loss_max = float(np.max(loss_array))
        loss_p50 = float(np.quantile(loss_array, 0.50))
        loss_p90 = float(np.quantile(loss_array, 0.90))
        loss_p99 = float(np.quantile(loss_array, 0.99))
        loss_delta = float(end_loss_average - start_loss_average)
        eval_stats = self._model_eval_stats()
        # 打印损失统计
        log.message("Loss statistics:")
        log.message(f"\tFirst {LOSS_AVERAGE_WINDOW}: {start_loss_average}")
        log.message(f"\tLast {LOSS_AVERAGE_WINDOW}: {end_loss_average}")
        log.message(f"\tDeciles: {deciles(model_losses)}")
        log.message(
            "Model loss curve: "
            f"mean={loss_mean:.4f} p50={loss_p50:.4f} p90={loss_p90:.4f} p99={loss_p99:.4f} "
            f"min={loss_min:.4f} max={loss_max:.4f} delta_last-first={loss_delta:+.4f} "
            f"eval_state_mse={eval_stats['state_mse']:.6f} eval_reward_mse={eval_stats['reward_mse']:.6f} "
            f"eval_nll_p90={eval_stats['nll_p90']:.4f} eval_nll_max={eval_stats['nll_max']:.4f} "
            f"terminal_batch_rate={eval_stats['terminal_rate']:.4f} "
            f"log_var_gap_mean={eval_stats['log_var_gap_mean']:.4f} "
            f"log_var_gap_min={eval_stats['log_var_gap_min']:.4f}"
        )
        self._write_model_loss_curve(
            model_steps=model_steps,
            start_loss_average=float(start_loss_average),
            end_loss_average=float(end_loss_average),
            loss_mean=loss_mean,
            loss_min=loss_min,
            loss_max=loss_max,
            loss_p50=loss_p50,
            loss_p90=loss_p90,
            loss_p99=loss_p99,
            loss_delta=loss_delta,
            eval_stats=eval_stats,
        )
        self.last_model_update_stats = {
            "model_steps": int(model_steps),
            "start_loss_average": float(start_loss_average),
            "end_loss_average": float(end_loss_average),
            "loss_mean": float(loss_mean),
            "loss_min": float(loss_min),
            "loss_max": float(loss_max),
            "loss_p50": float(loss_p50),
            "loss_p90": float(loss_p90),
            "loss_p99": float(loss_p99),
            "loss_delta": float(loss_delta),
            **eval_stats,
            "min_log_var_param_mean": float(self.model_ensemble.min_log_var.mean().item()),
            "max_log_var_param_mean": float(self.model_ensemble.max_log_var.mean().item()),
        }

        # 获取真实缓冲池中的奖励
        buffer_rewards = self.replay_buffer.get("rewards", device=None)
        # 奖励最小值
        r_min = buffer_rewards.min().item()
        # 奖励最大值
        r_max = buffer_rewards.max().item()
        # 更新策略的奖励范围
        self.solver.update_r_bounds(r_min, r_max)

    def _model_eval_stats(self, batch_size=8192):
        n_samples = min(int(batch_size), len(self.replay_buffer))
        if n_samples <= 0:
            return {
                "state_mse": 0.0,
                "reward_mse": 0.0,
                "nll_mean": 0.0,
                "nll_p50": 0.0,
                "nll_p90": 0.0,
                "nll_p99": 0.0,
                "nll_max": 0.0,
                "terminal_rate": 0.0,
                "violation_rate": 0.0,
                "log_var_mean": 0.0,
                "log_var_min": 0.0,
                "log_var_max": 0.0,
                "log_var_gap_mean": 0.0,
                "log_var_gap_min": 0.0,
                "log_var_gap_max": 0.0,
                "nll_over_cap_rate": 0.0,
            }

        samples = self.replay_buffer.sample(n_samples, device=device)
        states, actions, next_states, rewards, dones, violations = samples[:6]
        targets = torch.cat([next_states, rewards.unsqueeze(1)], dim=1)
        with torch.no_grad():
            pred_states, pred_rewards = self.model_ensemble.means(states, actions)
            pred_state_mean = pred_states.mean(dim=0)
            pred_reward_mean = pred_rewards.mean(dim=0)
            state_mse = torch.mean((pred_state_mean - next_states) ** 2)
            reward_mse = torch.mean((pred_reward_mean - rewards) ** 2)

            repeated_states = states.repeat(self.model_cfg.ensemble_size, 1, 1)
            repeated_actions = actions.repeat(self.model_cfg.ensemble_size, 1, 1)
            means, log_vars = self.model_ensemble._forward_all(repeated_states, repeated_actions)
            repeated_targets = targets.unsqueeze(0).expand_as(means)
            nll = torch.sum((repeated_targets - means) ** 2 * torch.exp(-log_vars), dim=-1) + torch.sum(log_vars, dim=-1)
            nll_flat = nll.reshape(-1)
            terminal_mask = (dones | violations).to(dtype=torch.float)
            nll_cap = float(getattr(self.model_cfg, "nll_sample_clip", 0.0) or 0.0)
            if nll_cap > 0.0:
                nll_over_cap_rate = (nll_flat > nll_cap).to(dtype=torch.float).mean()
            else:
                nll_over_cap_rate = torch.zeros((), device=device)
            log_var_gap = (self.model_ensemble.max_log_var - self.model_ensemble.min_log_var)

        return {
            "state_mse": float(state_mse.item()),
            "reward_mse": float(reward_mse.item()),
            "nll_mean": float(nll_flat.mean().item()),
            "nll_p50": float(torch.quantile(nll_flat, 0.50).item()),
            "nll_p90": float(torch.quantile(nll_flat, 0.90).item()),
            "nll_p99": float(torch.quantile(nll_flat, 0.99).item()),
            "nll_max": float(nll_flat.max().item()),
            "terminal_rate": float(terminal_mask.mean().item()),
            "violation_rate": float(violations.to(dtype=torch.float).mean().item()),
            "log_var_mean": float(log_vars.mean().item()),
            "log_var_min": float(log_vars.min().item()),
            "log_var_max": float(log_vars.max().item()),
            "log_var_gap_mean": float(log_var_gap.mean().item()),
            "log_var_gap_min": float(log_var_gap.min().item()),
            "log_var_gap_max": float(log_var_gap.max().item()),
            "nll_over_cap_rate": float(nll_over_cap_rate.item()),
        }

    def _write_model_loss_curve(self, **row):
        if log.dir is None:
            return
        if self.model_loss_curve_log is None:
            self.model_loss_curve_log = TabularLog(log.dir, "model_loss_curve.csv")
        eval_stats = row.pop("eval_stats")
        flat_row = {
            "epoch": int(self.epochs_completed.item()) + 1,
            "steps": int(self.steps_sampled.item()),
            "buffer": len(self.replay_buffer),
            "phase": "warmup" if self.in_warmup() else "train",
            **row,
            **eval_stats,
            "min_log_var_param_mean": float(self.model_ensemble.min_log_var.mean().item()),
            "max_log_var_param_mean": float(self.model_ensemble.max_log_var.mean().item()),
        }
        self.model_loss_curve_log.row(flat_row)

    # 重置虚拟缓冲池
    def _reset_virtual_buffer(self):
        self.virt_buffer = self._create_buffer(self.virt_buffer_max)

    # 判断模型是否准备就绪：轮数达标 或 步数达标
    def _model_ready(self):
        epoch_ready = int(self.epochs_completed.item()) >= int(self.model_start_epoch)
        steps_ready = int(self.model_start_steps) > 0 and int(self.steps_sampled.item()) >= int(self.model_start_steps)
        exploration_finished = not self.in_warmup()
        return exploration_finished and (epoch_ready or steps_ready)

    # 尝试启动模型阶段：满足条件则初始化模型和虚拟缓冲池
    def _maybe_start_model_phase(self):
        # 已启动则直接返回
        if self.model_phase_started:
            return
        # 未准备好则返回
        if not self._model_ready():
            return
        # 打印日志：启动模型阶段
        log.message(
            f"Starting model phase @ epoch={int(self.epochs_completed.item())} "
            f"t={int(self.steps_sampled.item())}"
        )
        # 初始训练模型
        self.update_models(self.model_initial_steps)
        # 重置虚拟缓冲池
        self._reset_virtual_buffer()
        # 标记模型阶段已启动
        self.model_phase_started = True
        # 记录模型更新步数
        self._last_model_update_t = int(self.steps_sampled.item())

    def _should_update_model_during_training(self):
        # warmup-only 模式下，动力学模型只在 warmup 结束时拟合一次，
        # 训练阶段继续采真实数据，但不再用新数据重拟合模型。
        return not bool(self.warmup_only_model_fit)

    def in_warmup(self):
        return int(self.epochs_completed.item()) < int(self.warmup_epochs)

    def warmup_progress(self):
        current_epochs = min(int(self.epochs_completed.item()), int(self.warmup_epochs))
        target_epochs = int(self.warmup_epochs)
        ratio = 1.0 if target_epochs <= 0 else min(current_epochs / float(target_epochs), 1.0)
        return current_epochs, target_epochs, ratio

    def _activate_pretrained_model_phase(self):
        if self.model_phase_started:
            return
        log.message(
            f"Warmup finished; activating pretrained model phase @ "
            f"epoch={int(self.epochs_completed.item())} t={int(self.steps_sampled.item())}"
        )
        self._reset_virtual_buffer()
        self.model_phase_started = True
        self._last_model_update_t = int(self.steps_sampled.item())

    # 模型推演：用环境模型生成虚拟经验
    def rollout(self, policy, initial_states=None):
        # 未指定初始状态，则从真实缓冲池随机采样
        if initial_states is None:
            initial_states = self.replay_buffer.sample(self.rollout_batch_size, device=device)[0]
        # 创建临时缓冲池，存储本次推演数据
        buffer = self._create_buffer(self.rollout_batch_size * self.horizon)
        # 当前状态 = 初始状态
        states = initial_states
        # 按推演长度循环
        for rollout_step in range(self.horizon):
            # 调试日志
            if self.debug_step_logs:
                log.message(f"[debug] rollout step {rollout_step} start: batch={len(states)}")
            # 禁用梯度计算
            with torch.no_grad():
                # 策略生成动作
                actions = policy.act(states, eval=False)
                # 模型预测下一状态和奖励
                next_states, rewards = self.model_ensemble.sample(states, actions)
            # 检查是否结束
            dones = self.check_done(next_states)
            # 检查是否违反约束
            violations = self.check_violation(next_states)
            # 将数据存入临时缓冲池
            buffer.extend(
                states=states,
                actions=actions,
                next_states=next_states,
                rewards=rewards,
                dones=dones,
                violations=violations,
            )
            # 未结束且未违反的样本，继续推演
            continues = ~(dones | violations)
            # 所有样本都结束，跳出循环
            if continues.sum() == 0:
                break
            # 更新状态为未结束的样本
            states = next_states[continues]

        # 将推演数据存入虚拟缓冲池
        self.virt_buffer.extend(**buffer.get(as_dict=True, device=None))
        # 返回临时缓冲池
        return buffer

    # 更新策略求解器：混合真实+虚拟数据训练
    def current_real_fraction(self):
        schedule_epochs = max(int(self.real_fraction_schedule_epochs), 0)
        start = float(self.real_fraction)
        end = float(self.real_fraction_final)
        if schedule_epochs <= 0:
            return end
        progress = min(float(self.epochs_completed.item()) / float(schedule_epochs), 1.0)
        return start + (end - start) * progress

    def update_solver(self, update_actor=True, use_virtual=True):
        solver = self.solver
        # 判断是否使用虚拟数据
        use_virtual = bool(use_virtual and len(self.virt_buffer) > 0)
        if use_virtual:
            real_fraction = self.current_real_fraction()
            # 真实数据数量
            n_real = int(real_fraction * solver.batch_size)
            # 虚拟数据数量
            n_virt = solver.batch_size - n_real
        else:
            # 只用真实数据
            n_real = solver.batch_size
            n_virt = 0
        # 调试日志
        if self.debug_step_logs:
            log.message(
                f"[debug] solver sample start: batch={solver.batch_size} real={n_real} virt={n_virt}"
            )
        # 采样真实数据
        real_samples = self.replay_buffer.sample(n_real)
        if n_virt > 0:
            # 采样虚拟数据
            virt_samples = self.virt_buffer.sample(n_virt)
            # 拼接真实+虚拟数据
            combined_samples = [torch.cat([real, virt]) for real, virt in zip(real_samples, virt_samples)]
        else:
            # 只用真实数据
            combined_samples = real_samples
        # 更新评论家网络，返回损失
        critic_loss = solver.update_critic(*combined_samples)
        # 保存评论家损失
        self.recent_critic_losses.append(float(critic_loss.item()))
        # 如果需要，更新演员网络
        if update_actor:
            solver.update_actor_and_alpha(combined_samples[0])

    # 推演+更新：模型生成数据 + 训练策略
    def rollout_and_update(self):
        trace_this_call = self.debug_step_logs or self._rollout_update_calls < 3
        try:
            # 调试日志
            if trace_this_call:
                log.message(
                    f"[trace] rollout/update call {self._rollout_update_calls + 1} start: "
                    f"replay={len(self.replay_buffer)} virt={len(self.virt_buffer)}",
                    flush=True,
                )
            # 模型推演生成数据
            self.rollout(self.actor)
            if trace_this_call:
                self._sync_cuda_if_needed()
                log.message(
                    f"[trace] rollout done: replay={len(self.replay_buffer)} virt={len(self.virt_buffer)}",
                    flush=True,
                )
            # 多次更新策略
            for update_idx in range(self.solver_updates_per_step):
                if trace_this_call:
                    log.message(
                        f"[trace] solver update {update_idx + 1}/{self.solver_updates_per_step} start",
                        flush=True,
                    )
                self.update_solver()
                if trace_this_call:
                    self._sync_cuda_if_needed()
                    log.message(
                        f"[trace] solver update {update_idx + 1}/{self.solver_updates_per_step} done",
                        flush=True,
                    )
        except Exception:
            log.message("Exception inside rollout_and_update:", flush=True)
            log.message(traceback.format_exc(), timestamp=False, flush=True)
            raise
        finally:
            self._rollout_update_calls += 1

    # 仅用真实数据更新策略
    def real_only_update(self):
        for _ in range(self.solver_updates_per_step):
            self.update_solver(use_virtual=False)

    # 步进生成器：核心训练循环，无限生成环境交互步数
    def step_generator(self):
        # 重置环境，获取初始状态
        states = self._extract_policy_obs(self.env.reset())
        # 记录每个环境是否违反约束
        episode_has_violation = torch.zeros(self.n_envs, dtype=torch.bool, device=device)
        # 预热阶段为每个并行环境维护“当前 episode 使用的固定高斯动作”
        warmup_actions = torch.zeros(self.n_envs, self.action_dim, dtype=torch.float, device=device)
        warmup_initialized = torch.zeros(self.n_envs, dtype=torch.bool, device=device)
        # actor 真实采样阶段也低频保持动作，避免每帧高斯采样互相抵消成原地抖动。
        held_policy_actions = torch.zeros(self.n_envs, self.action_dim, dtype=torch.float, device=device)
        held_policy_action_age = torch.full(
            (self.n_envs,),
            int(self.actor_action_hold_steps),
            dtype=torch.long,
            device=device,
        )

        # 无限循环
        while True:
            actions_before_bias = None
            forward_bias_value = 0.0
            # 当前总步数
            t = int(self.steps_sampled.item())
            # 达到策略训练步数，使用训练好的策略
            if not self.in_warmup():
                policy = self.actor
                # 缓冲池数据足够
                if t >= int(self.buffer_min):
                    # 尝试启动模型阶段
                    self._maybe_start_model_phase()
                    # 模型阶段已启动
                    if self.model_phase_started:
                        # 达到模型更新周期
                        if (
                            self._should_update_model_during_training()
                            and (
                                self._last_model_update_t < 0
                                or (t - self._last_model_update_t) >= int(self.model_update_period)
                            )
                        ):
                            if self.debug_step_logs:
                                log.message(f"[debug] model update start @ t={t}")
                            # 更新模型
                            self.update_models(self.model_steps)
                            # 重置虚拟缓冲池
                            self._reset_virtual_buffer()
                            # 更新模型更新时间
                            self._last_model_update_t = t
                        # 模型推演 + 策略更新
                        self.rollout_and_update()
                    else:
                        # 未启动模型阶段，仅用真实数据更新
                        self.real_only_update()
                else:
                    # 数据不足，仅用真实数据更新
                    self.real_only_update()
            else:
                # 预热阶段：每个环境在一个 episode 开始时采样一次高斯动作，
                # 之后整局保持该方向，直到死亡/超时再为新 episode 重新采样。
                resample_mask = ~warmup_initialized
                if resample_mask.any():
                    sampled_actions = self.warmup_policy.act(states[resample_mask], eval=False)
                    sampled_actions = self._postprocess_warmup_actions(sampled_actions)
                    warmup_actions[resample_mask] = sampled_actions
                    warmup_initialized[resample_mask] = True
                actions = warmup_actions.clone()
                policy = None

            # 策略生成动作
            if policy is not None:
                hold_steps = max(int(self.actor_action_hold_steps), 1)
                resample_policy_mask = held_policy_action_age >= hold_steps
                if resample_policy_mask.any():
                    sampled_policy_actions = policy.act(states[resample_policy_mask], eval=False)
                    held_policy_actions[resample_policy_mask] = sampled_policy_actions
                    held_policy_action_age[resample_policy_mask] = 0
                actions = held_policy_actions.clone()
                held_policy_action_age += 1
                actions, actions_before_bias, forward_bias_value = self._postprocess_actor_env_actions(actions)
            # 环境步进，解析返回值
            next_states, rewards, dones, violations, info = self._parse_env_step(self.env.step(actions))
            self._record_step_diagnostics(actions, actions_before_bias, forward_bias_value)

            # 将真实经验存入缓冲池
            self.replay_buffer.extend(
                states=states,
                actions=actions,
                next_states=next_states,
                rewards=rewards,
                dones=dones,
                violations=violations,
            )
            # 总步数+1
            self.steps_sampled += len(states)
            # 累计奖励
            self.env_reward_sum += rewards.sum()
            # 奖励计数
            self.env_reward_count += len(rewards)
            # 当前回合长度+1
            self.current_episode_lengths += 1

            # 更新约束违反标志：只要违反过就为True
            episode_has_violation = episode_has_violation | violations
            # 获取环境截断标志（超时）
            truncated = info.get("truncated", torch.zeros_like(dones, dtype=torch.bool))
            truncated = truncated.to(device=device, dtype=torch.bool)
            # 结束掩码：结束/违反/超时
            done_mask = dones | violations | truncated
            # 存在结束的环境
            if done_mask.any():
                # 获取目标达成标志
                target_reached = info.get("target_reached", torch.zeros_like(done_mask, dtype=torch.bool))
                target_reached = target_reached.to(device=device, dtype=torch.bool)
                # 获取障碍物碰撞标志
                obstacle_collision = info.get("obstacle_collision", torch.zeros_like(done_mask, dtype=torch.bool))
                obstacle_collision = obstacle_collision.to(device=device, dtype=torch.bool)
                # 获取高度越界标志
                too_low = info.get("too_low", torch.zeros_like(done_mask, dtype=torch.bool))
                too_low = too_low.to(device=device, dtype=torch.bool)
                too_high = info.get("too_high", torch.zeros_like(done_mask, dtype=torch.bool))
                too_high = too_high.to(device=device, dtype=torch.bool)
                out_of_bounds = info.get("out_of_bounds", torch.zeros_like(done_mask, dtype=torch.bool))
                out_of_bounds = out_of_bounds.to(device=device, dtype=torch.bool)
                # 已结束环境在下一个 episode 重新采样 warmup 高斯动作
                warmup_initialized = torch.where(
                    done_mask,
                    torch.zeros_like(warmup_initialized),
                    warmup_initialized,
                )
                held_policy_action_age = torch.where(
                    done_mask,
                    torch.full_like(held_policy_action_age, int(self.actor_action_hold_steps)),
                    held_policy_action_age,
                )

                # 累计回合数
                self.episodes_sampled += done_mask.sum()
                # 累计成功次数
                self.n_successes += target_reached[done_mask].sum()
                # 累计违反次数
                self.n_violations += episode_has_violation[done_mask].sum()
                # 累计碰撞次数
                self.n_collisions += obstacle_collision[done_mask].sum()
                # 累计超时次数
                self.n_timeouts += truncated[done_mask].sum()
                # 累计高度越界次数
                self.n_too_low += too_low[done_mask].sum()
                self.n_too_high += too_high[done_mask].sum()
                # 累计水平越界次数
                self.n_out_of_bounds += out_of_bounds[done_mask].sum()
                # 保存结束回合的长度
                self.recent_episode_lengths.extend(self.current_episode_lengths[done_mask].tolist())

                # 重置结束的环境
                reset_states = self._extract_policy_obs(self.env.reset_done(done_mask))
                # 替换结束环境的状态为重置后的状态
                next_states = torch.where(done_mask.unsqueeze(1), reset_states, next_states)
                # 重置结束环境的违反标志
                episode_has_violation = torch.where(done_mask, torch.zeros_like(episode_has_violation), episode_has_violation)
                # 重置结束环境的回合长度
                self.current_episode_lengths = torch.where(
                    done_mask,
                    torch.zeros_like(self.current_episode_lengths),
                    self.current_episode_lengths,
                )

            # 更新当前状态
            states = next_states
            # 生成器返回当前步数
            yield t

    # 初始化设置：预热收集数据
    def setup(self):
        # 初始化步进生成器
        self.stepper = self.step_generator()
        log.message(
            f"Warmup configured for {int(self.warmup_epochs)} epochs "
            f"(reference solver_start_steps={int(self.solver_start_steps)})"
        )

    # 重新开始采样：重置生成器
    def restart_sampling(self):
        self.current_episode_lengths.zero_()
        self.stepper = self.step_generator()

    # 执行一轮迭代：固定步数
    def epoch(self, on_step=None):
        was_in_warmup = self.in_warmup()
        for _ in trange(self.steps_per_epoch):
            next(self.stepper)
            if on_step is not None:
                on_step()
        if was_in_warmup and len(self.replay_buffer) >= int(self.buffer_min):
            fit_steps = int(self.model_initial_steps) if self._last_model_update_t < 0 else int(self.model_steps)
            log.message(
                f"Warmup epoch-end model pretraining @ epoch={int(self.epochs_completed.item()) + 1} "
                f"(buffer={len(self.replay_buffer)}, fit_steps={fit_steps})"
            )
            self.update_models(fit_steps)
            self._last_model_update_t = int(self.steps_sampled.item())
        # 完成轮数+1
        self.epochs_completed += 1
        if was_in_warmup and not self.in_warmup():
            log.message("Warmup finished; resetting all environments before policy/model training.")
            self.restart_sampling()
            self._activate_pretrained_model_phase()

    # 计算近期评论家平均损失
    def average_recent_critic_loss(self):
        if not self.recent_critic_losses:
            return None
        return pythonic_mean(self.recent_critic_losses)

    # 计算成功率
    def success_rate(self):
        episodes = int(self.episodes_sampled.item())
        if episodes == 0:
            return 0.0
        return float(self.n_successes.item()) / float(episodes)

    # 计算约束违反率
    def violation_rate(self):
        episodes = int(self.episodes_sampled.item())
        if episodes == 0:
            return 0.0
        return float(self.n_violations.item()) / float(episodes)

    # 计算碰撞率
    def collision_rate(self):
        episodes = int(self.episodes_sampled.item())
        if episodes == 0:
            return 0.0
        return float(self.n_collisions.item()) / float(episodes)

    def out_of_bounds_rate(self):
        episodes = int(self.episodes_sampled.item())
        if episodes == 0:
            return 0.0
        return float(self.n_out_of_bounds.item()) / float(episodes)

    # 计算超时率
    def timeout_rate(self):
        episodes = int(self.episodes_sampled.item())
        if episodes == 0:
            return 0.0
        return float(self.n_timeouts.item()) / float(episodes)

    # 计算近期模型平均损失
    def average_recent_model_loss(self):
        if not self.recent_model_losses:
            return None
        return pythonic_mean(self.recent_model_losses)

    # 计算平均回合长度
    def average_episode_length(self):
        if not self.recent_episode_lengths:
            return None
        return pythonic_mean(self.recent_episode_lengths)

    # 评估环境模型：计算预测误差
    def evaluate_models(self):
        if len(self.replay_buffer) == 0:
            return
        # 获取真实数据
        sample_size = min(len(self.replay_buffer), 32768)
        if sample_size <= 0:
            return
        states, actions, next_states = self.replay_buffer.sample(sample_size, device=device)[:3]
        # 状态标准差，防止除零
        state_std = states.std(dim=0).clamp_min(1e-6)
        with torch.no_grad():
            # 模型预测下一状态
            predicted_states = self.model_ensemble.means(states, actions)[0]
        # 遍历每个模型
        for i in range(self.model_cfg.ensemble_size):
            # 计算归一化误差
            errors = torch.norm((predicted_states[i] - next_states) / state_std, dim=1)
            # 打印误差分位数
            log.message(f"Model {i + 1} error deciles: {deciles(errors)}")

    # 打印训练统计信息
    def log_statistics(self):
        # 评估模型
        self.evaluate_models()
        # 打印评论家损失
        avg_critic_loss = self.average_recent_critic_loss()
        if avg_critic_loss is not None:
            log.message(f"Average recent critic loss: {avg_critic_loss}")
        self.recent_critic_losses.clear()
        # 打印模型损失
        avg_model_loss = self.average_recent_model_loss()
        if avg_model_loss is not None:
            log.message(f"Average recent model loss: {avg_model_loss}")
        self.recent_model_losses.clear()
        # 打印缓冲池大小
        log.message("Buffer sizes:")
        log.message(f"\tReal: {len(self.replay_buffer)}")
        log.message(f"\tVirtual: {len(self.virt_buffer)}")
