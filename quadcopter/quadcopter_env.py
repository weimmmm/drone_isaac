# Copyright (c) 2026 Alex Jauregui & Erik Eguskiza.
# All rights reserved.

# 导入未来的类型注解支持，允许在类内部引用自身类型
from __future__ import annotations

# 导入强化学习环境基础库
import gymnasium as gym
# 导入PyTorch用于张量计算
import torch

# 导入IsaacLab核心模块
import isaaclab.sim as sim_utils  # 仿真相关工具
from isaaclab.assets import Articulation, ArticulationCfg  # 关节机器人资产配置
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg  # 直接强化学习环境基类
from isaaclab.envs.ui import BaseEnvWindow  # UI窗口基类
from isaaclab.markers import VisualizationMarkers  # 可视化标记工具
from isaaclab.scene import InteractiveSceneCfg  # 交互式场景配置
from isaaclab.sim import SimulationCfg  # 仿真配置
from isaaclab.terrains import TerrainImporterCfg  # 地形导入配置
from isaaclab.utils import configclass  # 配置类装饰器
from isaaclab.utils.math import quat_apply_inverse

##
# 预定义配置
##
from isaaclab_assets import CRAZYFLIE_CFG  # Crazyflie无人机配置
from isaaclab.markers import CUBOID_MARKER_CFG  # 立方体标记配置


def _quat_to_yaw(quat_w: torch.Tensor) -> torch.Tensor:
    # Isaac Lab stores quaternions in (w, x, y, z) order.
    w, x, y, z = quat_w[:, 0], quat_w[:, 1], quat_w[:, 2], quat_w[:, 3]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(siny_cosp, cosy_cosp)


def _wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


class QuadcopterEnvWindow(BaseEnvWindow):
    """无人机环境的自定义UI窗口管理器
    继承自BaseEnvWindow，用于扩展环境的可视化UI功能
    """

    def __init__(self, env: QuadcopterEnv, window_name: str = "IsaacLab"):
        """初始化UI窗口

        Args:
            env: 无人机环境实例
            window_name: 窗口名称，默认为"IsaacLab"
        """
        # 调用父类初始化方法
        super().__init__(env, window_name)
        # 添加自定义UI元素
        with self.ui_window_elements["main_vstack"]:  # 主垂直布局
            with self.ui_window_elements["debug_frame"]:  # 调试框架
                with self.ui_window_elements["debug_vstack"]:  # 调试垂直布局
                    # 创建调试可视化UI元素，用于显示目标位置
                    self._create_debug_vis_ui_element("targets", self.env)


@configclass  # 装饰器，将类转换为配置类，支持序列化/反序列化
class QuadcopterEnvCfg(DirectRLEnvCfg):
    """无人机强化学习环境的配置类
    继承自DirectRLEnvCfg，包含环境、仿真、场景、奖励等所有配置参数
    """
    # 环境配置
    episode_length_s = 10.0  # 每个episode的时长（秒）
    decimation = 2  # 动作执行频率 = 仿真频率 / decimation (100/2=50Hz)
    action_space = 4  # 动作空间维度：1个总推力 + 3个力矩
    observation_space = 12  # 观测空间总维度（12个有效特征）
    state_space = 0  # 状态空间维度（未使用）
    debug_vis = True  # 是否启用调试可视化

    ui_window_class_type = QuadcopterEnvWindow  # 自定义UI窗口类

    # 仿真配置
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,  # 仿真步长（100Hz）
        render_interval=decimation,  # 渲染间隔（每2步渲染一次）
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",  # 摩擦组合模式：相乘
            restitution_combine_mode="multiply",  # 恢复系数组合模式：相乘
            static_friction=1.0,  # 静摩擦系数
            dynamic_friction=1.0,  # 动摩擦系数
            restitution=0.0,  # 恢复系数（无弹性）
        ),
    )
    # 地形配置
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",  # 地形在USD中的路径
        terrain_type="plane",  # 地形类型：平面
        collision_group=-1,  # 碰撞组（-1表示所有组）
        physics_material=sim_utils.RigidBodyMaterialCfg(  # 地形物理材质
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,  # 地形调试可视化关闭
    )

    # 场景配置
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,  # 并行环境数量（大规模批量训练）
        env_spacing=2.5,  # 环境之间的间距（米）
        replicate_physics=True,  # 复制物理参数
        clone_in_fabric=True,  # 在Fabric中克隆环境（加速仿真）
    )

    # 机器人配置
    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")  # 替换无人机的USD路径
    thrust_to_weight = 1.9  # 推力重量比（最大推力是重量的1.9倍）
    moment_scale = 0.01  # 力矩缩放系数

    # 指令配置
    cmd_lin_vel_xy_range = (-2.0, 2.0)  # 世界系x/y方向目标线速度范围（m/s）
    cmd_lin_vel_z_range = (-0.5, 0.5)  # 世界系z方向目标线速度范围（m/s）
    cmd_resample_time_range = (2.0, 4.0)  # 指令重采样时间范围（s）
    cmd_transition_tau = 0.35  # 指令平滑时间常数（s）
    cmd_zero_prob = 0.25  # 采样到悬停命令的概率
    max_distance_from_origin = 6.0  # 超过该水平距离则终止，避免串环境

    # 奖励系数配置
    lin_vel_tracking_reward_scale = 2.0  # 跟踪目标线速度奖励系数（对齐Go2风格）
    yaw_tracking_reward_scale = 1.0  # 跟踪速度方向对应偏航角奖励系数（对齐Go2风格）
    lin_vel_tracking_exp_scale = 4.0  # 线速度误差指数缩放
    yaw_tracking_exp_scale = 4.0  # 偏航误差指数缩放
    alive_reward_scale = 0.5  # 存活奖励
    ang_vel_penalty_scale = -0.02  # 角速度惩罚系数
    action_rate_penalty_scale = -0.01  # 动作变化惩罚系数
    vertical_speed_penalty_scale = -0.5  # 垂向速度惩罚
    position_drift_penalty_scale = -0.05  # 水平漂移惩罚
    flat_orientation_penalty_scale = -0.3  # 姿态倾斜惩罚
    termination_penalty_scale = -5.0  # 提前终止惩罚


class QuadcopterEnv(DirectRLEnv):
    """无人机强化学习环境类
    继承自DirectRLEnv，实现了无人机的核心逻辑：动作处理、观测、奖励、重置等
    """
    cfg: QuadcopterEnvCfg  # 类型注解：cfg是QuadcopterEnvCfg类型

    def __init__(self, cfg: QuadcopterEnvCfg, render_mode: str | None = None, **kwargs):
        """初始化环境

        Args:
            cfg: 环境配置实例
            render_mode: 渲染模式（None/人类可视化/rgb_array等）
            **kwargs: 其他参数
        """
        # 调用父类初始化方法
        super().__init__(cfg, render_mode, **kwargs)

        # ========== 初始化张量 ==========
        # 动作张量：[环境数, 动作维度(4)]，存储当前步的动作
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        # 推力张量：[环境数, 1, 3]，存储施加到无人机的推力（仅z轴）
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        # 力矩张量：[环境数, 1, 3]，存储施加到无人机的力矩（x/y/z轴）
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)
        self._applied_actions = torch.zeros_like(self._actions)
        self._desired_lin_vel_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._desired_lin_vel_target_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._cmd_resample_interval = torch.zeros(self.num_envs, device=self.device)
        self._cmd_resample_elapsed = torch.zeros(self.num_envs, device=self.device)

        # ========== 日志初始化 ==========
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "lin_vel",      # 线速度奖励累计
                "yaw",          # 偏航奖励累计
                "alive",        # 存活奖励累计
                "ang_vel",      # 角速度惩罚累计
                "action_rate",  # 动作变化惩罚累计
                "vertical_vel",  # 垂向速度惩罚累计
                "drift",        # 位置漂移惩罚累计
                "flat_orientation",  # 姿态倾斜惩罚累计
                "lin_vel_error",  # 速度跟踪误差累计
                "yaw_error",  # 偏航跟踪误差累计
                "lin_vel_score",  # 线速度跟踪得分累计
                "yaw_score",  # 偏航跟踪得分累计
            ]
        }

        # ========== 机器人参数初始化 ==========
        # 获取无人机主体的body ID
        self._body_id = self._robot.find_bodies("body")[0]
        # 计算机器人总质量（根节点所有部分的质量和）
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        # 计算重力加速度的大小
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        # 计算机器人重量（质量×重力加速度）
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        # ========== 调试可视化 ==========
        # 设置调试可视化（首次调用会创建标记器）
        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        """设置仿真场景（IsaacLab核心方法）
        负责创建机器人、地形、灯光等场景元素
        """
        # 创建无人机关节机器人实例
        self._robot = Articulation(self.cfg.robot)
        # 将机器人添加到场景的关节机器人字典中
        self.scene.articulations["robot"] = self._robot

        # 配置地形的环境数量和间距（与场景配置一致）
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        # 创建地形实例
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        
        # 克隆并复制环境（批量创建多个环境）
        self.scene.clone_environments(copy_from_source=False)
        
        # CPU仿真时需要显式过滤碰撞（GPU不需要）
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        
        # 添加环境灯光
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))  # 穹顶光配置
        light_cfg.func("/World/Light", light_cfg)  # 创建灯光

    def _pre_physics_step(self, actions: torch.Tensor):
        """物理步前处理（IsaacLab核心方法）
        将模型输出的动作转换为无人机的推力和力矩

        Args:
            actions: 模型输出的动作张量 [环境数, 4]
        """
        # 克隆并裁剪动作到[-1, 1]范围（防止动作值超出范围）
        self._actions = actions.clone().clamp(-1.0, 1.0)
        
        self._cmd_resample_elapsed += self.step_dt
        self._resample_commands(self._cmd_resample_elapsed >= self._cmd_resample_interval)
        cmd_alpha = min(1.0, self.step_dt / max(1.0e-6, self.cfg.cmd_transition_tau))
        self._desired_lin_vel_w = (1.0 - cmd_alpha) * self._desired_lin_vel_w + cmd_alpha * self._desired_lin_vel_target_w

        # Low-pass filter actions to reduce high-frequency thrust/torque spikes.
        self._applied_actions = 0.75 * self._actions + 0.25 * self._previous_actions

        # ========== 计算推力 ==========
        # 动作第0维：总推力控制（范围[-1,1] → [0, 1.9×重量]）
        # (action+1)/2 将动作映射到[0,1]，乘以推力重量比×重量得到实际推力
        self._thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * (self._applied_actions[:, 0] + 1.0) / 2.0
        
        # ========== 计算力矩 ==========
        # 动作第1-3维：x/y/z轴力矩控制，乘以缩放系数
        self._moment[:, 0, :] = self.cfg.moment_scale * self._applied_actions[:, 1:]

    def _apply_action(self):
        """应用动作到仿真（IsaacLab核心方法）
        将计算好的推力和力矩施加到无人机主体
        """
        # 给无人机主体施加外部力（推力）和力矩
        self._robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)

    def _get_observations(self) -> dict:
        """获取观测（IsaacLab核心方法）
        构建智能体的观测空间，返回字典格式

        Returns:
            dict: 包含policy观测的字典
        """
        desired_lin_vel_b = quat_apply_inverse(self._robot.data.root_quat_w, self._desired_lin_vel_w)

        # ========== 构建基础观测 ==========
        base_obs = torch.cat(
            [
                self._robot.data.root_lin_vel_b,      # 机体坐标系下线速度 (3维)
                self._robot.data.root_ang_vel_b,      # 机体坐标系下角速度 (3维)
                self._robot.data.projected_gravity_b, # 机体坐标系下的重力向量 (3维)
                desired_lin_vel_b,                    # 机体系目标线速度 (3维)
            ],
            dim=-1,
        )  # 基础观测总维度：12维

        # 返回观测字典（policy键是强化学习策略使用的观测）
        observations = {"policy": base_obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        """计算奖励（IsaacLab核心方法）
        基于当前环境状态计算每个环境的奖励值

        Returns:
            torch.Tensor: 每个环境的奖励值 [环境数,]
        """
        # ========== 计算奖励项 ==========
        desired_lin_vel_b = quat_apply_inverse(self._robot.data.root_quat_w, self._desired_lin_vel_w)
        lin_vel_error = torch.sum(torch.square(desired_lin_vel_b - self._robot.data.root_lin_vel_b), dim=1)
        lin_vel_tracking = torch.exp(-self.cfg.lin_vel_tracking_exp_scale * lin_vel_error)

        actual_horiz_vel = self._robot.data.root_lin_vel_w[:, :2]
        actual_horiz_speed = torch.linalg.norm(actual_horiz_vel, dim=1)
        desired_yaw_w = torch.atan2(actual_horiz_vel[:, 1], actual_horiz_vel[:, 0])
        yaw_error = _wrap_to_pi(desired_yaw_w - _quat_to_yaw(self._robot.data.root_quat_w))
        yaw_active = (actual_horiz_speed > 0.1).float()
        yaw_tracking = torch.exp(-self.cfg.yaw_tracking_exp_scale * torch.square(yaw_error))

        alive = torch.ones(self.num_envs, device=self.device)
        ang_vel = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
        vertical_vel = torch.square(self._robot.data.root_lin_vel_b[:, 2])
        drift = torch.linalg.norm(
            self._robot.data.root_pos_w[:, :2] - self._terrain.env_origins[:, :2], dim=1
        )
        flat_orientation = torch.sum(torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1)
        termination = self.reset_terminated.float()
        lin_vel_error_norm = torch.linalg.norm(desired_lin_vel_b - self._robot.data.root_lin_vel_b, dim=1)
        yaw_error_abs = torch.abs(yaw_error)

        # ========== 计算加权奖励 ==========
        rewards = {
            "lin_vel": lin_vel_tracking * self.cfg.lin_vel_tracking_reward_scale * self.step_dt,
            "yaw": yaw_tracking * yaw_active * self.cfg.yaw_tracking_reward_scale * self.step_dt,
            "alive": alive * self.cfg.alive_reward_scale * self.step_dt,
            "ang_vel": ang_vel * self.cfg.ang_vel_penalty_scale * self.step_dt,
            "action_rate": action_rate * self.cfg.action_rate_penalty_scale * self.step_dt,
            "vertical_vel": vertical_vel * self.cfg.vertical_speed_penalty_scale * self.step_dt,
            "drift": drift * self.cfg.position_drift_penalty_scale * self.step_dt,
            "flat_orientation": flat_orientation * self.cfg.flat_orientation_penalty_scale * self.step_dt,
            "termination": termination * self.cfg.termination_penalty_scale,
        }

        # 总奖励：所有奖励项求和
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        self._previous_actions.copy_(self._actions)

        # ========== 更新奖励累计（用于日志） ==========
        for key, value in rewards.items():
            self._episode_sums[key] += value
        self._episode_sums["lin_vel_error"] += lin_vel_error_norm * self.step_dt
        self._episode_sums["yaw_error"] += yaw_error_abs * self.step_dt
        self._episode_sums["lin_vel_score"] += lin_vel_tracking * self.step_dt
        self._episode_sums["yaw_score"] += (yaw_tracking * yaw_active) * self.step_dt

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """判断episode结束条件（IsaacLab核心方法）

        Returns:
            tuple: (终止标志, 超时标志)，均为[环境数,]的布尔张量
        """
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        drift = torch.linalg.norm(self._robot.data.root_pos_w[:, :2] - self._terrain.env_origins[:, :2], dim=1)
        died = torch.logical_or(self._robot.data.root_pos_w[:, 2] < 0.1, self._robot.data.root_pos_w[:, 2] > 2.5)
        died |= drift > self.cfg.max_distance_from_origin
        
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """重置指定环境（IsaacLab核心方法）
        重置环境状态、目标位置、日志等

        Args:
            env_ids: 需要重置的环境ID列表
        """
        # 如果未指定env_ids，重置所有环境
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # ========== 日志记录 ==========
        desired_lin_vel_b = quat_apply_inverse(self._robot.data.root_quat_w[env_ids], self._desired_lin_vel_w[env_ids])
        final_lin_vel_error = torch.linalg.norm(
            desired_lin_vel_b - self._robot.data.root_lin_vel_b[env_ids], dim=1
        ).mean()
        final_lin_vel_score = torch.exp(
            -self.cfg.lin_vel_tracking_exp_scale
            * torch.sum(torch.square(desired_lin_vel_b - self._robot.data.root_lin_vel_b[env_ids]), dim=1)
        ).mean()
        actual_horiz_vel = self._robot.data.root_lin_vel_w[env_ids, :2]
        actual_horiz_speed = torch.linalg.norm(actual_horiz_vel, dim=1)
        desired_yaw_w = torch.atan2(actual_horiz_vel[:, 1], actual_horiz_vel[:, 0])
        yaw_error_abs = torch.abs(_wrap_to_pi(desired_yaw_w - _quat_to_yaw(self._robot.data.root_quat_w[env_ids])))
        yaw_active = actual_horiz_speed > 0.1
        final_yaw_error = yaw_error_abs[yaw_active].mean() if yaw_active.any() else torch.zeros(1, device=self.device)[0]
        final_yaw_score = (
            torch.exp(-self.cfg.yaw_tracking_exp_scale * torch.square(yaw_error_abs[yaw_active])).mean()
            if yaw_active.any()
            else torch.zeros(1, device=self.device)[0]
        )
        
        # 准备日志数据：各奖励项的平均每步奖励
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            if key in {"lin_vel_error", "yaw_error", "lin_vel_score", "yaw_score"}:
                extras["Metrics/avg_" + key] = episodic_sum_avg / self.max_episode_length_s
            else:
                extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0  # 重置累计奖励
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        
        # 记录终止原因统计
        extras = dict()
        extras["Episode_Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()  # 坠毁数量
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()  # 超时数量
        extras["Metrics/final_lin_vel_error"] = final_lin_vel_error.item()
        extras["Metrics/final_lin_vel_score"] = final_lin_vel_score.item()
        extras["Metrics/final_yaw_error"] = final_yaw_error.item()
        extras["Metrics/final_yaw_score"] = final_yaw_score.item()
        self.extras["log"].update(extras)

        # ========== 重置机器人 ==========
        self._robot.reset(env_ids)  # 重置机器人状态
        super()._reset_idx(env_ids)  # 调用父类重置方法
        
        # 如果重置所有环境，随机化episode长度（避免同时重置）
        if len(env_ids) == self.num_envs:
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        # ========== 重置动作 ==========
        self._actions[env_ids] = 0.0  # 重置动作到0
        self._previous_actions[env_ids] = 0.0
        self._applied_actions[env_ids] = 0.0

        self._resample_commands(torch.ones(len(env_ids), dtype=torch.bool, device=self.device), env_ids=env_ids, instant=True)

        # ========== 重置机器人初始状态 ==========
        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()  # 默认关节位置
        joint_vel = self._robot.data.default_joint_vel[env_ids].clone()  # 默认关节速度
        default_root_state = self._robot.data.default_root_state[env_ids].clone()  # 默认根状态
        
        # 根位置加上环境原点偏移（确保每个环境的机器人在正确位置）
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        default_root_state[:, 2] = torch.empty(len(env_ids), device=self.device).uniform_(0.8, 1.2)
        random_yaw = torch.empty(len(env_ids), device=self.device).uniform_(-3.14159, 3.14159)
        half_yaw = random_yaw * 0.5
        default_root_state[:, 3] = torch.cos(half_yaw)
        default_root_state[:, 4] = 0.0
        default_root_state[:, 5] = 0.0
        default_root_state[:, 6] = torch.sin(half_yaw)
        default_root_state[:, 7:10] = torch.zeros(len(env_ids), 3, device=self.device)
        default_root_state[:, 10:13] = torch.zeros(len(env_ids), 3, device=self.device)
        
        # 将状态写入仿真
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)  # 写入根位姿
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)  # 写入根速度
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)  # 写入关节状态

    def _set_debug_vis_impl(self, debug_vis: bool):
        """设置调试可视化（IsaacLab核心方法）
        创建/销毁可视化标记器

        Args:
            debug_vis: 是否启用调试可视化
        """
        if debug_vis:
            # 如果还没有创建目标位置可视化器，创建它
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()  # 复制立方体标记配置
                marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)  # 设置标记大小（5cm立方体）
                marker_cfg.prim_path = "/Visuals/Command/goal_velocity"  # 标记在USD中的路径
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)  # 创建可视化器
            # 显示标记
            self.goal_pos_visualizer.set_visibility(True)
        else:
            # 隐藏标记（如果存在）
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        """调试可视化回调函数
        更新目标位置标记的显示

        Args:
            event: 可视化事件（由IsaacLab触发）
        """
        command_marker_pos = self._robot.data.root_pos_w.clone()
        command_marker_pos += torch.cat([self._desired_lin_vel_w[:, :2], torch.zeros(self.num_envs, 1, device=self.device)], dim=1)
        command_marker_pos[:, 2] = self._robot.data.root_pos_w[:, 2] + 0.3
        self.goal_pos_visualizer.visualize(command_marker_pos)

    def _resample_commands(self, masks: torch.Tensor, env_ids: torch.Tensor | None = None, instant: bool = False):
        if env_ids is None:
            env_ids = torch.nonzero(masks, as_tuple=False).squeeze(-1)
        elif masks.ndim > 0 and masks.numel() == len(env_ids):
            env_ids = env_ids[masks]

        if env_ids.numel() == 0:
            return

        num_envs = len(env_ids)
        self._desired_lin_vel_target_w[env_ids, :2] = torch.empty(num_envs, 2, device=self.device).uniform_(
            self.cfg.cmd_lin_vel_xy_range[0], self.cfg.cmd_lin_vel_xy_range[1]
        )
        self._desired_lin_vel_target_w[env_ids, 2] = torch.empty(num_envs, device=self.device).uniform_(
            self.cfg.cmd_lin_vel_z_range[0], self.cfg.cmd_lin_vel_z_range[1]
        )

        zero_mask = torch.rand(num_envs, device=self.device) < self.cfg.cmd_zero_prob
        self._desired_lin_vel_target_w[env_ids[zero_mask]] = 0.0
        if instant:
            self._desired_lin_vel_w[env_ids] = self._desired_lin_vel_target_w[env_ids]

        self._cmd_resample_interval[env_ids] = torch.empty(num_envs, device=self.device).uniform_(
            self.cfg.cmd_resample_time_range[0], self.cfg.cmd_resample_time_range[1]
        )
        self._cmd_resample_elapsed[env_ids] = 0.0
