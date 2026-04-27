# Copyright (c) 2026 Alex Jauregui & Erik Eguskiza.
# Stage 2 v6: Obstacles + Single Target Navigation
# 第二阶段版本6：带障碍物和单目标点导航的无人机环境

from __future__ import annotations

import gymnasium as gym
import torch

# 导入IsaacLab核心模块
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg  # 关节机器人资产
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg  # 直接强化学习环境基类
from isaaclab.scene import InteractiveSceneCfg  # 交互式场景配置
from isaaclab.sim import SimulationCfg  # 仿真配置
from isaaclab.terrains import TerrainImporterCfg  # 地形导入配置
from isaaclab.utils import configclass  # 配置类装饰器
from isaaclab.utils.math import subtract_frame_transforms, quat_apply_inverse  # 坐标变换和四元数操作
from isaaclab.markers import VisualizationMarkers  # 可视化标记工具
from isaaclab.markers.config import CUBOID_MARKER_CFG  # 立方体标记配置

from assets.omninxt.omninxt import OMNINXT_CFG
from controller import CrazyflieController, config as controller_config


def _quat_to_euler_deg(quat_w: torch.Tensor) -> torch.Tensor:
    """Convert (w, x, y, z) quaternion to Euler XYZ angles in degrees."""
    w, x, y, z = quat_w[:, 0], quat_w[:, 1], quat_w[:, 2], quat_w[:, 3]

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    pitch = torch.asin(torch.clamp(sinp, -1.0, 1.0))

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return torch.rad2deg(torch.stack([roll, pitch, yaw], dim=1))


@configclass  # 将类转换为配置类，支持序列化/反序列化
class QuadcopterObstaclesEnvCfg(DirectRLEnvCfg):
    """带障碍物和方向观测的单目标点环境配置类"""
    
    # Episode配置
    episode_length_s = 90.0  # 每个episode时长（秒），拉长以减少环境重置频率
    decimation = 2  # 动作执行频率 = 仿真频率/decimation (100/2=50Hz)
    
    # 障碍物配置
    num_obstacles = 100  # 每个环境的障碍物数量
    num_closest_obstacles = 5  # 只观测最近的5个障碍物（降低观测维度）
    
    # 目标点配置
    target_reach_threshold = 0.5  # 到达目标点的距离阈值（米）
    
    # 观测空间：12基础特征 + 3当前命令 + 5*4（最近障碍物：方向3+距离1） + 1完成标志 = 36维
    # 填充维度：98 - 36 = 62（保持与基础版一致的98维观测空间）
    observation_space = 98
    action_space = 3  # 动作空间：3轴速度命令，传入底层controller
    state_space = 0  # 状态空间维度（未使用）
    debug_vis = True  # 启用调试可视化

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
        env_spacing=45.0,  # 环境间距需大于单环境活动区域，避免跨环境干扰
        replicate_physics=True  # 复制物理参数
    )

    # 机器人配置
    robot: ArticulationCfg = OMNINXT_CFG.replace(prim_path="/World/envs/env_.*/Robot")  # OmniNxt USD路径
    cmd_body_vel_xy_max = 1.0  # 机体系x/y速度命令范围（m/s）
    cmd_vel_z_max = 1.0  # z速度命令范围（m/s）

    # 障碍物详细配置
    obstacle_height = 1.5  # 障碍物高度（米）
    obstacle_radius = 0.15  # 障碍物半径（米）
    obstacle_spawn_range = 20.0  # 障碍物生成半宽（对应40x40区域）
    obstacle_safe_zone = 1.0  # 无人机周围安全区（障碍物不会生成在该范围内）
    obstacle_min_separation = 0.6  # 障碍物之间最小间距（米）
    obstacle_detection_range = 4.0  # 障碍物检测范围（米）

    # 目标点详细配置
    target_spawn_range = 20.0  # 目标点采样半宽（对应40x40区域）
    target_min_height = 0.5  # 目标点最小高度（米）
    target_max_height = 1.5  # 目标点最大高度（米）
    target_obstacle_clearance = 2.0  # 目标点与障碍物的最小水平间距（米）

    # 奖励系数配置
    vel_tracking_reward_scale = 0.5  # 弱化命令跟踪，避免策略学会原地稳定
    vel_tracking_exp_scale = 4.0  # 速度跟踪指数缩放（对齐Go2/阶段1风格）
    ang_vel_reward_scale = -0.01  # 角速度惩罚（鼓励姿态稳定）
    distance_to_target_reward_scale = 20.0  # 到目标点距离奖励
    target_velocity_reward_scale = 4.0  # 朝目标方向速度奖励
    target_reached_bonus = 100.0  # 到达目标点奖励
    obstacle_proximity_reward_scale = -6.0  # 障碍物接近惩罚
    progress_reward_scale = 15.0  # 向目标点前进的进度奖励


class QuadcopterObstaclesEnv(DirectRLEnv):
    """带方向障碍物观测的单目标点无人机环境类"""
    
    cfg: QuadcopterObstaclesEnvCfg  # 类型注解：cfg是QuadcopterObstaclesEnvCfg类型

    def __init__(self, cfg: QuadcopterObstaclesEnvCfg, render_mode: str | None = None, **kwargs):
        """初始化环境
        
        Args:
            cfg: 环境配置实例
            render_mode: 渲染模式
            **kwargs: 其他参数
        """
        super().__init__(cfg, render_mode, **kwargs)

        # ========== 动作和力初始化 ==========
        # 动作张量：[环境数, 动作维度(3)]
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        # 推力张量：[环境数, 1, 3]
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        # 力矩张量：[环境数, 1, 3]
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._cmd_vel_b = torch.zeros(self.num_envs, 3, device=self.device)
        
        # ========== 目标点初始化 ==========
        self._target_positions_local = torch.zeros(self.num_envs, 3, device=self.device)
        self._target_reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._prev_dist_to_target = torch.zeros(self.num_envs, device=self.device)
        
        # ========== 障碍物初始化 ==========
        # 障碍物位置（局部坐标系）：[环境数, 障碍物数, 3]（包含高度信息）
        self._obstacle_positions_local = torch.zeros(
            self.num_envs, self.cfg.num_obstacles, 3, device=self.device
        )
        
        # ========== 环境重置 ==========
        all_env_ids = torch.arange(self.num_envs, device=self.device)
        self._randomize_obstacles(all_env_ids)  # 随机生成障碍物
        self._randomize_targets(all_env_ids)  # 随机生成目标点

        # ========== 日志初始化 ==========
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "vel_tracking",          # 速度跟踪奖励累计
                "ang_vel",               # 角速度惩罚累计
                "distance_to_target",    # 到目标点距离奖励累计
                "target_velocity",       # 朝目标方向运动奖励累计
                "target_bonus",          # 目标点奖励累计
                "obstacle_proximity",    # 障碍物接近惩罚累计
                "progress",              # 进度奖励累计
                "vel_tracking_error",    # 速度跟踪误差累计
            ]
        }
        
        # ========== 机器人参数初始化 ==========
        self._body_id = self._robot.find_bodies("body")[0]  # 无人机主体ID
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum().item()
        self._controller = CrazyflieController(
            num_envs=self.num_envs,
            device=self.device,
            attitude_dt=self.step_dt,
            position_dt=self.step_dt,
        )

        # 碰撞阈值（小于该距离视为碰撞）
        self._collision_threshold = 0.05

        # 启用调试可视化
        self.set_debug_vis(self.cfg.debug_vis)

        # 打印环境信息
        print(f"[INFO] Quadcopter Obstacles v6 - Single Target Navigation")
        print(f"[INFO] Obstacles: {self.cfg.num_obstacles} (observing {self.cfg.num_closest_obstacles} closest)")
        print(f"[INFO] Task: reach one random target, then reset")
        print(f"[INFO] Policy action: 3-axis velocity command -> OmniNxt controller")
        print(f"[INFO] Observation space: {self.cfg.observation_space}")
        print(f"[INFO] OmniNxt mass from asset: {self._robot_mass:.4f} kg")
        print(f"[INFO] Controller configured mass: {controller_config.CF_MASS:.4f} kg")
        if abs(self._robot_mass - controller_config.CF_MASS) > 0.05:
            print("[WARN] OmniNxt mass and controller CF_MASS differ significantly. Retune controller/config.py if flight is unstable.")

    def _setup_scene(self):
        """设置仿真场景（IsaacLab核心方法）"""
        # 创建无人机关节机器人实例
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        # 配置地形
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # 克隆环境（批量创建多个环境）
        self.scene.clone_environments(copy_from_source=False)
        
        # CPU仿真时过滤碰撞
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        
        # 添加环境灯光
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _randomize_obstacles(self, env_ids: torch.Tensor):
        """随机生成障碍物位置
        
        Args:
            env_ids: 需要生成障碍物的环境ID列表
        """
        num_envs_to_reset = len(env_ids)
        placed_xy = torch.zeros(num_envs_to_reset, self.cfg.num_obstacles, 2, device=self.device)
        min_separation = self.cfg.obstacle_min_separation

        for obstacle_idx in range(self.cfg.num_obstacles):
            chosen_xy = None
            chosen_score = None

            for _ in range(32):
                candidate_xy = torch.empty(num_envs_to_reset, 2, device=self.device).uniform_(
                    -self.cfg.obstacle_spawn_range, self.cfg.obstacle_spawn_range
                )
                candidate_radius = torch.linalg.norm(candidate_xy, dim=1)
                valid_mask = candidate_radius >= (self.cfg.obstacle_safe_zone + self.cfg.obstacle_radius + 0.2)

                if obstacle_idx == 0:
                    min_dist = torch.full((num_envs_to_reset,), float("inf"), device=self.device)
                else:
                    distances = torch.linalg.norm(
                        candidate_xy.unsqueeze(1) - placed_xy[:, :obstacle_idx], dim=2
                    )
                    min_dist = distances.min(dim=1).values
                    valid_mask &= min_dist >= min_separation

                if chosen_xy is None:
                    chosen_xy = candidate_xy
                    chosen_score = torch.where(valid_mask, min_dist, torch.full_like(min_dist, -1.0))
                else:
                    candidate_score = torch.where(valid_mask, min_dist, torch.full_like(min_dist, -1.0))
                    better_mask = candidate_score > chosen_score
                    chosen_xy[better_mask] = candidate_xy[better_mask]
                    chosen_score[better_mask] = candidate_score[better_mask]

                if valid_mask.all():
                    chosen_xy = candidate_xy
                    break

                chosen_xy[valid_mask] = candidate_xy[valid_mask]
                chosen_score[valid_mask] = min_dist[valid_mask]

            placed_xy[:, obstacle_idx] = chosen_xy

        self._obstacle_positions_local[env_ids, :, :2] = placed_xy
        self._obstacle_positions_local[env_ids, :, 2] = self.cfg.obstacle_height / 2

    def _randomize_targets(self, env_ids: torch.Tensor):
        """随机生成目标点位置
        
        Args:
            env_ids: 需要生成目标点的环境ID列表
        """
        num_envs_to_reset = len(env_ids)
        xy_limit = self.cfg.target_spawn_range
        clearance = self.cfg.target_obstacle_clearance + self.cfg.obstacle_radius

        best_xy = None
        best_clearance = None

        for _ in range(64):
            candidate_xy = torch.empty(num_envs_to_reset, 2, device=self.device).uniform_(-xy_limit, xy_limit)
            distances = torch.linalg.norm(
                candidate_xy.unsqueeze(1) - self._obstacle_positions_local[env_ids, :, :2], dim=2
            )
            min_clearance = distances.min(dim=1).values
            origin_clearance = torch.linalg.norm(candidate_xy, dim=1)
            valid_mask = (min_clearance > clearance) & (origin_clearance > self.cfg.obstacle_safe_zone)

            if best_xy is None:
                best_xy = candidate_xy
                best_clearance = min_clearance
            else:
                better_mask = min_clearance > best_clearance
                best_xy[better_mask] = candidate_xy[better_mask]
                best_clearance[better_mask] = min_clearance[better_mask]

            if valid_mask.all():
                best_xy = candidate_xy
                break

            unresolved_mask = min_clearance <= clearance
            if not unresolved_mask.any():
                best_xy = candidate_xy
                break

            if best_xy is not None:
                best_xy[valid_mask] = candidate_xy[valid_mask]

        z = torch.empty(num_envs_to_reset, device=self.device).uniform_(
            self.cfg.target_min_height,
            self.cfg.target_max_height,
        )

        self._target_positions_local[env_ids, :2] = best_xy
        self._target_positions_local[env_ids, 2] = z

        self._target_reached[env_ids] = False
        self._prev_dist_to_target[env_ids] = 0.0

    def _get_target_world(self) -> torch.Tensor:
        """获取当前目标点的世界坐标
        
        Returns:
            torch.Tensor: [环境数, 3] 当前目标点的世界坐标
        """
        target_world = self._target_positions_local.clone()
        target_world[:, :2] += self._terrain.env_origins[:, :2]
        return target_world

    def _update_prev_target_distance(self, env_ids: torch.Tensor):
        target_world = self._target_positions_local[env_ids].clone()
        target_world[:, :2] += self._terrain.env_origins[env_ids, :2]
        distance_to_target = torch.linalg.norm(
            target_world - self._robot.data.root_pos_w[env_ids], dim=1
        )
        self._prev_dist_to_target[env_ids] = distance_to_target

    def _compute_closest_obstacles_directional(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        计算机体坐标系下最近N个障碍物的方向和距离（核心方法）
        
        Returns:
            directions: (num_envs, num_closest, 3) - 机体坐标系下的单位方向向量
            distances: (num_envs, num_closest) - 归一化距离
        """
        # 获取无人机世界坐标和姿态
        drone_pos_w = self._robot.data.root_pos_w  # (num_envs, 3)
        drone_quat_w = self._robot.data.root_quat_w  # (num_envs, 4)
        env_origins = self._terrain.env_origins  # (num_envs, 3) 环境原点
        
        # 转换为局部坐标（环境内的相对坐标）
        drone_pos_local = drone_pos_w - env_origins  # (num_envs, 3)
        
        # 计算无人机到每个障碍物的向量
        # drone_pos_local: (num_envs, 3) → (num_envs, 1, 3) 扩展维度
        drone_expanded = drone_pos_local.unsqueeze(1)  # (num_envs, 1, 3)
        # 障碍物向量：(num_envs, num_obstacles, 3)
        to_obstacles = self._obstacle_positions_local - drone_expanded
        
        # 计算3D距离
        distances = torch.linalg.norm(to_obstacles, dim=2)  # (num_envs, num_obstacles)
        distances = distances - self.cfg.obstacle_radius  # 减去障碍物半径，得到到表面的距离
        
        # 获取最近N个障碍物的索引
        _, closest_indices = torch.topk(distances, self.cfg.num_closest_obstacles, dim=1, largest=False)
        # closest_indices: (num_envs, num_closest)
        
        # 提取最近障碍物的向量和距离
        batch_indices = torch.arange(self.num_envs, device=self.device).unsqueeze(1).expand(-1, self.cfg.num_closest_obstacles)
        closest_vectors = to_obstacles[batch_indices, closest_indices]  # (num_envs, num_closest, 3)
        closest_distances = distances[batch_indices, closest_indices]  # (num_envs, num_closest)
        
        # 归一化方向向量（单位向量）
        closest_vectors_norm = closest_vectors / (torch.linalg.norm(closest_vectors, dim=2, keepdim=True) + 1e-6)
        
        # 将世界坐标系下的方向向量转换为机体坐标系
        # 展平向量以便使用quat_apply_inverse函数
        num_closest = self.cfg.num_closest_obstacles
        closest_vectors_flat = closest_vectors_norm.reshape(self.num_envs * num_closest, 3)
        # 扩展四元数维度并展平
        drone_quat_expanded = drone_quat_w.unsqueeze(1).expand(-1, num_closest, -1).reshape(self.num_envs * num_closest, 4)
        
        # 四元数逆变换：世界坐标系 → 机体坐标系
        directions_body = quat_apply_inverse(drone_quat_expanded, closest_vectors_flat)
        
        # 归一化距离（0-1之间）
        distances_normalized = (closest_distances / self.cfg.obstacle_detection_range).clamp(0.0, 1.0)
        
        return directions_body, distances_normalized

    def _compute_min_obstacle_distance(self) -> torch.Tensor:
        """计算到最近障碍物的距离（用于碰撞检测和奖励计算）
        
        Returns:
            torch.Tensor: [环境数,] 每个环境到最近障碍物的距离
        """
        # 只考虑xy平面的距离（简化碰撞检测）
        drone_pos_w = self._robot.data.root_pos_w[:, :2]
        env_origins_xy = self._terrain.env_origins[:, :2]
        drone_pos_local = drone_pos_w - env_origins_xy
        
        # 扩展维度计算距离
        drone_expanded = drone_pos_local.unsqueeze(1)
        distances = torch.linalg.norm(
            drone_expanded - self._obstacle_positions_local[:, :, :2], dim=2
        )
        # 减去障碍物半径
        distances = distances - self.cfg.obstacle_radius
        
        # 返回每个环境的最小距离
        return distances.min(dim=1).values

    def _pre_physics_step(self, actions: torch.Tensor):
        """物理步前处理：将策略输出速度命令传入controller"""
        self._actions = actions.clone().clamp(-1.0, 1.0)
        self._cmd_vel_b[:, :2] = self._actions[:, :2] * self.cfg.cmd_body_vel_xy_max
        self._cmd_vel_b[:, 2] = self._actions[:, 2] * self.cfg.cmd_vel_z_max

        self._controller.set_velocity_setpoint(
            vx=self._cmd_vel_b[:, 0],
            vy=self._cmd_vel_b[:, 1],
            vz=self._cmd_vel_b[:, 2],
            velocity_body=True,
        )

        state = {
            "position": self._robot.data.root_pos_w,
            "velocity": self._robot.data.root_lin_vel_w,
            "attitude": _quat_to_euler_deg(self._robot.data.root_quat_w),
            "angular_velocity": torch.rad2deg(self._robot.data.root_ang_vel_b),
        }
        force, torque = self._controller.compute(state)
        self._thrust[:, 0, :] = force
        self._moment[:, 0, :] = torque

    def _apply_action(self):
        """应用动作到仿真：施加推力和力矩到无人机"""
        self._robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)

    def _get_observations(self) -> dict:
        """构建观测空间（包含方向障碍物信息）
        
        Returns:
            dict: 包含policy观测的字典
        """
        # ========== 当前目标点在机体坐标系下的位置 ==========
        target_world = self._get_target_world()
        target_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_pos_w, self._robot.data.root_quat_w, target_world
        )
        
        # ========== 最近障碍物的方向和距离 ==========
        obstacle_directions, obstacle_distances = self._compute_closest_obstacles_directional()
        # obstacle_directions: (num_envs, 5, 3) → 展平为15维
        obstacle_dirs_flat = obstacle_directions.reshape(self.num_envs, -1)  # (num_envs, 15)
        
        # ========== 构建完整观测 ==========
        obs = torch.cat(
            [
                self._robot.data.root_lin_vel_b,      # 3: 机体坐标系线速度
                self._robot.data.root_ang_vel_b,      # 3: 机体坐标系角速度
                self._robot.data.projected_gravity_b, # 3: 机体坐标系重力向量
                target_pos_b,                          # 3: 机体坐标系目标点位置
                self._cmd_vel_b,                       # 3: 当前下发给controller的速度命令
                obstacle_dirs_flat,                    # 15: 5个障碍物的方向向量
                obstacle_distances,                    # 5: 5个障碍物的归一化距离
                self._target_reached.float().unsqueeze(1),  # 1: 目标点完成标志
            ],
            dim=-1,
        )  # 总计：36维
        
        # ========== 添加padding（保持98维） ==========
        padding = torch.zeros(self.num_envs, 62, device=self.device)
        obs = torch.cat([obs, padding], dim=-1)
        
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """计算奖励（核心方法）
        
        Returns:
            torch.Tensor: [环境数,] 每个环境的总奖励
        """
        # ========== 速度命令跟踪奖励 ==========
        velocity_cmd_frame = torch.stack(
            [
                self._robot.data.root_lin_vel_b[:, 0],
                self._robot.data.root_lin_vel_b[:, 1],
                self._robot.data.root_lin_vel_w[:, 2],
            ],
            dim=1,
        )
        vel_tracking_sq_error = torch.sum(torch.square(self._cmd_vel_b - velocity_cmd_frame), dim=1)
        vel_tracking_error = torch.linalg.norm(self._cmd_vel_b - velocity_cmd_frame, dim=1)
        vel_tracking = torch.exp(-self.cfg.vel_tracking_exp_scale * vel_tracking_sq_error)
        ang_vel = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)  # 角速度平方和
        
        # ========== 到目标点距离奖励 ==========
        target_world = self._get_target_world()
        distance_to_target = torch.linalg.norm(target_world - self._robot.data.root_pos_w, dim=1)
        distance_reward = 1 - torch.tanh(distance_to_target / 2.0)  # 距离越近，奖励越高

        target_direction = target_world - self._robot.data.root_pos_w
        target_direction = target_direction / torch.linalg.norm(target_direction, dim=1, keepdim=True).clamp_min(1e-6)
        target_velocity = torch.sum(self._robot.data.root_lin_vel_w * target_direction, dim=1).clamp(-1.0, 1.5)

        # ========== 进度奖励（向目标点前进） ==========
        progress_reward = (self._prev_dist_to_target - distance_to_target).clamp(-1.0, 1.0)
        self._prev_dist_to_target = distance_to_target.clone()

        # ========== 目标点到达奖励 ==========
        target_reached = (distance_to_target < self.cfg.target_reach_threshold) & (~self._target_reached)
        target_bonus = torch.zeros(self.num_envs, device=self.device)
        if target_reached.any():
            target_bonus[target_reached] = self.cfg.target_reached_bonus
            self._target_reached[target_reached] = True
        
        # ========== 障碍物接近惩罚 ==========
        min_obstacle_dist = self._compute_min_obstacle_distance()
        
        # 距离越近，惩罚越严重（指数衰减）
        obstacle_proximity = torch.where(
            min_obstacle_dist < 1.0,
            torch.exp(-min_obstacle_dist * 3.0),  # 更激进的惩罚函数
            torch.zeros_like(min_obstacle_dist)
        )
        
        # ========== 总奖励计算 ==========
        rewards = {
            "vel_tracking": vel_tracking * self.cfg.vel_tracking_reward_scale * self.step_dt,
            "ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
            "distance_to_target": distance_reward * self.cfg.distance_to_target_reward_scale * self.step_dt,
            "target_velocity": target_velocity.clamp(min=0.0) * self.cfg.target_velocity_reward_scale * self.step_dt,
            "target_bonus": target_bonus,  # 目标点奖励不乘以步长（一次性奖励）
            "obstacle_proximity": obstacle_proximity * self.cfg.obstacle_proximity_reward_scale * self.step_dt,
            "progress": progress_reward * self.cfg.progress_reward_scale * self.step_dt,
        }
        
        # 求和得到总奖励
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        
        # 更新奖励累计（日志）
        for key, value in rewards.items():
            self._episode_sums[key] += value
        self._episode_sums["vel_tracking_error"] += vel_tracking_error * self.step_dt
            
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """判断episode结束条件
        
        Returns:
            tuple: (终止标志, 超时标志)
        """
        # 超时判断：步数达到最大值
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        # 坠毁判断：高度过低/过高
        too_low = self._robot.data.root_pos_w[:, 2] < 0.1
        too_high = self._robot.data.root_pos_w[:, 2] > 2.5
        
        # 碰撞判断：距离障碍物过近
        min_obstacle_dist = self._compute_min_obstacle_distance()
        collision = min_obstacle_dist < self._collision_threshold
        
        # 成功到达目标点
        success = self._target_reached
        
        # 终止条件：坠毁 或 成功完成
        died = too_low | too_high | collision
        terminated = died | success
        
        # 每500步打印调试信息
        if self.common_step_counter % 500 == 0:
            success_rate = self._target_reached.float().mean().item()
            print(f"[DEBUG] Step {self.common_step_counter}: died={died.sum().item()}, "
                  f"success={success.sum().item()}, collision={collision.sum().item()}, "
                  f"success_rate={success_rate:.3f}")
        
        return terminated, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """重置指定环境
        
        Args:
            env_ids: 需要重置的环境ID列表
        """
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # ========== 日志记录 ==========
        success_rate = self._target_reached[env_ids].float().mean().item()
        
        # 记录奖励统计
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            if key == "vel_tracking_error":
                extras["Metrics/avg_vel_tracking_error"] = episodic_sum_avg / self.max_episode_length_s
            else:
                extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0  # 重置累计奖励
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        
        # 记录终止统计
        extras = dict()
        extras["Episode_Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        extras["Metrics/success_rate"] = success_rate
        self.extras["log"].update(extras)

        # ========== 重置机器人 ==========
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        
        # 随机化episode长度（避免同时重置）
        if len(env_ids) == self.num_envs:
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        # ========== 重置动作 ==========
        self._actions[env_ids] = 0.0
        self._cmd_vel_b[env_ids] = 0.0

        # ========== 重新生成障碍物和目标点 ==========
        self._randomize_obstacles(env_ids)
        self._randomize_targets(env_ids)
        
        # ========== 重置机器人初始状态 ==========
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        controller_state = {
            "position": self._robot.data.root_pos_w,
            "attitude": _quat_to_euler_deg(self._robot.data.root_quat_w),
        }
        self._controller.reset(controller_state, env_ids=env_ids)
        self._controller.set_velocity_setpoint(
            vx=torch.zeros(len(env_ids), device=self.device),
            vy=torch.zeros(len(env_ids), device=self.device),
            vz=torch.zeros(len(env_ids), device=self.device),
            velocity_body=True,
            env_ids=env_ids,
        )
        self._update_prev_target_distance(env_ids)

    def _set_debug_vis_impl(self, debug_vis: bool):
        """设置调试可视化标记器"""
        if debug_vis:
            # ========== 当前目标点可视化器 ==========
            if not hasattr(self, "target_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.2, 0.2, 0.2)  # 绿色大立方体
                marker_cfg.markers["cuboid"].visual_material = sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 1.0, 0.0)  # 绿色
                )
                marker_cfg.prim_path = "/Visuals/Target"
                self.target_visualizer = VisualizationMarkers(marker_cfg)
            self.target_visualizer.set_visibility(True)
            
            # ========== 障碍物可视化器 ==========
            if not hasattr(self, "obstacle_visualizer"):
                obs_marker_cfg = CUBOID_MARKER_CFG.copy()
                pillar_size = self.cfg.obstacle_radius * 2  # 障碍物宽度（直径）
                obs_marker_cfg.markers["cuboid"].size = (pillar_size, pillar_size, self.cfg.obstacle_height)
                obs_marker_cfg.markers["cuboid"].visual_material = sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.8, 0.2, 0.2)  # 红色
                )
                obs_marker_cfg.prim_path = "/Visuals/Obstacles"
                self.obstacle_visualizer = VisualizationMarkers(obs_marker_cfg)
            self.obstacle_visualizer.set_visibility(True)
        else:
            # 隐藏所有可视化标记
            if hasattr(self, "target_visualizer"):
                self.target_visualizer.set_visibility(False)
            if hasattr(self, "obstacle_visualizer"):
                self.obstacle_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        """调试可视化回调：更新标记位置"""
        # 更新当前目标点标记
        if hasattr(self, "target_visualizer"):
            target_world = self._get_target_world()
            self.target_visualizer.visualize(target_world)
        
        # 更新障碍物标记
        if hasattr(self, "obstacle_visualizer"):
            env_origins = self._terrain.env_origins
            # 扩展环境原点维度：(num_envs, 3) → (num_envs, num_obstacles, 3)
            env_origins_expanded = env_origins.unsqueeze(1).repeat(1, self.cfg.num_obstacles, 1)
            
            # 转换障碍物坐标到世界坐标系
            obstacle_pos_w = self._obstacle_positions_local.clone()
            obstacle_pos_w[:, :, :2] += env_origins_expanded[:, :, :2]
            
            # 展平并可视化所有障碍物
            obstacle_pos_flat = obstacle_pos_w.reshape(-1, 3)
            self.obstacle_visualizer.visualize(obstacle_pos_flat)
