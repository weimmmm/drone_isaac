from __future__ import annotations

from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
import torch

# 导入 IsaacLab 的核心仿真模块和资产模块
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import schemas as sim_schemas
from isaaclab.utils.math import quat_apply_inverse # 用于将世界系向量转换到机体系
from pxr import Gf, UsdGeom

# 导入自定义的无人机资产 (OmniNxt) 和底层控制器 (CrazyflieController)
from assets.omninxt.omninxt import OMNINXT_CFG
from controller import CrazyflieController, config as controller_config


def _quat_to_euler_deg(quat_w: torch.Tensor) -> torch.Tensor:
    """Convert quaternions in (w, x, y, z) format to Euler XYZ in degrees."""
    # 这里把四元数转换成欧拉角，主要是为了对接 CrazyflieController，
    # 因为控制器内部使用的是 roll / pitch / yaw 的角度表达。
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


@dataclass
class QuadcopterObstaclesEnvCfg:
    # ==================== 基础仿真与并行环境配置 ====================
    @dataclass
    class SimCfg:
        device: str = "cuda:0"

    @dataclass
    class SceneCfg:
        num_envs: int = 8

    episode_length_s: float = 90.0
    decimation: int = 2
    physics_dt: float = 1.0 / 100.0
    device: str = "cuda:0"
    is_finite_horizon: bool = False
    seed: int | None = None

    # ==================== 地图与边界配置 ====================
    map_half_extent: float = 20.0
    wall_half_extent: float = 30.0
    wall_height: float = 5.0
    wall_thickness: float = 0.3

    # ==================== 障碍物配置 ====================
    num_obstacles: int = 120
    num_closest_obstacles: int = 8
    num_closest_dynamic_obstacles: int = 4
    obstacle_height_min: float = 3.0
    obstacle_height_max: float = 5.0
    obstacle_radius_min: float = 0.12
    obstacle_radius_max: float = 0.20
    obstacle_spawn_range: float = 20.0
    obstacle_safe_zone: float = 1.0
    obstacle_min_separation: float = 0.6
    obstacle_detection_range: float = 6.0
    obstacle_proximity_trigger_distance: float = 2.5
    obstacle_collision_margin: float = 0.10
    target_obstacle_clearance: float = 2.0
    dynamic_obstacles_enabled: bool = True
    num_dynamic_obstacles: int = 30
    dynamic_obstacle_min_speed: float = 0.2
    dynamic_obstacle_max_speed: float = 0.6
    dynamic_obstacle_goal_threshold: float = 0.5
    dynamic_obstacle_local_range_x: float = 4.0
    dynamic_obstacle_local_range_y: float = 4.0
    dynamic_obstacle_velocity_resample_interval_s: float = 2.0

    # ==================== 起点 / 终点采样配置 ====================
    spawn_edge_distance: float = 23.0
    target_spawn_range: float = 23.0
    spawn_min_height: float = 0.5
    spawn_max_height: float = 2.5
    target_min_height: float = 0.5
    target_max_height: float = 2.5
    target_reach_threshold: float = 0.5

    # ==================== 动作解释为速度指令的范围 ====================
    cmd_body_vel_xy_max: float = 3.0
    cmd_vel_z_max: float = 1.0

    # ==================== 奖励项权重 ====================
    # 目标推进：既鼓励整体接近目标，也鼓励沿目标方向稳定前进。
    distance_to_target_reward_scale: float = 8.0
    target_velocity_reward_scale: float = 1.0
    progress_reward_scale: float = 4.0
    target_reached_bonus: float = 120.0
    # 避障：保持和动态/静态障碍物及边界墙的安全距离。
    obstacle_proximity_reward_scale: float = -15.0
    time_to_collision_reward_scale: float = -3.0
    dynamic_obstacle_safety_reward_scale: float = 1.0
    # 平滑：抑制相邻控制指令突变。
    action_smoothness_reward_scale: float = -0.2

    # ==================== Gym / IsaacLab 通用配置 ====================
    observation_space: int = 0
    action_space: int = 3
    state_space: int = 0
    auto_reset_done: bool = True
    debug_vis: bool = False
    viewer_eye: tuple[float, float, float] = (-30.0, 0.0, 80.0)
    viewer_lookat: tuple[float, float, float] = (0.0, 0.0, 0.0)
    viewer_cam_prim_path: str = "/OmniverseKit_Persp"
    sim: SimCfg = field(default_factory=SimCfg)
    scene: SceneCfg = field(default_factory=SceneCfg)

    def policy_observation_dim(self) -> int:
        # 低维特权观测由几部分组成：
        # 机体速度、姿态、目标相对位置、当前速度命令，
        # 最近静态障碍物方向/距离，最近动态障碍物结构化特征，以及目标方向障碍距离。
        base_dim = 3 + 3 + 3 + 3
        static_obstacle_dim = self.num_closest_obstacles * 3 + self.num_closest_obstacles
        dynamic_obstacle_dim = self.num_closest_dynamic_obstacles * 10
        extra_dim = 1
        return base_dim + static_obstacle_dim + dynamic_obstacle_dim + extra_dim

    def critic_observation_dim(self) -> int:
        static_count = min(8, max(self.num_obstacles - self.num_dynamic_obstacles, 0))
        dynamic_count = min(5, min(self.num_dynamic_obstacles, self.num_obstacles))
        base_dim = 3 + 3 + 3 + 3
        static_privileged_dim = static_count * 3  # rel_xy + radius
        dynamic_privileged_dim = dynamic_count * 8  # rel_pos_b + rel_vel_b + radius + height
        return base_dim + static_privileged_dim + dynamic_privileged_dim

    def __post_init__(self):
        # 构造完成后，统一把环境数和 device 对齐，并自动写回 observation 维度。
        self.scene.num_envs = int(self.scene.num_envs)
        self.sim.device = self.device
        self.observation_space = self.policy_observation_dim()


class QuadcopterObstaclesEnv(gym.Env):
    """Single-world multi-drone teacher environment with privileged obstacle truth."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, cfg: QuadcopterObstaclesEnvCfg | None = None, render_mode: str | None = None, **kwargs):
        del kwargs
        # 没有传配置时使用默认配置。
        self.cfg = cfg if cfg is not None else QuadcopterObstaclesEnvCfg()
        self.render_mode = render_mode
        if hasattr(self.cfg, "scene"):
            self.cfg.scene.num_envs = int(self.cfg.scene.num_envs)
        if hasattr(self.cfg, "sim"):
            self.cfg.device = self.cfg.sim.device

        self.device = torch.device(self.cfg.device)
        self.num_envs = self.cfg.scene.num_envs
        # 一个 RL step 等于若干个 physics step，因此真正用于奖励和时长计算的是 step_dt。
        self.step_dt = self.cfg.physics_dt * self.cfg.decimation
        self.max_episode_length = int(round(self.cfg.episode_length_s / self.step_dt))
        self.max_episode_length_s = self.max_episode_length * self.step_dt
        self.num_states = 0
        self.common_step_counter = 0

        self.single_action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.cfg.action_space,), dtype=float)
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)
        # teacher 环境只提供一组低维特权观测，因此这里的 observation_space 是一维向量。
        self.single_observation_space = gym.spaces.Dict(
            {
                "policy": gym.spaces.Box(
                    low=-float("inf"),
                    high=float("inf"),
                    shape=(self.cfg.observation_space,),
                    dtype=float,
                ),
                "critic": gym.spaces.Box(
                    low=-float("inf"),
                    high=float("inf"),
                    shape=(self.cfg.critic_observation_dim(),),
                    dtype=float,
                ),
            }
        )
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space, self.num_envs)
        self.num_states = self.cfg.critic_observation_dim()

        self._built = False
        self._sim: sim_utils.SimulationContext | None = None
        self._robot: Articulation | None = None
        self._body_id: torch.Tensor | None = None
        self._rgb_annotator = None
        self._render_product = None

        self._obstacle_paths: list[str] = []
        self._wall_paths: list[str] = []
        self._robot_mass = 0.0

        self._actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self._prev_actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        # 动作会被解释成机体系速度命令，再交给 Crazyflie controller 变成力矩 / 推力。
        self._cmd_vel_b = torch.zeros((self.num_envs, 3), device=self.device)
        self._thrust = torch.zeros((self.num_envs, 1, 3), device=self.device)
        self._moment = torch.zeros((self.num_envs, 1, 3), device=self.device)
        self._controller = CrazyflieController(
            num_envs=self.num_envs,
            device=self.device,
            attitude_dt=self.step_dt,
            position_dt=self.step_dt,
        )

        self._target_positions_w = torch.zeros((self.num_envs, 3), device=self.device)
        self._target_reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # 上一时刻到目标的距离，用于构造 progress reward。
        self._prev_dist_to_target = torch.zeros(self.num_envs, device=self.device)
        self._dynamic_obstacle_step_count = 0
        self._obstacle_positions_w = torch.zeros((self.cfg.num_obstacles, 3), device=self.device)
        self._obstacle_base_positions_w = torch.zeros((self.cfg.num_obstacles, 3), device=self.device)
        default_obstacle_height = 0.5 * (self.cfg.obstacle_height_min + self.cfg.obstacle_height_max)
        default_obstacle_radius = 0.5 * (self.cfg.obstacle_radius_min + self.cfg.obstacle_radius_max)
        self._obstacle_heights = torch.full((self.cfg.num_obstacles,), default_obstacle_height, device=self.device)
        self._obstacle_radii = torch.full((self.cfg.num_obstacles,), default_obstacle_radius, device=self.device)
        self._obstacle_positions_xy = self._obstacle_positions_w[:, :2]
        self._active_obstacle_mask = torch.zeros(self.cfg.num_obstacles, dtype=torch.bool, device=self.device)
        self._dynamic_obstacle_mask = torch.zeros(self.cfg.num_obstacles, dtype=torch.bool, device=self.device)
        self._dynamic_obstacle_speed = torch.zeros(self.cfg.num_obstacles, device=self.device)
        self._dynamic_obstacle_velocity_xy = torch.zeros((self.cfg.num_obstacles, 2), device=self.device)
        self._dynamic_obstacle_origin_w = torch.zeros((self.cfg.num_obstacles, 3), device=self.device)
        self._dynamic_obstacle_goal_w = torch.zeros((self.cfg.num_obstacles, 3), device=self.device)
        self._obstacle_translate_ops: list[UsdGeom.XformOp | None] = []
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._rew_buf = torch.zeros(self.num_envs, device=self.device)
        self._done_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._gravity_vec_w = torch.tensor([0.0, 0.0, -1.0], device=self.device).repeat(self.num_envs, 1)
        self._target_obstacle_clearance_sq = self.cfg.target_obstacle_clearance**2
        obstacle_min_center_distance = self.cfg.obstacle_min_separation + 2.0 * self.cfg.obstacle_radius_max
        self._obstacle_min_center_distance_sq = obstacle_min_center_distance**2
        self._obstacle_safe_radius_sq = (self.cfg.obstacle_safe_zone + self.cfg.obstacle_radius_max + 0.2) ** 2
        self._obstacle_collision_margin_sq = self.cfg.obstacle_collision_margin**2
        self._current_static_obstacle_count = 0
        self._current_dynamic_obstacle_count = 0
        self._current_total_obstacle_count = 0

        self._episode_sums = {
            # 这些值用于 TensorBoard 里按 episode 统计不同奖励项和诊断指标。
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "distance_to_target",
                "target_velocity",
                "target_bonus",
                "obstacle_proximity",
                "progress",
                "action_smoothness",
                "time_to_collision",
                "dynamic_obstacle_safety",
            ]
        }

    @property
    def unwrapped(self) -> QuadcopterObstaclesEnv:
        return self

    @property
    def sim(self) -> sim_utils.SimulationContext:
        if self._sim is None:
            raise RuntimeError("Environment not built yet.")
        return self._sim

    @property
    def robot(self) -> Articulation:
        if self._robot is None:
            raise RuntimeError("Environment not built yet.")
        return self._robot

    def seed(self, seed: int = -1) -> int:
        # 尽量把 torch 和 cuda 的随机种子一起固定，便于复现实验。
        if seed >= 0:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            self.cfg.seed = seed
        return seed

    def _build(self) -> None:
        # 懒构建：只有第一次 reset / step 前才真正创建仿真世界。
        if self._built:
            return

        # 创建 IsaacLab 仿真上下文。
        sim_cfg = sim_utils.SimulationCfg(dt=self.cfg.physics_dt, device=str(self.device))
        self._sim = sim_utils.SimulationContext(sim_cfg)
        self.sim.set_camera_view(
            eye=list(self.cfg.viewer_eye),
            target=list(self.cfg.viewer_lookat),
            camera_prim_path=self.cfg.viewer_cam_prim_path,
        )

        # 创建地面、灯光、共享障碍物、边界墙和所有无人机资产。
        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_cfg.func("/World/Ground", ground_cfg)
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        self._spawn_shared_obstacles()
        self._spawn_boundary_walls()
        self._build_robot_assets()
        
        # 禁用无人机之间的碰撞。强化学习让无人机避障是通过“障碍物惩罚”来学习的，
        # 如果不禁用无人机互撞，在早期探索(exploration)阶段会因为频繁互撞导致样本效率极低。
        self._disable_robot_collisions()

        # reset 之后再读取 body id / mass 等信息，此时物理对象已经真正进入仿真。
        self.sim.reset()
        self._body_id = self.robot.find_bodies("body")[0]  # 找到机体主体 body 的索引。
        self._robot_mass = self.robot.root_physx_view.get_masses()[0].sum().item()  # 汇总整架无人机质量。
        self.robot.write_joint_state_to_sim(self.robot.data.default_joint_pos, self.robot.data.default_joint_vel)  # 把默认关节状态写入仿真。
        self.robot.update(self.cfg.physics_dt)  # 刷新一次本地缓存。
        for _ in range(5):
            # 先空跑几步，让 articulation 缓冲和物理状态稳定下来。
            self.sim.step()
            self.robot.update(self.cfg.physics_dt)

        print(
            f"[INFO] Quadcopter Obstacles Teacher - Single World with {self.num_envs} robots and "
            f"{self._current_total_obstacle_count}/{self.cfg.num_obstacles} active shared obstacles "
            f"(dynamic={self._current_dynamic_obstacle_count})"
        )
        print(
            f"[INFO] Observation space: policy={self.cfg.observation_space}, "
            f"critic={self.cfg.critic_observation_dim()}"
        )
        print(f"[INFO] OmniNxt mass from asset: {self._robot_mass:.4f} kg")
        print(f"[INFO] Controller configured mass: {controller_config.CF_MASS:.4f} kg")
        if abs(self._robot_mass - controller_config.CF_MASS) > 0.05:
            print("[WARN] OmniNxt mass and controller CF_MASS differ significantly. Retune controller/config.py.")

        self._built = True  # 标记环境已经构建完成。

    def _sample_edge_positions(
        self, num_samples: int, lateral_range: float, min_height: float, max_height: float
    ) -> torch.Tensor:
        # 在地图四条边上采样出生点/目标点：
        # 两条 y=±spawn_edge_distance，另外两条 x=±spawn_edge_distance。
        side_indices = torch.randint(0, 4, (num_samples,), device=self.device)  # 随机决定每个点来自哪一条边。
        side_signs = torch.where(
            side_indices % 2 == 0,
            torch.ones(num_samples, device=self.device),  # 偶数边对应正方向。
            -torch.ones(num_samples, device=self.device),  # 奇数边对应负方向。
        )
        lateral = torch.empty(num_samples, device=self.device).uniform_(-lateral_range, lateral_range)  # 沿边方向均匀采样。
        heights = torch.empty(num_samples, device=self.device).uniform_(min_height, max_height)  # 高度均匀采样。

        positions = torch.zeros((num_samples, 3), device=self.device)  # 输出的 xyz 采样结果。
        x_side_mask = side_indices < 2  # 上下边：固定 y，x 为 lateral。
        y_side_mask = ~x_side_mask  # 左右边：固定 x，y 为 lateral。
        positions[x_side_mask, 0] = lateral[x_side_mask]  # 上下边时写入 x。
        positions[x_side_mask, 1] = side_signs[x_side_mask] * self.cfg.spawn_edge_distance  # 上下边时 y 固定为 ±edge。
        positions[y_side_mask, 0] = side_signs[y_side_mask] * self.cfg.spawn_edge_distance  # 左右边时 x 固定为 ±edge。
        positions[y_side_mask, 1] = lateral[y_side_mask]  # 左右边时写入 y。
        positions[:, 2] = heights  # 全部 z 维写入高度。
        return positions  # 返回边缘采样位置。

    def _sample_edge_positions_with_clearance(
        self,
        num_samples: int,
        lateral_range: float,
        min_height: float,
        max_height: float,
        min_separation: float = 0.0,
        avoid_obstacles: bool = False,
        obstacle_clearance: float | None = None,
        avoid_positions_xy: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # 带约束的边缘采样：保证新生成的无人机与其他无人机/障碍物保持安全距离
        positions = torch.zeros((num_samples, 3), device=self.device)  # 保存最终生成的起点/边缘点。
        obstacle_clearance = self.cfg.target_obstacle_clearance if obstacle_clearance is None else obstacle_clearance  # 若未指定则使用默认净空。
        obstacle_clearance_sq = obstacle_clearance**2  # 转成平方，后面统一做平方距离比较。
        for idx in range(num_samples):
            best_candidate = None  # 当前第 idx 个位置最优的备选点。
            best_score = None  # 当前最优备选点的评分。
            # 拒绝采样：每个点最多尝试 128 次随机采样，寻找满足所有约束的解
            for _ in range(128):
                candidate = self._sample_edge_positions(1, lateral_range, min_height, max_height)[0]  # 先随机采一个边缘点。
                valid = True  # 当前候选点是否满足所有约束。
                score = torch.tensor(float("inf"), device=self.device)  # 用最小安全余量作为综合评分。

                # 约束 1：与其他正在生成的无人机保持距离
                if min_separation > 0.0 and idx > 0:
                    d_prev_sq = torch.sum(torch.square(candidate.unsqueeze(0) - positions[:idx]), dim=1)  # 与已经放好的候选点之间的平方距离。
                    min_prev_dist_sq = d_prev_sq.min()  # 最近已放置点的距离平方。
                    valid = bool(valid and (min_prev_dist_sq >= min_separation**2))  # 若最近距离不够则判无效。
                    score = torch.minimum(score, min_prev_dist_sq)  # 用最小间距更新评分。

                # 约束 2：避开已经存在的其他目标点
                if avoid_positions_xy is not None and avoid_positions_xy.numel() > 0:
                    d_keep_sq = torch.sum(torch.square(candidate[:2].unsqueeze(0) - avoid_positions_xy), dim=1)  # 与其他正在运行无人机的 xy 平面距离平方。
                    min_keep_dist_sq = d_keep_sq.min()  # 最近 active 无人机的距离平方。
                    valid = bool(valid and (min_keep_dist_sq >= min_separation**2))  # 太近则拒绝。
                    score = torch.minimum(score, min_keep_dist_sq)  # 最近 active 无人机距离也纳入评分。

                # 约束 3：避开静态障碍物
                if avoid_obstacles and self.cfg.num_obstacles > 0:
                    d_obs_sq = torch.sum(torch.square(candidate[:2].unsqueeze(0) - self._obstacle_positions_xy), dim=1)  # 与所有障碍物中心的平方距离。
                    obstacle_clearance_sq_all = torch.square(obstacle_clearance + self._obstacle_radii)  # 每个障碍物对应的最小允许中心距离平方。
                    valid_mask = d_obs_sq >= obstacle_clearance_sq_all  # 对全部障碍物分别检查是否满足净空。
                    min_margin_sq = (d_obs_sq - obstacle_clearance_sq_all).min()  # 当前候选对“最危险障碍物”的安全余量。
                    valid = bool(valid and bool(valid_mask.all()))  # 有任意一个障碍物不满足就判无效。
                    score = torch.minimum(score, min_margin_sq)  # 用最小安全余量更新评分。

                # 记录得分最高的候选点
                if best_candidate is None or score > best_score:
                    best_candidate = candidate  # 当前候选更优，更新最佳备选点。
                    best_score = score  # 同步更新最佳评分。
                # 如果满足所有约束直接跳出循环
                if valid:
                    best_candidate = candidate  # 当前候选合法，直接采用。
                    break  # 结束当前位置的拒绝采样。
            positions[idx] = best_candidate  # 写入最终点位。
        return positions  # 返回所有带安全约束的边缘采样位置。

    def _infer_edge_side_indices(self, positions: torch.Tensor) -> torch.Tensor:
        # 逆向推导函数：给出一个 (x, y) 坐标，判断它位于矩形场地的哪一条边上
        # 巧妙利用绝对值比较 (x_abs vs y_abs) 来区分是在 X 边还是 Y 边
        side_indices = torch.zeros(len(positions), dtype=torch.long, device=self.device)  # 输出的边编号。
        x_abs = torch.abs(positions[:, 0])  # 取绝对 x。
        y_abs = torch.abs(positions[:, 1])  # 取绝对 y。
        y_side_mask = y_abs >= x_abs  # |y| 更大说明更靠近上下边。
        side_indices[y_side_mask] = torch.where(
            positions[y_side_mask, 1] >= 0.0,
            torch.zeros_like(side_indices[y_side_mask]),
            torch.ones_like(side_indices[y_side_mask]),
        )
        x_side_mask = ~y_side_mask  # 否则视为更靠近左右边。
        side_indices[x_side_mask] = torch.where(
            positions[x_side_mask, 0] >= 0.0,
            torch.full_like(side_indices[x_side_mask], 2),
            torch.full_like(side_indices[x_side_mask], 3),
        )
        return side_indices

    def _sample_opposite_edge_positions(self, source_side_indices: torch.Tensor, min_height: float, max_height: float) -> torch.Tensor:
        # 根据出生点所在的边，在对立面 (Opposite Edge) 采样目标点。
        # 这种设计强制无人机必须“横穿”地图，从而必然经过中间的障碍物区域，提高训练数据的有效性。
        num_samples = len(source_side_indices)  # 要采样的目标数量。
        opposite_side_indices = torch.where(source_side_indices % 2 == 0, source_side_indices + 1, source_side_indices - 1)  # 直接映射到对边。
        lateral = torch.empty(num_samples, device=self.device).uniform_(
            -self.cfg.spawn_edge_distance, self.cfg.spawn_edge_distance
        )
        heights = torch.empty(num_samples, device=self.device).uniform_(min_height, max_height)

        positions = torch.zeros((num_samples, 3), device=self.device)  # 输出的目标点。
        y_side_mask = opposite_side_indices < 2  # 对边若是上下边，则固定 y。
        x_side_mask = ~y_side_mask  # 对边若是左右边，则固定 x。
        positions[y_side_mask, 0] = lateral[y_side_mask]
        positions[y_side_mask, 1] = torch.where(
            opposite_side_indices[y_side_mask] == 0,
            torch.full_like(lateral[y_side_mask], self.cfg.spawn_edge_distance),
            torch.full_like(lateral[y_side_mask], -self.cfg.spawn_edge_distance),
        )
        positions[x_side_mask, 1] = lateral[x_side_mask]
        positions[x_side_mask, 0] = torch.where(
            opposite_side_indices[x_side_mask] == 2,
            torch.full_like(lateral[x_side_mask], self.cfg.spawn_edge_distance),
            torch.full_like(lateral[x_side_mask], -self.cfg.spawn_edge_distance),
        )
        positions[:, 2] = heights  # 写入 z 高度。
        return positions  # 返回对边目标。

    def _sample_opposite_edge_positions_with_clearance(
        self,
        source_side_indices: torch.Tensor,
        min_height: float,
        max_height: float,
        min_separation: float = 0.0,
        avoid_obstacles: bool = False,
        obstacle_clearance: float | None = None,
    ) -> torch.Tensor:
        # 对边目标点的带约束版本，逻辑与起点采样类似。
        positions = torch.zeros((len(source_side_indices), 3), device=self.device)  # 最终目标点张量。
        obstacle_clearance = self.cfg.target_obstacle_clearance if obstacle_clearance is None else obstacle_clearance  # 目标点也要避障。
        obstacle_clearance_sq = obstacle_clearance**2  # 提前平方。
        for idx in range(len(source_side_indices)):
            best_candidate = None  # 当前 idx 的最佳目标候选。
            best_score = None  # 当前 idx 的最佳评分。
            side_idx = source_side_indices[idx : idx + 1]  # 当前起点所在边编号。
            for _ in range(128):
                candidate = self._sample_opposite_edge_positions(side_idx, min_height, max_height)[0]  # 采一个对边目标候选。
                valid = True  # 当前候选是否满足全部约束。
                score = torch.tensor(float("inf"), device=self.device)  # 评分初始化为正无穷。

                if min_separation > 0.0 and idx > 0:
                    d_prev_sq = torch.sum(torch.square(candidate.unsqueeze(0) - positions[:idx]), dim=1)  # 与已有目标点的平方距离。
                    min_prev_dist_sq = d_prev_sq.min()  # 最近目标点距离平方。
                    valid = bool(valid and (min_prev_dist_sq >= min_separation**2))  # 若太近则非法。
                    score = torch.minimum(score, min_prev_dist_sq)  # 更新评分。

                if avoid_obstacles and self.cfg.num_obstacles > 0:
                    d_obs_sq = torch.sum(torch.square(candidate[:2].unsqueeze(0) - self._obstacle_positions_xy), dim=1)  # 与全部障碍物中心的平方距离。
                    obstacle_clearance_sq_all = torch.square(obstacle_clearance + self._obstacle_radii)  # 每个障碍物对应的净空阈值。
                    valid_mask = d_obs_sq >= obstacle_clearance_sq_all  # 是否全部满足净空。
                    min_margin_sq = (d_obs_sq - obstacle_clearance_sq_all).min()  # 对最危险障碍物的安全余量。
                    valid = bool(valid and bool(valid_mask.all()))  # 任何一个障碍物不满足都算无效。
                    score = torch.minimum(score, min_margin_sq)  # 更新评分。

                if best_candidate is None or score > best_score:
                    best_candidate = candidate  # 更新当前最优目标候选。
                    best_score = score  # 更新当前最优评分。
                if valid:
                    best_candidate = candidate  # 若当前候选有效则直接使用。
                    break  # 结束当前 idx 的采样。
            positions[idx] = best_candidate  # 写入目标点。
        return positions  # 返回所有对边目标。

    def _sample_obstacle_positions(self, obstacle_indices: torch.Tensor) -> torch.Tensor:
        num_obstacles = int(obstacle_indices.numel())
        positions = torch.zeros((self.cfg.num_obstacles, 3), device=self.device)
        if num_obstacles == 0:
            positions[:, 0] = self.cfg.obstacle_spawn_range + 100.0 + torch.arange(self.cfg.num_obstacles, device=self.device)
            positions[:, 2] = self._obstacle_heights * 0.5
            return positions

        obstacle_radii = self._obstacle_radii[obstacle_indices]
        placed_xy = torch.zeros((num_obstacles, 2), device=self.device)
        for obstacle_idx in range(num_obstacles):
            chosen_xy = None
            chosen_score = None
            candidate_radius = obstacle_radii[obstacle_idx]
            safe_radius_sq = (self.cfg.obstacle_safe_zone + candidate_radius + 0.2) ** 2
            for _ in range(64):
                candidate_xy = torch.empty(2, device=self.device).uniform_(
                    -self.cfg.obstacle_spawn_range, self.cfg.obstacle_spawn_range
                )
                candidate_radius_sq = torch.dot(candidate_xy, candidate_xy)
                valid = candidate_radius_sq >= safe_radius_sq
                if obstacle_idx == 0:
                    min_margin = torch.tensor(float("inf"), device=self.device)
                else:
                    distances = torch.linalg.norm(candidate_xy.unsqueeze(0) - placed_xy[:obstacle_idx], dim=1)
                    required_distance = self.cfg.obstacle_min_separation + candidate_radius + obstacle_radii[:obstacle_idx]
                    margins = distances - required_distance
                    min_margin = margins.min()
                    valid = bool(valid and (min_margin >= 0.0))

                candidate_score = min_margin if valid else torch.tensor(-1.0, device=self.device)
                if chosen_xy is None or candidate_score > chosen_score:
                    chosen_xy = candidate_xy
                    chosen_score = candidate_score
                if valid:
                    chosen_xy = candidate_xy
                    break
            placed_xy[obstacle_idx] = chosen_xy

        positions[:, 0] = self.cfg.obstacle_spawn_range + 100.0 + torch.arange(self.cfg.num_obstacles, device=self.device)
        positions[:, 2] = self._obstacle_heights * 0.5
        positions[obstacle_indices, :2] = placed_xy
        positions[obstacle_indices, 2] = self._obstacle_heights[obstacle_indices] * 0.5
        return positions

    def _apply_obstacle_layout(self, sync_prims: bool = True) -> None:
        self._dynamic_obstacle_step_count = 0
        self._dynamic_obstacle_speed.zero_()
        self._dynamic_obstacle_velocity_xy.zero_()
        self._dynamic_obstacle_origin_w.zero_()
        self._dynamic_obstacle_goal_w.zero_()
        self._active_obstacle_mask.zero_()
        self._dynamic_obstacle_mask.zero_()

        dynamic_count = min(max(int(self.cfg.num_dynamic_obstacles), 0), int(self.cfg.num_obstacles))
        static_count = max(int(self.cfg.num_obstacles) - dynamic_count, 0)
        static_indices = torch.arange(static_count, device=self.device, dtype=torch.long)
        dynamic_indices = torch.arange(static_count, static_count + dynamic_count, device=self.device, dtype=torch.long)
        active_indices = torch.cat((static_indices, dynamic_indices), dim=0) if dynamic_count > 0 else static_indices

        self._current_static_obstacle_count = int(static_count)
        self._current_dynamic_obstacle_count = int(dynamic_count)
        self._current_total_obstacle_count = int(active_indices.numel())

        if active_indices.numel() > 0:
            self._active_obstacle_mask[active_indices] = True
            self._obstacle_positions_w[:] = self._sample_obstacle_positions(active_indices)
        else:
            self._obstacle_positions_w[:, 0] = self.cfg.obstacle_spawn_range + 100.0 + torch.arange(self.cfg.num_obstacles, device=self.device)
            self._obstacle_positions_w[:, 1] = 0.0
            self._obstacle_positions_w[:, 2] = self._obstacle_heights * 0.5

        self._obstacle_base_positions_w[:] = self._obstacle_positions_w
        if dynamic_count > 0:
            self._dynamic_obstacle_mask[dynamic_indices] = True
            self._dynamic_obstacle_origin_w[dynamic_indices] = self._obstacle_positions_w[dynamic_indices]
            self._dynamic_obstacle_goal_w[dynamic_indices] = self._obstacle_positions_w[dynamic_indices]
            self._resample_dynamic_obstacle_goals(dynamic_indices)
            self._resample_dynamic_obstacle_velocities(dynamic_indices)

        if sync_prims:
            self._sync_obstacle_prims()

    def _sample_obstacles(self) -> tuple[torch.Tensor, torch.Tensor]:
        obstacle_radii = torch.empty(self.cfg.num_obstacles, device=self.device).uniform_(
            self.cfg.obstacle_radius_min, self.cfg.obstacle_radius_max
        )
        obstacle_heights = torch.empty(self.cfg.num_obstacles, device=self.device).uniform_(
            self.cfg.obstacle_height_min, self.cfg.obstacle_height_max
        )
        return obstacle_radii, obstacle_heights

    def _sync_obstacle_prims(self) -> None:
        if self._sim is None:
            return

        import omni.usd

        stage = omni.usd.get_context().get_stage()
        for idx, prim_path in enumerate(self._obstacle_paths):
            translate_op = self._obstacle_translate_ops[idx] if idx < len(self._obstacle_translate_ops) else None
            if translate_op is None:
                prim = stage.GetPrimAtPath(prim_path)
                if not prim.IsValid():
                    continue
                xform = UsdGeom.Xformable(prim)
                translate_ops = [op for op in xform.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTranslate]
                if translate_ops:
                    translate_op = translate_ops[0]
                else:
                    xform.ClearXformOpOrder()
                    translate_op = xform.AddTranslateOp()
                if idx < len(self._obstacle_translate_ops):
                    self._obstacle_translate_ops[idx] = translate_op
                else:
                    self._obstacle_translate_ops.append(translate_op)

            pos = self._obstacle_positions_w[idx]
            translate_op.Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))

    def _resample_dynamic_obstacle_goals(self, obstacle_indices: torch.Tensor | None = None) -> None:
        if (not self.cfg.dynamic_obstacles_enabled) or self.cfg.num_obstacles == 0:
            return

        if obstacle_indices is None:
            obstacle_indices = torch.nonzero(self._dynamic_obstacle_mask, as_tuple=False).squeeze(-1)
        if obstacle_indices.numel() == 0:
            return

        goal_xy = self._dynamic_obstacle_origin_w[obstacle_indices, :2].clone()
        goal_xy[:, 0] += torch.empty(obstacle_indices.numel(), device=self.device).uniform_(
            -self.cfg.dynamic_obstacle_local_range_x, self.cfg.dynamic_obstacle_local_range_x
        )
        goal_xy[:, 1] += torch.empty(obstacle_indices.numel(), device=self.device).uniform_(
            -self.cfg.dynamic_obstacle_local_range_y, self.cfg.dynamic_obstacle_local_range_y
        )

        xy_limits = self.cfg.obstacle_spawn_range - self._obstacle_radii[obstacle_indices].unsqueeze(1)
        goal_xy = torch.maximum(torch.minimum(goal_xy, xy_limits), -xy_limits)

        self._dynamic_obstacle_goal_w[obstacle_indices, :2] = goal_xy
        self._dynamic_obstacle_goal_w[obstacle_indices, 2] = self._dynamic_obstacle_origin_w[obstacle_indices, 2]

    def _resample_dynamic_obstacle_velocities(self, obstacle_indices: torch.Tensor | None = None) -> None:
        if (not self.cfg.dynamic_obstacles_enabled) or self.cfg.num_obstacles == 0:
            return

        if obstacle_indices is None:
            obstacle_indices = torch.nonzero(self._dynamic_obstacle_mask, as_tuple=False).squeeze(-1)
        if obstacle_indices.numel() == 0:
            return

        goal_vectors = self._dynamic_obstacle_goal_w[obstacle_indices, :2] - self._obstacle_positions_w[obstacle_indices, :2]
        goal_dist = torch.linalg.norm(goal_vectors, dim=1, keepdim=True)
        direction = goal_vectors / goal_dist.clamp_min(1e-6)
        speeds = torch.empty(obstacle_indices.numel(), 1, device=self.device).uniform_(
            self.cfg.dynamic_obstacle_min_speed, self.cfg.dynamic_obstacle_max_speed
        )
        self._dynamic_obstacle_speed[obstacle_indices] = speeds.squeeze(1)
        self._dynamic_obstacle_velocity_xy[obstacle_indices] = direction * speeds

    def _update_dynamic_obstacles(self, dt: float = 0.0) -> None:
        if (not self.cfg.dynamic_obstacles_enabled) or self.cfg.num_obstacles == 0:
            return

        dynamic_indices = torch.nonzero(self._dynamic_obstacle_mask, as_tuple=False).squeeze(-1)
        if dynamic_indices.numel() == 0:
            return

        if self._dynamic_obstacle_step_count != 0:
            goal_dist = torch.linalg.norm(
                self._obstacle_positions_w[dynamic_indices, :3] - self._dynamic_obstacle_goal_w[dynamic_indices, :3], dim=1
            )
            reached_mask = goal_dist < self.cfg.dynamic_obstacle_goal_threshold
            if torch.any(reached_mask):
                reached_indices = dynamic_indices[reached_mask]
                self._resample_dynamic_obstacle_goals(reached_indices)
                self._resample_dynamic_obstacle_velocities(reached_indices)

        resample_steps = max(int(round(self.cfg.dynamic_obstacle_velocity_resample_interval_s / self.cfg.physics_dt)), 1)
        if self._dynamic_obstacle_step_count % resample_steps == 0:
            self._resample_dynamic_obstacle_velocities(dynamic_indices)

        self._obstacle_positions_w[dynamic_indices, :2] += self._dynamic_obstacle_velocity_xy[dynamic_indices] * float(dt)
        xy_limits = self.cfg.obstacle_spawn_range - self._obstacle_radii[dynamic_indices].unsqueeze(1)
        self._obstacle_positions_w[dynamic_indices, :2] = torch.maximum(
            torch.minimum(self._obstacle_positions_w[dynamic_indices, :2], xy_limits),
            -xy_limits,
        )
        self._obstacle_positions_w[dynamic_indices, 2] = self._dynamic_obstacle_origin_w[dynamic_indices, 2]

        self._dynamic_obstacle_step_count += 1
        self._sync_obstacle_prims()

    def _spawn_shared_obstacles(self) -> None:
        # 先采样障碍物尺寸，再按 curriculum 阶段决定哪些障碍物当前激活。
        sampled_radii, sampled_heights = self._sample_obstacles()
        self._obstacle_radii[:] = sampled_radii
        self._obstacle_heights[:] = sampled_heights
        self._apply_obstacle_layout(sync_prims=False)
        self._obstacle_paths.clear()
        self._obstacle_translate_ops.clear()
        dynamic_start = self.cfg.num_obstacles - self.cfg.num_dynamic_obstacles
        for idx in range(self.cfg.num_obstacles):
            prim_path = f"/World/SharedObstacles/Obstacle_{idx:03d}"
            is_dynamic_pool = idx >= dynamic_start
            obstacle_cfg = sim_utils.CylinderCfg(
                radius=float(self._obstacle_radii[idx].item()),
                height=float(self._obstacle_heights[idx].item()),
                axis="Z",
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.92, 0.18, 0.18) if is_dynamic_pool else (0.65, 0.65, 0.68), metallic=0.0
                ),
            )
            obstacle_cfg.func(prim_path, obstacle_cfg, translation=tuple(self._obstacle_positions_w[idx].tolist()))
            self._obstacle_paths.append(prim_path)
            self._obstacle_translate_ops.append(None)

    def _spawn_boundary_walls(self) -> None:
        # 生成四面边界墙，防止无人机飞出有效任务区域。
        wall_cfg = sim_utils.CuboidCfg(  # 所有墙体共享的基础配置模板。
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.55, 0.60), metallic=0.0),
        )
        half = self.cfg.wall_half_extent  # 墙体中心相对原点的偏移量。
        height_center = self.cfg.wall_height * 0.5  # 墙体中心高度。
        thickness = self.cfg.wall_thickness  # 墙厚。
        long_span = self.cfg.wall_half_extent * 2 + thickness  # 南北墙或东西墙的长边尺寸。
        wall_specs = [
            ("North", (long_span, thickness, self.cfg.wall_height), (0.0, half, height_center)),
            ("South", (long_span, thickness, self.cfg.wall_height), (0.0, -half, height_center)),
            ("East", (thickness, long_span, self.cfg.wall_height), (half, 0.0, height_center)),
            ("West", (thickness, long_span, self.cfg.wall_height), (-half, 0.0, height_center)),
        ]
        self._wall_paths.clear()  # 清空旧墙体路径。
        for name, size, translation in wall_specs:
            prim_path = f"/World/BoundaryWalls/{name}"  # 当前墙体 prim 路径。
            cfg = wall_cfg.replace(size=size)  # 在基础模板上替换当前墙体尺寸。
            cfg.func(prim_path, cfg, translation=translation)  # 生成墙体 prim。
            self._wall_paths.append(prim_path)  # 记录该墙体路径。

    def _build_robot_assets(self) -> None:
        # 按边缘采样的初始位置生成每一架无人机。
        robot_spawn_cfg = OMNINXT_CFG.replace(prim_path="/World/Robot_.*/OmniNxt")  # 创建可批量匹配多架无人机的资产配置。
        initial_spawn = self._sample_edge_positions_with_clearance(
            self.num_envs,
            self.cfg.spawn_edge_distance,
            self.cfg.spawn_min_height,
            self.cfg.spawn_max_height,
            min_separation=0.0,
        )
        for idx in range(self.num_envs):
            prim_path = f"/World/Robot_{idx:02d}/OmniNxt"  # 第 idx 架无人机的 prim 路径。
            robot_spawn_cfg.spawn.func(prim_path, robot_spawn_cfg.spawn, translation=tuple(initial_spawn[idx].tolist()))  # 生成该无人机实例。

        # 这里再创建一个 Articulation 视图对象，用于后续统一读取和写入所有无人机状态。
        robot_view_cfg = robot_spawn_cfg.copy()  # 复制一份配置给 Articulation 视图使用。
        robot_view_cfg.spawn = None  # 视图对象不再负责 spawn，避免重复创建。
        self._robot = Articulation(robot_view_cfg)  # 构建统一管理所有无人机的视图。

    def _disable_robot_collisions(self) -> None:
        # 关闭不同无人机之间以及无人机与环境之间的碰撞体，
        # 让任务把“障碍物风险”更多地交给 reward/termination 逻辑来定义。
        collision_cfg = sim_utils.CollisionPropertiesCfg(collision_enabled=False)  # 定义“禁用碰撞”的配置。
        for idx in range(self.num_envs):
            sim_schemas.modify_collision_properties(f"/World/Robot_{idx:02d}/OmniNxt", collision_cfg)  # 对每一架无人机应用禁碰撞配置。

    def _compute_closest_static_obstacles_directional(self) -> tuple[torch.Tensor, torch.Tensor]:
        drone_pos_w = self.robot.data.root_pos_w
        drone_quat_w = self.robot.data.root_quat_w
        static_indices = torch.nonzero(~self._dynamic_obstacle_mask, as_tuple=False).squeeze(-1)
        if static_indices.numel() == 0:
            directions = torch.zeros((self.num_envs, self.cfg.num_closest_obstacles, 3), device=self.device)
            distances = torch.ones((self.num_envs, self.cfg.num_closest_obstacles), device=self.device)
            return directions, distances

        static_positions_w = self._obstacle_positions_w[static_indices]
        static_radii = self._obstacle_radii[static_indices]
        to_obstacles = static_positions_w.unsqueeze(0) - drone_pos_w.unsqueeze(1)
        distances_sq = torch.sum(torch.square(to_obstacles), dim=2)
        k = min(self.cfg.num_closest_obstacles, static_indices.numel())
        _, closest_local = torch.topk(distances_sq, k, dim=1, largest=False)
        if k < self.cfg.num_closest_obstacles:
            pad = closest_local[:, -1:].expand(-1, self.cfg.num_closest_obstacles - k)
            closest_local = torch.cat([closest_local, pad], dim=1)
        batch_indices = torch.arange(self.num_envs, device=self.device).unsqueeze(1).expand(-1, self.cfg.num_closest_obstacles)
        closest_vectors = to_obstacles[batch_indices, closest_local]
        closest_radii = static_radii[closest_local]
        closest_distances = torch.sqrt(distances_sq[batch_indices, closest_local].clamp_min(1e-12)) - closest_radii
        closest_vectors_norm = closest_vectors / torch.sqrt(torch.sum(torch.square(closest_vectors), dim=2, keepdim=True).clamp_min(1e-12))

        num_closest = self.cfg.num_closest_obstacles
        closest_vectors_flat = closest_vectors_norm.reshape(self.num_envs * num_closest, 3)
        drone_quat_expanded = drone_quat_w.unsqueeze(1).expand(-1, num_closest, -1).reshape(self.num_envs * num_closest, 4)
        directions_body = quat_apply_inverse(drone_quat_expanded, closest_vectors_flat).reshape(self.num_envs, num_closest, 3)
        distances_normalized = (closest_distances / self.cfg.obstacle_detection_range).clamp(0.0, 1.0)
        return directions_body, distances_normalized

    def _compute_closest_dynamic_obstacle_features_body(self, drone_quat_w: torch.Tensor) -> torch.Tensor:
        dynamic_indices = torch.nonzero(self._dynamic_obstacle_mask, as_tuple=False).squeeze(-1)
        num_dyn = self.cfg.num_closest_dynamic_obstacles
        if dynamic_indices.numel() == 0:
            return torch.zeros((self.num_envs, num_dyn, 10), device=self.device)

        drone_pos_w = self.robot.data.root_pos_w
        drone_lin_vel_w = self.robot.data.root_lin_vel_w
        dyn_positions_w = self._obstacle_positions_w[dynamic_indices]
        dyn_radii = self._obstacle_radii[dynamic_indices]
        dyn_heights = self._obstacle_heights[dynamic_indices]
        dyn_velocities_w = torch.zeros((dynamic_indices.numel(), 3), device=self.device)
        dyn_velocities_w[:, :2] = self._dynamic_obstacle_velocity_xy[dynamic_indices]

        rel_vectors_w = dyn_positions_w.unsqueeze(0) - drone_pos_w.unsqueeze(1)
        distances_2d = torch.linalg.norm(rel_vectors_w[..., :2], dim=2)
        k = min(num_dyn, dynamic_indices.numel())
        _, closest_local = torch.topk(distances_2d, k, dim=1, largest=False)
        if k < num_dyn:
            pad = closest_local[:, -1:].expand(-1, num_dyn - k)
            closest_local = torch.cat([closest_local, pad], dim=1)

        batch_indices = torch.arange(self.num_envs, device=self.device).unsqueeze(1).expand(-1, num_dyn)
        closest_rel_w = rel_vectors_w[batch_indices, closest_local]
        closest_dist_total = torch.linalg.norm(closest_rel_w, dim=2, keepdim=True)
        closest_dir_w = closest_rel_w / closest_dist_total.clamp_min(1e-6)

        quat_expanded = drone_quat_w.unsqueeze(1).expand(-1, num_dyn, -1).reshape(self.num_envs * num_dyn, 4)
        closest_dir_b = quat_apply_inverse(quat_expanded, closest_dir_w.reshape(self.num_envs * num_dyn, 3)).reshape(self.num_envs, num_dyn, 3)

        closest_dist_2d = torch.linalg.norm(closest_rel_w[..., :2], dim=2, keepdim=True) / self.cfg.obstacle_detection_range
        closest_dist_z = torch.abs(closest_rel_w[..., 2:3]) / max(self.cfg.obstacle_height_max, 1.0)

        closest_dyn_vel_w = dyn_velocities_w[closest_local]
        rel_vel_w = closest_dyn_vel_w - drone_lin_vel_w.unsqueeze(1)
        rel_vel_b = quat_apply_inverse(quat_expanded, rel_vel_w.reshape(self.num_envs * num_dyn, 3)).reshape(self.num_envs, num_dyn, 3)
        rel_vel_scale = max(self.cfg.cmd_body_vel_xy_max + self.cfg.dynamic_obstacle_max_speed, self.cfg.cmd_vel_z_max, 1.0)
        rel_vel_b = (rel_vel_b / rel_vel_scale).clamp(-2.0, 2.0)

        closest_width = (2.0 * dyn_radii[closest_local]).unsqueeze(-1) / (2.0 * self.cfg.obstacle_radius_max)
        closest_height = dyn_heights[closest_local].unsqueeze(-1) / max(self.cfg.obstacle_height_max, 1.0)

        return torch.cat([closest_dir_b, closest_dist_2d.clamp(0.0, 1.0), closest_dist_z.clamp(0.0, 1.0), rel_vel_b, closest_width.clamp(0.0, 1.5), closest_height.clamp(0.0, 1.5)], dim=2)

    def _compute_closest_obstacle_signed_distance(self, drone_pos_w: torch.Tensor) -> torch.Tensor:
        # 计算无人机到最近障碍物表面的有符号距离 (Signed Distance SDF)。
        # SDF 的特性：正数表示在障碍物外，负数表示已穿模进障碍物内部
        rel_xy = drone_pos_w[:, None, :2] - self._obstacle_positions_w[None, :, :2]
        rel_z = torch.abs(drone_pos_w[:, None, 2] - self._obstacle_positions_w[None, :, 2])
        
        # 减去障碍物的半径和高度的一半，得到表面距离
        radial = torch.linalg.norm(rel_xy, dim=2) - self._obstacle_radii[None, :]
        vertical = rel_z - self._obstacle_heights[None, :] * 0.5
        
        # 经典的 SDF 圆柱体计算公式：
        # 如果都在外部，取两个分量的欧氏距离；如果部分或全部在内部，取最大值的负数
        outside = torch.stack((radial, vertical), dim=-1).clamp_min(0.0)
        inside = torch.maximum(radial, vertical).clamp_max(0.0)
        signed_distance = torch.linalg.norm(outside, dim=-1) + inside
        
        # 取所有障碍物中 SDF 值最小的一个，即为“最近威胁”
        return signed_distance.min(dim=1).values

    def _compute_target_ray_obstacle_distance(self, drone_pos_w: torch.Tensor, target_direction_w: torch.Tensor) -> torch.Tensor:
        # 评估“目标方向是否被障碍物遮挡”。这个特征能让网络学会判断是否需要“绕路”
        rel_vectors_w = self._obstacle_positions_w.unsqueeze(0) - drone_pos_w.unsqueeze(1)
        rel_distances = torch.sqrt(torch.sum(torch.square(rel_vectors_w), dim=2).clamp_min(1e-12))
        
        # 计算障碍物相对向量与目标方向的余弦相似度 (点积)
        target_alignment = torch.sum(rel_vectors_w * target_direction_w.unsqueeze(1), dim=2) / rel_distances.clamp_min(1e-6)
        
        # 划定一个锥形区域：余弦值 > 0.75（大约 41 度视角），并且在前方
        in_target_cone = (target_alignment > 0.75) & (torch.sum(rel_vectors_w * target_direction_w.unsqueeze(1), dim=2) > 0.0)
        surface_distances = rel_distances - self._obstacle_radii.unsqueeze(0)
        
        # 仅保留锥形区域内的障碍物距离
        target_ray_distances = torch.where(
            in_target_cone,
            surface_distances,
            torch.full_like(surface_distances, self.cfg.obstacle_detection_range),
        )
        return target_ray_distances.min(dim=1).values.clamp(0.0, self.cfg.obstacle_detection_range)

    def _compute_wall_signed_distance(self, drone_pos_w: torch.Tensor) -> torch.Tensor:
        # 计算到边界墙的 signed distance。
        # 这里把无人机当作点，只按位置和地图边界关系来估算。
        z_clearance = torch.minimum(
            drone_pos_w[:, 2],
            torch.full_like(drone_pos_w[:, 2], self.cfg.wall_height) - drone_pos_w[:, 2],
        )
        z_valid = z_clearance >= 0.0
        dx = self.cfg.wall_half_extent - torch.abs(drone_pos_w[:, 0])
        dy = self.cfg.wall_half_extent - torch.abs(drone_pos_w[:, 1])
        wall_band_half = self.cfg.wall_thickness * 0.5
        dist_x_wall = torch.abs(dx) - wall_band_half
        dist_y_wall = torch.abs(dy) - wall_band_half

        x_inside_span = torch.abs(drone_pos_w[:, 1]) <= self.cfg.wall_half_extent + wall_band_half
        y_inside_span = torch.abs(drone_pos_w[:, 0]) <= self.cfg.wall_half_extent + wall_band_half

        valid_x = z_valid & x_inside_span
        valid_y = z_valid & y_inside_span
        inf = torch.full_like(dist_x_wall, float("inf"))
        dist_x_wall = torch.where(valid_x, dist_x_wall, inf)
        dist_y_wall = torch.where(valid_y, dist_y_wall, inf)
        return torch.minimum(dist_x_wall, dist_y_wall)

    def _compute_wall_distances_xy(self, drone_pos_w: torch.Tensor) -> torch.Tensor:
        wall_half = self.cfg.wall_half_extent
        distances = torch.stack(
            [
                wall_half - drone_pos_w[:, 0],  # east
                wall_half + drone_pos_w[:, 0],  # west
                wall_half - drone_pos_w[:, 1],  # north
                wall_half + drone_pos_w[:, 1],  # south
            ],
            dim=1,
        )
        return (distances / wall_half).clamp(-1.0, 1.5)

    def _get_critic_observations(
        self,
        root_pos_w: torch.Tensor,
        root_quat_w: torch.Tensor,
        root_lin_vel_b: torch.Tensor,
        projected_gravity_b: torch.Tensor,
        target_pos_b: torch.Tensor,
    ) -> torch.Tensor:
        static_pool_count = max(self.cfg.num_obstacles - self.cfg.num_dynamic_obstacles, 0)
        static_k = min(8, static_pool_count)
        if static_k > 0:
            static_positions_w = self._obstacle_positions_w[:static_pool_count]
            static_rel_xy_w = static_positions_w.unsqueeze(0)[:, :, :2] - root_pos_w[:, None, :2]
            static_dist_sq = torch.sum(torch.square(static_rel_xy_w), dim=2)
            _, static_idx = torch.topk(static_dist_sq, static_k, dim=1, largest=False)
            batch_idx = torch.arange(self.num_envs, device=self.device).unsqueeze(1).expand(-1, static_k)
            closest_static_rel_xy_w = static_rel_xy_w[batch_idx, static_idx]
            closest_static_rel_xy = (closest_static_rel_xy_w / self.cfg.obstacle_spawn_range).clamp(-1.5, 1.5)
            closest_static_radii = (self._obstacle_radii[:static_pool_count][static_idx] / self.cfg.obstacle_radius_max).clamp(0.0, 1.5)
            static_features = torch.cat([closest_static_rel_xy, closest_static_radii.unsqueeze(-1)], dim=2).reshape(self.num_envs, -1)
        else:
            static_features = torch.zeros((self.num_envs, 0), device=self.device)

        dynamic_start = static_pool_count
        dynamic_indices = torch.arange(dynamic_start, self.cfg.num_obstacles, device=self.device, dtype=torch.long)
        dynamic_k = min(5, dynamic_indices.numel())
        if dynamic_k > 0:
            dyn_rel_pos_w_all = self._obstacle_positions_w[dynamic_indices].unsqueeze(0) - root_pos_w.unsqueeze(1)
            dyn_dist_sq = torch.sum(torch.square(dyn_rel_pos_w_all), dim=2)
            _, dyn_local_idx = torch.topk(dyn_dist_sq, dynamic_k, dim=1, largest=False)
            batch_idx = torch.arange(self.num_envs, device=self.device).unsqueeze(1).expand(-1, dynamic_k)
            selected_dynamic_indices = dynamic_indices[dyn_local_idx]
            dyn_rel_pos_w = dyn_rel_pos_w_all[batch_idx, dyn_local_idx]
            dyn_rel_pos_b = quat_apply_inverse(
                root_quat_w.unsqueeze(1).expand(-1, dynamic_k, -1).reshape(self.num_envs * dynamic_k, 4),
                dyn_rel_pos_w.reshape(self.num_envs * dynamic_k, 3),
            ).reshape(self.num_envs, dynamic_k, 3)
            dyn_rel_pos_b = (dyn_rel_pos_b / self.cfg.obstacle_detection_range).clamp(-1.5, 1.5)

            dyn_vel_w = torch.zeros((self.cfg.num_obstacles, 3), device=self.device)
            dyn_vel_w[:, :2] = self._dynamic_obstacle_velocity_xy
            dyn_rel_vel_w = dyn_vel_w[selected_dynamic_indices] - self.robot.data.root_lin_vel_w.unsqueeze(1)
            dyn_rel_vel_b = quat_apply_inverse(
                root_quat_w.unsqueeze(1).expand(-1, dynamic_k, -1).reshape(self.num_envs * dynamic_k, 4),
                dyn_rel_vel_w.reshape(self.num_envs * dynamic_k, 3),
            ).reshape(self.num_envs, dynamic_k, 3)
            vel_scale = max(self.cfg.cmd_body_vel_xy_max + self.cfg.dynamic_obstacle_max_speed, self.cfg.cmd_vel_z_max, 1.0)
            dyn_rel_vel_b = (dyn_rel_vel_b / vel_scale).clamp(-2.0, 2.0)

            dyn_radii = (self._obstacle_radii[selected_dynamic_indices] / self.cfg.obstacle_radius_max).clamp(0.0, 1.5)
            dyn_heights = (self._obstacle_heights[selected_dynamic_indices] / self.cfg.obstacle_height_max).clamp(0.0, 1.5)
            dyn_size = torch.stack([dyn_radii, dyn_heights], dim=2)
            dynamic_features = torch.cat([dyn_rel_pos_b, dyn_rel_vel_b, dyn_size], dim=2).reshape(self.num_envs, -1)
        else:
            dynamic_features = torch.zeros((self.num_envs, 0), device=self.device)

        return torch.cat(
            [
                root_lin_vel_b,
                projected_gravity_b,
                target_pos_b,
                self._cmd_vel_b,
                static_features,
                dynamic_features,
            ],
            dim=1,
        )

    def _update_prev_target_distance(self, env_ids: torch.Tensor) -> None:
        # 每次 reset 或 step 后，都维护一下“上一步到目标距离”，供 progress reward 使用。
        distances = torch.linalg.norm(self._target_positions_w[env_ids] - self.robot.data.root_pos_w[env_ids], dim=1)
        self._prev_dist_to_target[env_ids] = distances

    def _get_observations(self) -> dict[str, torch.Tensor]:
        # teacher 的观测是完整的特权低维状态，不包含图像。
        root_pos_w = self.robot.data.root_pos_w
        root_quat_w = self.robot.data.root_quat_w
        
        # 线速度统一转到机体系，策略看到的是“自身坐标系下”的运动状态。
        root_lin_vel_b = quat_apply_inverse(root_quat_w, self.robot.data.root_lin_vel_w)
        
        # projected_gravity_b 可以理解成一种紧凑姿态表示：
        # 如果机体水平，它大约会接近固定方向；若有 roll/pitch 偏转，这个向量也会跟着偏。
        projected_gravity_b = quat_apply_inverse(root_quat_w, self._gravity_vec_w)
        
        # 目标相对位置先在世界系求差，再转到机体系。
        target_vec_w = self._target_positions_w - root_pos_w
        target_pos_b = quat_apply_inverse(root_quat_w, target_vec_w)
        # 归一化后的目标方向，后面给“目标方向障碍距离”用。
        target_direction_w = target_vec_w / torch.linalg.norm(target_vec_w, dim=1, keepdim=True).clamp_min(1e-6)

        # 静态障碍和动态障碍分开编码，更接近 NavRL 的动态障碍表达方式。
        static_obstacle_directions, static_obstacle_distances = self._compute_closest_static_obstacles_directional()
        dynamic_obstacle_features = self._compute_closest_dynamic_obstacle_features_body(root_quat_w)
        target_ray_obstacle_distance = self._compute_target_ray_obstacle_distance(root_pos_w, target_direction_w)
        # 最终把所有低维量拼成一个大向量：
        # [自身速度, 姿态, 目标相对位置, 当前速度命令,
        #  最近静态障碍方向/距离, 最近动态障碍结构化特征, 目标方向障碍距离]
        obs = torch.cat(
            [
                root_lin_vel_b,
                projected_gravity_b,
                target_pos_b,
                self._cmd_vel_b,
                static_obstacle_directions.reshape(self.num_envs, -1),
                static_obstacle_distances,
                dynamic_obstacle_features.reshape(self.num_envs, -1),
                (target_ray_obstacle_distance / self.cfg.obstacle_detection_range).unsqueeze(1),
            ],
            dim=-1,
        )
        critic_obs = self._get_critic_observations(
            root_pos_w=root_pos_w,
            root_quat_w=root_quat_w,
            root_lin_vel_b=root_lin_vel_b,
            projected_gravity_b=projected_gravity_b,
            target_pos_b=target_pos_b,
        )
        return {"policy": obs, "critic": critic_obs}

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        # gymnasium reset 接口。首次 reset 会触发整个仿真世界的 build。
        del options
        if seed is not None:
            self.seed(seed)
        if not self._built:
            self._build()
        env_ids = torch.arange(self.num_envs, device=self.device)
        obs = self._reset_idx(env_ids)
        return obs, {}

    def _reset_idx(self, env_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        # 根据传入的 env_ids (张量)，局部重置发生碰撞或超时的环境。
        # 非常关键：这保证了不需要等所有飞机死掉才重置，大幅提升采样吞吐量。
        if env_ids.numel() == 0:
            return self._get_observations()

        # active_ids 表示当前不参与 reset 的 env，用它们的位置来避免新出生点和正在运行的无人机撞在一起。
        active_ids = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        active_ids[env_ids] = False
        avoid_positions_xy = self.robot.data.root_pos_w[active_ids, :2] if self._built else None

        # 起点：从边缘采样，并保证多架无人机之间有最小分离距离。
        start_pos = self._sample_edge_positions_with_clearance(
            len(env_ids),
            self.cfg.spawn_edge_distance,
            self.cfg.spawn_min_height,
            self.cfg.spawn_max_height,
            min_separation=0.0,
            avoid_positions_xy=avoid_positions_xy,
        )
        # 先判断起点在哪一条边，再从对边采样目标点，让任务更像“横穿场地”。
        start_side_indices = self._infer_edge_side_indices(start_pos)
        target_pos = self._sample_opposite_edge_positions_with_clearance(
            start_side_indices,
            self.cfg.target_min_height,
            self.cfg.target_max_height,
            min_separation=0.0,
            avoid_obstacles=True,
        )

        # default_root_state 是资产默认 root state 的模板，
        # 这里在它的基础上把位置、朝向、速度重写成新的 episode 初始状态。
        root_state = self.robot.data.default_root_state.clone()
        diff = target_pos - start_pos
        # 让出生时机头大致朝向目标方向，减轻控制器初始瞬态。
        yaw = torch.atan2(diff[:, 1], diff[:, 0])
        cy = torch.cos(yaw * 0.5)
        sy = torch.sin(yaw * 0.5)

        root_state[env_ids, :3] = start_pos
        root_state[env_ids, 3] = cy
        root_state[env_ids, 4] = 0.0
        root_state[env_ids, 5] = 0.0
        root_state[env_ids, 6] = sy
        # 线速度和角速度一律清零，从静止开始新 episode。
        root_state[env_ids, 7:] = 0.0

        # 把新的根状态写回仿真。
        self.robot.write_root_state_to_sim(root_state[env_ids], env_ids=env_ids)
        self.robot.write_joint_state_to_sim(
            self.robot.data.default_joint_pos[env_ids],
            self.robot.data.default_joint_vel[env_ids],
            env_ids=env_ids,
        )

        # 同步更新与任务相关的所有“逻辑状态量”。
        self._target_positions_w[env_ids] = target_pos
        self._target_reached[env_ids] = False
        self._done_buf[env_ids] = False
        self._rew_buf[env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0
        self._cmd_vel_b[env_ids] = 0.0
        self._actions[env_ids] = 0.0
        self._prev_actions[env_ids] = 0.0

        # ★ 重置底层控制器的内部积分器 (I-term) 和状态，防止“带着历史偏差”开始新回合
        controller_state = {
            # 这里传给 controller 的是全部 env 的状态，
            # 但 controller 只会按 env_ids 重置指定那部分内部状态。
            "position": root_state[:, :3].clone(),
            "attitude": _quat_to_euler_deg(root_state[:, 3:7]),
        }
        self._controller.reset(controller_state, env_ids=env_ids)
        self._controller.set_velocity_setpoint(
            vx=torch.zeros(len(env_ids), device=self.device),
            vy=torch.zeros(len(env_ids), device=self.device),
            vz=torch.zeros(len(env_ids), device=self.device),
            velocity_body=True,
            env_ids=env_ids,
        )

        # Refresh local robot buffers after writing the reset state without advancing physics.
        self.robot.update(0.0)
        # progress reward 从新的起点开始累计，所以这里重置“上一时刻距离”。
        self._update_prev_target_distance(env_ids)
        for key in self._episode_sums:
            self._episode_sums[key][env_ids] = 0.0
        return self._get_observations()

    def step(self, actions: torch.Tensor):
        # 这里的 actions 不是电机推力，而是高层速度指令：
        # [body_vx_cmd, body_vy_cmd, world_vz_cmd]
        if not self._built:
            raise RuntimeError("Call reset() before step().")

        actions = actions.to(self.device).clamp(-1.0, 1.0)
        prev_actions = self._prev_actions.clone()
        self._actions[:] = actions
        # 前两个动作维度解释为机体系水平速度命令。
        self._cmd_vel_b[:, :2] = actions[:, :2] * self.cfg.cmd_body_vel_xy_max
        # 第三个维度解释为竖直速度命令。
        self._cmd_vel_b[:, 2] = actions[:, 2] * self.cfg.cmd_vel_z_max

        # 把速度指令送进 Crazyflie controller，控制器输出力和力矩。
        root_quat_w = self.robot.data.root_quat_w
        # 控制器需要角速度的机体系表达。
        root_ang_vel_b = quat_apply_inverse(root_quat_w, self.robot.data.root_ang_vel_w)
        
        # 给底层控制器下发速度期望值
        self._controller.set_velocity_setpoint(
            vx=self._cmd_vel_b[:, 0],
            vy=self._cmd_vel_b[:, 1],
            vz=self._cmd_vel_b[:, 2],
            velocity_body=True,
        )
        # 控制器计算出期望的 Force (推力) 和 Torque (扭矩)
        force, torque = self._controller.compute(
            {
                # 位置/速度用世界系，姿态/角速度按控制器的接口要求转成欧拉角与角速度角度制。
                "position": self.robot.data.root_pos_w,
                "velocity": self.robot.data.root_lin_vel_w,
                "attitude": _quat_to_euler_deg(root_quat_w),
                "angular_velocity": torch.rad2deg(root_ang_vel_b),
            }
        )
        # controller 输出的结果会直接作为外力和外力矩施加到机体 body 上。
        self._thrust[:, 0, :] = force
        self._moment[:, 0, :] = torque

        for _ in range(self.cfg.decimation):
            # 一个 RL step 内进行若干次 physics step，使控制更稳定。
            # 将力矩直接施加在 Base Link 上，然后步进物理引擎
            self.robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)
            self.robot.write_data_to_sim()
            self._update_dynamic_obstacles(self.cfg.physics_dt)
            self.sim.step()
            self.robot.update(self.cfg.physics_dt)

        self.common_step_counter += 1
        self.episode_length_buf += 1

        # ==================== reward 计算 (奖励塑形) ====================
        root_quat_w = self.robot.data.root_quat_w
        root_lin_vel_b = quat_apply_inverse(root_quat_w, self.robot.data.root_lin_vel_w)
        target_vec_w = self._target_positions_w - self.robot.data.root_pos_w
        # 当前到目标的欧氏距离。
        distance_to_target = torch.linalg.norm(target_vec_w, dim=1)
        target_direction = target_vec_w / distance_to_target.unsqueeze(1).clamp_min(1e-6)

        # 1. 距离奖励：离目标越近越高，但用 tanh 压缩，避免太远时梯度过大。
        distance_reward = 1.0 - torch.tanh(distance_to_target / 2.0)
        
        # 2. 目标方向速度奖励：鼓励无人机真的朝目标方向前进，而不是只在原地稳定。
        target_velocity = torch.sum(self.robot.data.root_lin_vel_w * target_direction, dim=1).clamp(-1.0, 3.0)
        
        # 3. progress 奖励：相比上一时刻是否更接近目标。
        progress_reward = (self._prev_dist_to_target - distance_to_target).clamp(-1.0, 1.0)
        self._prev_dist_to_target = distance_to_target.clone()

        # 4. 平滑性惩罚：动作变化过快会被惩罚，鼓励更平顺的控制输出。
        action_delta = self._actions - prev_actions
        action_smoothness = torch.sum(torch.square(action_delta), dim=1)
        self._prev_actions[:] = self._actions

        # 5. 第一次到达目标时给一个 bonus (稀疏大奖)，并锁存 target_reached。
        newly_reached = (distance_to_target < self.cfg.target_reach_threshold) & (~self._target_reached)
        target_bonus = torch.zeros(self.num_envs, device=self.device)
        target_bonus[newly_reached] = self.cfg.target_reached_bonus
        self._target_reached |= newly_reached

        closest_obstacle_distance = self._compute_closest_obstacle_signed_distance(self.robot.data.root_pos_w)
        closest_wall_distance = self._compute_wall_signed_distance(self.robot.data.root_pos_w)

        # 6. 障碍物和墙都属于 hazard，取两者里更危险的那个距离，作为惩罚基准
        closest_hazard_distance = torch.minimum(closest_obstacle_distance, closest_wall_distance)
        obstacle_proximity = torch.where(
            closest_hazard_distance < self.cfg.obstacle_proximity_trigger_distance,
            torch.exp(-closest_hazard_distance * 3.0),
            torch.zeros_like(closest_hazard_distance),
        )

        # 7. 动态障碍物安全奖励 + TTC 提前惩罚。
        to_obstacles_w = self._obstacle_positions_w.unsqueeze(0) - self.robot.data.root_pos_w.unsqueeze(1)
        distances_sq_all = torch.sum(torch.square(to_obstacles_w), dim=2)
        obstacle_velocities_w = torch.zeros((self.cfg.num_obstacles, 3), device=self.device)
        obstacle_velocities_w[:, :2] = self._dynamic_obstacle_velocity_xy

        dynamic_indices = torch.nonzero(self._dynamic_obstacle_mask, as_tuple=False).squeeze(-1)
        if dynamic_indices.numel() > 0:
            dyn_distances_sq = distances_sq_all[:, dynamic_indices]
            num_dyn_reward = min(self.cfg.num_closest_dynamic_obstacles, dynamic_indices.numel())
            _, closest_dyn_local = torch.topk(dyn_distances_sq, num_dyn_reward, dim=1, largest=False)
            if num_dyn_reward < self.cfg.num_closest_dynamic_obstacles:
                pad = closest_dyn_local[:, -1:].expand(-1, self.cfg.num_closest_dynamic_obstacles - num_dyn_reward)
                closest_dyn_local = torch.cat([closest_dyn_local, pad], dim=1)
            closest_dyn_indices = dynamic_indices[closest_dyn_local]
            dyn_batch_indices = torch.arange(self.num_envs, device=self.device).unsqueeze(1).expand(-1, self.cfg.num_closest_dynamic_obstacles)
            closest_dyn_vectors_w = to_obstacles_w[dyn_batch_indices, closest_dyn_indices]
            closest_dyn_center_distances = torch.sqrt(distances_sq_all[dyn_batch_indices, closest_dyn_indices].clamp_min(1e-12))
            closest_dyn_surface_distances = (closest_dyn_center_distances - self._obstacle_radii[closest_dyn_indices]).clamp_min(1e-3)
            dynamic_obstacle_safety = torch.log(
                closest_dyn_surface_distances.clamp(min=1e-6, max=self.cfg.obstacle_detection_range)
            ).mean(dim=1)

            closest_directions_w = closest_dyn_vectors_w / closest_dyn_center_distances.unsqueeze(-1).clamp_min(1e-6)
            closest_relative_velocities_w = obstacle_velocities_w[closest_dyn_indices] - self.robot.data.root_lin_vel_w.unsqueeze(1)
            closing_speed = -(closest_relative_velocities_w * closest_directions_w).sum(dim=2)
            time_to_collision = closest_dyn_surface_distances / closing_speed.clamp_min(1e-3)
            ttc_mask = (closing_speed > 0.05) & (time_to_collision < 3.0)
            time_to_collision_penalty = torch.where(
                ttc_mask,
                torch.exp(-time_to_collision).clamp_max(1.0),
                torch.zeros_like(time_to_collision),
            ).max(dim=1).values
        else:
            dynamic_obstacle_safety = torch.zeros(self.num_envs, device=self.device)
            time_to_collision_penalty = torch.zeros(self.num_envs, device=self.device)

        # 把奖励按三类组织：目标推进 / 避障 / 平滑。
        rewards = {
            "distance_to_target": distance_reward * self.cfg.distance_to_target_reward_scale * self.step_dt,
            "target_velocity": target_velocity.clamp(min=0.0) * self.cfg.target_velocity_reward_scale * self.step_dt,
            "progress": progress_reward * self.cfg.progress_reward_scale * self.step_dt,
            "target_bonus": target_bonus,
            "obstacle_proximity": obstacle_proximity * self.cfg.obstacle_proximity_reward_scale * self.step_dt,
            "time_to_collision": time_to_collision_penalty * self.cfg.time_to_collision_reward_scale * self.step_dt,
            "dynamic_obstacle_safety": dynamic_obstacle_safety * self.cfg.dynamic_obstacle_safety_reward_scale * self.step_dt,
            "action_smoothness": action_smoothness * self.cfg.action_smoothness_reward_scale * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # 这里把每一项单独累到 episode_sums，后面 episode 结束时统一做统计。
        for key, value in rewards.items():
            self._episode_sums[key] += value
        self._rew_buf[:] = reward

        # ==================== 终止条件 (Termination / Truncation) ====================
        # 终止既包括“自然成功到达目标”，也包括各种死亡条件。
        timeout = self.episode_length_buf >= self.max_episode_length # Truncation (时间耗尽)
        obstacle_collision = closest_obstacle_distance < self.cfg.obstacle_collision_margin # Termination (撞毁)
        wall_collision = closest_wall_distance < self.cfg.obstacle_collision_margin
        too_low = self.robot.data.root_pos_w[:, 2] < 0.1 # 坠地
        too_high = self.robot.data.root_pos_w[:, 2] > 4.0 # 飞出安全高度
        
        died = too_low | too_high | obstacle_collision | wall_collision
        terminated = self._target_reached | died
        self._done_buf[:] = terminated | timeout

        # extras 里放额外诊断信息，供评估脚本和日志使用。
        extras = {
            "time_outs": timeout.clone(),
            "target_reached": self._target_reached.clone(),
            "terminated": terminated.clone(),
            "obstacle_collision": obstacle_collision.clone(),
            "wall_collision": wall_collision.clone(),
            "closest_obstacle_distance": closest_obstacle_distance.clone(),
            "closest_wall_distance": closest_wall_distance.clone(),
            "distance_to_target": distance_to_target.clone(),
        }

        if self.common_step_counter % 500 == 0:
            success_rate = self._target_reached.float().mean().item()
            print(
                f"[DEBUG] Step {self.common_step_counter}: "
                f"active_obstacles={self._current_total_obstacle_count}, dynamic={self._current_dynamic_obstacle_count}, "
                f"died={died.sum().item()}, success={self._target_reached.sum().item()}, "
                f"collision={(obstacle_collision | wall_collision).sum().item()}, success_rate={success_rate:.3f}"
            )

        done_ids = torch.nonzero(self._done_buf, as_tuple=False).squeeze(-1)
        if done_ids.numel() > 0:
            batch_success_rate = self._target_reached[done_ids].float().mean().item()
            episode_duration_s = (self.episode_length_buf[done_ids].float() * self.step_dt).clamp_min(self.step_dt)

            log = {}
            for key in self._episode_sums:
                # 用真实 episode 时长归一化，避免短回合把日志数值系统性压低。
                episodic_avg = (self._episode_sums[key][done_ids] / episode_duration_s).mean()
                log[f"Episode_Reward/{key}"] = episodic_avg.item()
            log["Episode_Termination/died_rate"] = died[done_ids].float().mean().item()
            log["Episode_Termination/time_out_rate"] = timeout[done_ids].float().mean().item()
            log["Metrics/success_rate"] = batch_success_rate
            log["Metrics/avg_closest_hazard_distance"] = closest_hazard_distance.mean().item()
            log["Metrics/active_obstacles"] = float(self._current_total_obstacle_count)
            log["Metrics/dynamic_obstacles"] = float(self._current_dynamic_obstacle_count)
            extras["log"] = log

        rew = self._rew_buf.clone()
        terminated_out = terminated.clone()
        truncated_out = timeout.clone()
        if self.cfg.auto_reset_done and done_ids.numel() > 0:
            # 默认自动 reset 已完成的并行环境，保持采样持续进行。
            self._reset_idx(done_ids)

        # 自动 reset 之后，再重新读取一次 observation，返回给 PPO 的就是“下一状态”。
        obs = self._get_observations()
        return obs, rew, terminated_out, truncated_out, extras

    def render(self):
        # 只有在 render_mode == "rgb_array" 时，才真的返回当前 viewer 的 RGB 图像。
        if not self._built:
            return None
        self.sim.render()
        if self.render_mode != "rgb_array":
            return None
            
        # 确保仿真处于支持渲染的模式
        if self.sim.render_mode.value < self.sim.RenderMode.PARTIAL_RENDERING.value:
            raise RuntimeError(
                "Cannot render 'rgb_array' when the simulation render mode does not support rendering."
            )
            
        if self._rgb_annotator is None:
            # 延迟初始化 (Lazy Initialization)：
            # 首次 render 时再延迟创建 replicator annotator，避免训练时额外开销。
            import omni.replicator.core as rep

            # 绑定虚拟相机路径，设置分辨率为 720p
            self._render_product = rep.create.render_product(
                self.cfg.viewer_cam_prim_path,
                (1280, 720),
            )
            # 获取 RGB 标注器并挂载到渲染产物上
            self._rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
            self._rgb_annotator.attach([self._render_product])
            
        # 提取当前帧的数据并转换为 NumPy 格式 (H, W, C)
        rgb_data = self._rgb_annotator.get_data()
        rgb_data = np.frombuffer(rgb_data, dtype=np.uint8).reshape(*rgb_data.shape)
        
        if rgb_data.size == 0:
            # 在 annotator 还没真正产出图像前，返回一张全黑图，避免上层崩掉。
            return np.zeros((720, 1280, 3), dtype=np.uint8)
            
        return rgb_data[:, :, :3]

    def close(self):
        # 清理 annotator、timeline、callbacks 和仿真实例，避免反复启动时资源泄露。
        # 不彻底清理会导致 GPU 显存泄漏 (Memory Leak)，下次再跑脚本就会 out of memory。
        if self._sim is None:
            return
            
        # 断开渲染标注器
        if self._rgb_annotator is not None and self._render_product is not None:
            try:
                self._rgb_annotator.detach([self._render_product])
            except Exception:
                pass
        self._rgb_annotator = None
        self._render_product = None
        
        # 停止时间线、清空回调、清理底层实例
        self._sim._timeline.stop()
        self._sim.clear_all_callbacks()
        self._sim.clear_instance()
        
        self._sim = None
        self._robot = None
        self._built = False
