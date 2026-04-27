from __future__ import annotations

from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import schemas as sim_schemas
from isaaclab.utils.math import quat_apply_inverse

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
    num_obstacles: int = 100
    num_closest_obstacles: int = 8
    obstacle_height: float = 4.0
    obstacle_radius: float = 0.15
    obstacle_spawn_range: float = 20.0
    obstacle_safe_zone: float = 1.0
    obstacle_min_separation: float = 0.6
    obstacle_detection_range: float = 6.0
    obstacle_proximity_trigger_distance: float = 2.0
    obstacle_collision_margin: float = 0.10
    target_obstacle_clearance: float = 2.0

    # ==================== 起点 / 终点采样配置 ====================
    spawn_edge_distance: float = 23.0
    target_spawn_range: float = 23.0
    spawn_min_height: float = 0.5
    spawn_max_height: float = 2.5
    target_min_height: float = 0.5
    target_max_height: float = 2.5
    target_reach_threshold: float = 0.5
    drone_spawn_min_separation: float = 1.0

    # ==================== 动作解释为速度指令的范围 ====================
    cmd_body_vel_xy_max: float = 3.0
    cmd_vel_z_max: float = 1.0

    # ==================== 奖励项权重 ====================
    vel_tracking_reward_scale: float = 0.5
    vel_tracking_exp_scale: float = 4.0
    ang_vel_reward_scale: float = -0.01
    distance_to_target_reward_scale: float = 12.0
    target_velocity_reward_scale: float = 3.0
    target_reached_bonus: float = 100.0
    obstacle_proximity_reward_scale: float = -12.0
    progress_reward_scale: float = 8.0

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
        # 机体速度/角速度/姿态、目标相对位置、当前速度命令、
        # 最近若干障碍物方向与距离、前向障碍距离、目标方向障碍距离、是否已到达目标。
        base_dim = 3 + 3 + 3 + 3 + 3
        closest_obstacle_dim = self.num_closest_obstacles * 3 + self.num_closest_obstacles
        extra_dim = 1 + 1 + 1
        return base_dim + closest_obstacle_dim + extra_dim

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
                )
            }
        )
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space, self.num_envs)

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
        self._obstacle_positions_w = torch.zeros((self.cfg.num_obstacles, 3), device=self.device)
        self._obstacle_heights = torch.full((self.cfg.num_obstacles,), self.cfg.obstacle_height, device=self.device)
        self._obstacle_radii = torch.full((self.cfg.num_obstacles,), self.cfg.obstacle_radius, device=self.device)
        self._obstacle_positions_xy = self._obstacle_positions_w[:, :2]
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._rew_buf = torch.zeros(self.num_envs, device=self.device)
        self._done_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._gravity_vec_w = torch.tensor([0.0, 0.0, -1.0], device=self.device).repeat(self.num_envs, 1)
        self._target_obstacle_clearance_sq = self.cfg.target_obstacle_clearance**2
        self._drone_spawn_min_separation_sq = self.cfg.drone_spawn_min_separation**2
        obstacle_min_center_distance = self.cfg.obstacle_min_separation + 2.0 * self.cfg.obstacle_radius
        self._obstacle_min_center_distance_sq = obstacle_min_center_distance**2
        self._obstacle_safe_radius_sq = (self.cfg.obstacle_safe_zone + self.cfg.obstacle_radius + 0.2) ** 2
        self._obstacle_collision_margin_sq = self.cfg.obstacle_collision_margin**2

        self._episode_sums = {
            # 这些值用于 TensorBoard 里按 episode 统计不同奖励项和诊断指标。
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "vel_tracking",
                "ang_vel",
                "distance_to_target",
                "target_velocity",
                "target_bonus",
                "obstacle_proximity",
                "progress",
                "vel_tracking_error",
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
        self._disable_robot_collisions()

        # reset 之后再读取 body id / mass 等信息，此时物理对象已经真正进入仿真。
        self.sim.reset()
        self._body_id = self.robot.find_bodies("body")[0]
        self._robot_mass = self.robot.root_physx_view.get_masses()[0].sum().item()
        self.robot.write_joint_state_to_sim(self.robot.data.default_joint_pos, self.robot.data.default_joint_vel)
        self.robot.update(self.cfg.physics_dt)
        for _ in range(5):
            # 先空跑几步，让 articulation 缓冲和物理状态稳定下来。
            self.sim.step()
            self.robot.update(self.cfg.physics_dt)

        print(
            f"[INFO] Quadcopter Obstacles Teacher - Single World with {self.num_envs} robots and "
            f"{self.cfg.num_obstacles} shared obstacles"
        )
        print(f"[INFO] Observation space: {self.cfg.observation_space}")
        print(f"[INFO] OmniNxt mass from asset: {self._robot_mass:.4f} kg")
        print(f"[INFO] Controller configured mass: {controller_config.CF_MASS:.4f} kg")
        if abs(self._robot_mass - controller_config.CF_MASS) > 0.05:
            print("[WARN] OmniNxt mass and controller CF_MASS differ significantly. Retune controller/config.py.")

        self._built = True

    def _sample_edge_positions(
        self, num_samples: int, lateral_range: float, min_height: float, max_height: float
    ) -> torch.Tensor:
        # 在地图四条边上采样出生点/目标点：
        # 两条 y=±spawn_edge_distance，另外两条 x=±spawn_edge_distance。
        side_indices = torch.randint(0, 4, (num_samples,), device=self.device)
        side_signs = torch.where(
            side_indices % 2 == 0,
            torch.ones(num_samples, device=self.device),
            -torch.ones(num_samples, device=self.device),
        )
        lateral = torch.empty(num_samples, device=self.device).uniform_(-lateral_range, lateral_range)
        heights = torch.empty(num_samples, device=self.device).uniform_(min_height, max_height)

        positions = torch.zeros((num_samples, 3), device=self.device)
        x_side_mask = side_indices < 2
        y_side_mask = ~x_side_mask
        positions[x_side_mask, 0] = lateral[x_side_mask]
        positions[x_side_mask, 1] = side_signs[x_side_mask] * self.cfg.spawn_edge_distance
        positions[y_side_mask, 0] = side_signs[y_side_mask] * self.cfg.spawn_edge_distance
        positions[y_side_mask, 1] = lateral[y_side_mask]
        positions[:, 2] = heights
        return positions

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
        # 带约束的边缘采样：
        # 既要与已有无人机保持间距，也可以要求避开障碍物或某些给定位置。
        positions = torch.zeros((num_samples, 3), device=self.device)
        obstacle_clearance = self.cfg.target_obstacle_clearance if obstacle_clearance is None else obstacle_clearance
        obstacle_clearance_sq = obstacle_clearance**2
        for idx in range(num_samples):
            best_candidate = None
            best_score = None
            for _ in range(128):
                # 多次尝试采样，优先找满足所有约束的点；若找不到，就退而求其次选得分最高的候选。
                candidate = self._sample_edge_positions(1, lateral_range, min_height, max_height)[0]
                valid = True
                score = torch.tensor(float("inf"), device=self.device)

                if min_separation > 0.0 and idx > 0:
                    d_prev_sq = torch.sum(torch.square(candidate.unsqueeze(0) - positions[:idx]), dim=1)
                    min_prev_dist_sq = d_prev_sq.min()
                    valid = bool(valid and (min_prev_dist_sq >= min_separation**2))
                    score = torch.minimum(score, min_prev_dist_sq)

                if avoid_positions_xy is not None and avoid_positions_xy.numel() > 0:
                    d_keep_sq = torch.sum(torch.square(candidate[:2].unsqueeze(0) - avoid_positions_xy), dim=1)
                    min_keep_dist_sq = d_keep_sq.min()
                    valid = bool(valid and (min_keep_dist_sq >= min_separation**2))
                    score = torch.minimum(score, min_keep_dist_sq)

                if avoid_obstacles and self.cfg.num_obstacles > 0:
                    d_obs_sq = torch.sum(torch.square(candidate[:2].unsqueeze(0) - self._obstacle_positions_xy), dim=1)
                    obstacle_clearance_sq_all = torch.square(obstacle_clearance + self._obstacle_radii)
                    valid_mask = d_obs_sq >= obstacle_clearance_sq_all
                    min_margin_sq = (d_obs_sq - obstacle_clearance_sq_all).min()
                    valid = bool(valid and bool(valid_mask.all()))
                    score = torch.minimum(score, min_margin_sq)

                if best_candidate is None or score > best_score:
                    best_candidate = candidate
                    best_score = score
                if valid:
                    best_candidate = candidate
                    break
            positions[idx] = best_candidate
        return positions

    def _infer_edge_side_indices(self, positions: torch.Tensor) -> torch.Tensor:
        # 根据 (x, y) 位置反推出当前点属于地图哪一条边，
        # 后面采样“对边目标点”时会用到。
        side_indices = torch.zeros(len(positions), dtype=torch.long, device=self.device)
        x_abs = torch.abs(positions[:, 0])
        y_abs = torch.abs(positions[:, 1])
        y_side_mask = y_abs >= x_abs
        side_indices[y_side_mask] = torch.where(
            positions[y_side_mask, 1] >= 0.0,
            torch.zeros_like(side_indices[y_side_mask]),
            torch.ones_like(side_indices[y_side_mask]),
        )
        x_side_mask = ~y_side_mask
        side_indices[x_side_mask] = torch.where(
            positions[x_side_mask, 0] >= 0.0,
            torch.full_like(side_indices[x_side_mask], 2),
            torch.full_like(side_indices[x_side_mask], 3),
        )
        return side_indices

    def _sample_opposite_edge_positions(self, source_side_indices: torch.Tensor, min_height: float, max_height: float) -> torch.Tensor:
        # 从起点所在边，采样到对边的目标点，鼓励任务具有“穿越地图”的结构。
        num_samples = len(source_side_indices)
        opposite_side_indices = torch.where(source_side_indices % 2 == 0, source_side_indices + 1, source_side_indices - 1)
        lateral = torch.empty(num_samples, device=self.device).uniform_(
            -self.cfg.spawn_edge_distance, self.cfg.spawn_edge_distance
        )
        heights = torch.empty(num_samples, device=self.device).uniform_(min_height, max_height)

        positions = torch.zeros((num_samples, 3), device=self.device)
        y_side_mask = opposite_side_indices < 2
        x_side_mask = ~y_side_mask
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
        positions[:, 2] = heights
        return positions

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
        positions = torch.zeros((len(source_side_indices), 3), device=self.device)
        obstacle_clearance = self.cfg.target_obstacle_clearance if obstacle_clearance is None else obstacle_clearance
        obstacle_clearance_sq = obstacle_clearance**2
        for idx in range(len(source_side_indices)):
            best_candidate = None
            best_score = None
            side_idx = source_side_indices[idx : idx + 1]
            for _ in range(128):
                candidate = self._sample_opposite_edge_positions(side_idx, min_height, max_height)[0]
                valid = True
                score = torch.tensor(float("inf"), device=self.device)

                if min_separation > 0.0 and idx > 0:
                    d_prev_sq = torch.sum(torch.square(candidate.unsqueeze(0) - positions[:idx]), dim=1)
                    min_prev_dist_sq = d_prev_sq.min()
                    valid = bool(valid and (min_prev_dist_sq >= min_separation**2))
                    score = torch.minimum(score, min_prev_dist_sq)

                if avoid_obstacles and self.cfg.num_obstacles > 0:
                    d_obs_sq = torch.sum(torch.square(candidate[:2].unsqueeze(0) - self._obstacle_positions_xy), dim=1)
                    obstacle_clearance_sq_all = torch.square(obstacle_clearance + self._obstacle_radii)
                    valid_mask = d_obs_sq >= obstacle_clearance_sq_all
                    min_margin_sq = (d_obs_sq - obstacle_clearance_sq_all).min()
                    valid = bool(valid and bool(valid_mask.all()))
                    score = torch.minimum(score, min_margin_sq)

                if best_candidate is None or score > best_score:
                    best_candidate = candidate
                    best_score = score
                if valid:
                    best_candidate = candidate
                    break
            positions[idx] = best_candidate
        return positions

    def _sample_obstacles(self) -> torch.Tensor:
        # 采样整张地图里共享的障碍物位置。
        # 所有并行无人机都在同一个世界里共享这批障碍物，因此这里不是 per-env 障碍物。
        placed_xy = torch.zeros((self.cfg.num_obstacles, 2), device=self.device)
        for obstacle_idx in range(self.cfg.num_obstacles):
            chosen_xy = None
            chosen_score = None
            for _ in range(64):
                candidate_xy = torch.empty(2, device=self.device).uniform_(
                    -self.cfg.obstacle_spawn_range, self.cfg.obstacle_spawn_range
                )
                candidate_radius_sq = torch.dot(candidate_xy, candidate_xy)
                valid = candidate_radius_sq >= self._obstacle_safe_radius_sq
                if obstacle_idx == 0:
                    min_dist = torch.tensor(float("inf"), device=self.device)
                else:
                    distances_sq = torch.sum(torch.square(candidate_xy.unsqueeze(0) - placed_xy[:obstacle_idx]), dim=1)
                    min_dist = distances_sq.min()
                    valid = bool(valid and (min_dist >= self._obstacle_min_center_distance_sq))

                if chosen_xy is None:
                    chosen_xy = candidate_xy
                    chosen_score = min_dist if valid else torch.tensor(-1.0, device=self.device)
                else:
                    candidate_score = min_dist if valid else torch.tensor(-1.0, device=self.device)
                    if candidate_score > chosen_score:
                        chosen_xy = candidate_xy
                        chosen_score = candidate_score
                if valid:
                    chosen_xy = candidate_xy
                    break
            placed_xy[obstacle_idx] = chosen_xy

        positions = torch.zeros((self.cfg.num_obstacles, 3), device=self.device)
        positions[:, :2] = placed_xy
        positions[:, 2] = self.cfg.obstacle_height * 0.5
        return positions

    def _spawn_shared_obstacles(self) -> None:
        # 先在 tensor 里采样障碍物中心，再把它们真正生成到 USD stage 中。
        self._obstacle_positions_w[:] = self._sample_obstacles()
        self._obstacle_paths.clear()
        for idx in range(self.cfg.num_obstacles):
            prim_path = f"/World/SharedObstacles/Obstacle_{idx:03d}"
            obstacle_cfg = sim_utils.CuboidCfg(
                size=(
                    self.cfg.obstacle_radius * 2.0,
                    self.cfg.obstacle_radius * 2.0,
                    self.cfg.obstacle_height,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2), metallic=0.0),
            )
            obstacle_cfg.func(prim_path, obstacle_cfg, translation=tuple(self._obstacle_positions_w[idx].tolist()))
            self._obstacle_paths.append(prim_path)

    def _spawn_boundary_walls(self) -> None:
        # 生成四面边界墙，防止无人机飞出有效任务区域。
        wall_cfg = sim_utils.CuboidCfg(
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.55, 0.60), metallic=0.0),
        )
        half = self.cfg.wall_half_extent
        height_center = self.cfg.wall_height * 0.5
        thickness = self.cfg.wall_thickness
        long_span = self.cfg.wall_half_extent * 2 + thickness
        wall_specs = [
            ("North", (long_span, thickness, self.cfg.wall_height), (0.0, half, height_center)),
            ("South", (long_span, thickness, self.cfg.wall_height), (0.0, -half, height_center)),
            ("East", (thickness, long_span, self.cfg.wall_height), (half, 0.0, height_center)),
            ("West", (thickness, long_span, self.cfg.wall_height), (-half, 0.0, height_center)),
        ]
        self._wall_paths.clear()
        for name, size, translation in wall_specs:
            prim_path = f"/World/BoundaryWalls/{name}"
            cfg = wall_cfg.replace(size=size)
            cfg.func(prim_path, cfg, translation=translation)
            self._wall_paths.append(prim_path)

    def _build_robot_assets(self) -> None:
        # 按边缘采样的初始位置生成每一架无人机。
        robot_spawn_cfg = OMNINXT_CFG.replace(prim_path="/World/Robot_.*/OmniNxt")
        initial_spawn = self._sample_edge_positions_with_clearance(
            self.num_envs,
            self.cfg.spawn_edge_distance,
            self.cfg.spawn_min_height,
            self.cfg.spawn_max_height,
            min_separation=self.cfg.drone_spawn_min_separation,
        )
        for idx in range(self.num_envs):
            prim_path = f"/World/Robot_{idx:02d}/OmniNxt"
            robot_spawn_cfg.spawn.func(prim_path, robot_spawn_cfg.spawn, translation=tuple(initial_spawn[idx].tolist()))

        # 这里再创建一个 Articulation 视图对象，用于后续统一读取和写入所有无人机状态。
        robot_view_cfg = robot_spawn_cfg.copy()
        robot_view_cfg.spawn = None
        self._robot = Articulation(robot_view_cfg)

    def _disable_robot_collisions(self) -> None:
        # 关闭不同无人机之间以及无人机与环境之间的碰撞体，
        # 让任务把“障碍物风险”更多地交给 reward/termination 逻辑来定义。
        collision_cfg = sim_utils.CollisionPropertiesCfg(collision_enabled=False)
        for idx in range(self.num_envs):
            sim_schemas.modify_collision_properties(f"/World/Robot_{idx:02d}/OmniNxt", collision_cfg)

    def _compute_closest_obstacles_directional(self) -> tuple[torch.Tensor, torch.Tensor]:
        # 计算每架无人机最近若干个障碍物的方向（机体系）和距离（归一化）。
        drone_pos_w = self.robot.data.root_pos_w
        drone_quat_w = self.robot.data.root_quat_w
        # 先在世界系里计算“无人机 -> 每个障碍物中心”的向量。
        to_obstacles = self._obstacle_positions_w.unsqueeze(0) - drone_pos_w.unsqueeze(1)
        # 用平方距离排序，避免过早开方。
        distances_sq = torch.sum(torch.square(to_obstacles), dim=2)

        # 每架无人机只保留最近的 num_closest_obstacles 个障碍物。
        _, closest_indices = torch.topk(distances_sq, self.cfg.num_closest_obstacles, dim=1, largest=False)
        batch_indices = torch.arange(self.num_envs, device=self.device).unsqueeze(1).expand(-1, self.cfg.num_closest_obstacles)
        closest_vectors = to_obstacles[batch_indices, closest_indices]
        closest_radii = self._obstacle_radii[closest_indices]
        # 距离不是到障碍物中心，而是到障碍物表面的距离。
        closest_distances = torch.sqrt(distances_sq[batch_indices, closest_indices].clamp_min(1e-12)) - closest_radii
        # 方向信息只保留单位向量，距离信息单独编码。
        closest_vectors_norm = closest_vectors / torch.sqrt(
            torch.sum(torch.square(closest_vectors), dim=2, keepdim=True).clamp_min(1e-12)
        )

        num_closest = self.cfg.num_closest_obstacles
        closest_vectors_flat = closest_vectors_norm.reshape(self.num_envs * num_closest, 3)
        # 用无人机当前姿态，把世界系方向向量旋转到机体系，便于策略理解“左/右/前/后”的空间关系。
        drone_quat_expanded = drone_quat_w.unsqueeze(1).expand(-1, num_closest, -1).reshape(self.num_envs * num_closest, 4)
        directions_body = quat_apply_inverse(drone_quat_expanded, closest_vectors_flat).reshape(
            self.num_envs, num_closest, 3
        )
        # 距离统一按 detection range 归一化到 [0, 1]。
        distances_normalized = (closest_distances / self.cfg.obstacle_detection_range).clamp(0.0, 1.0)
        return directions_body, distances_normalized

    def _compute_closest_obstacle_signed_distance(self, drone_pos_w: torch.Tensor) -> torch.Tensor:
        # 计算到最近障碍物表面的 signed distance。
        # 大于 0 表示还在障碍物外部，小于 0 表示已经“穿进”障碍物安全边界。
        rel_xy = drone_pos_w[:, None, :2] - self._obstacle_positions_w[None, :, :2]
        rel_z = torch.abs(drone_pos_w[:, None, 2] - self._obstacle_positions_w[None, :, 2])
        radial = torch.linalg.norm(rel_xy, dim=2) - self._obstacle_radii[None, :]
        vertical = rel_z - self._obstacle_heights[None, :] * 0.5
        outside = torch.stack((radial, vertical), dim=-1).clamp_min(0.0)
        inside = torch.maximum(radial, vertical).clamp_max(0.0)
        signed_distance = torch.linalg.norm(outside, dim=-1) + inside
        return signed_distance.min(dim=1).values

    def _compute_forward_obstacle_distance(self, drone_pos_w: torch.Tensor, drone_quat_w: torch.Tensor) -> torch.Tensor:
        # 只关注无人机正前方半空间里的最近障碍物距离。
        rel_vectors_w = self._obstacle_positions_w.unsqueeze(0) - drone_pos_w.unsqueeze(1)
        # 先把障碍物相对向量旋转到机体系，这样 x>0 就表示在机头前方。
        rel_vectors_b = quat_apply_inverse(
            drone_quat_w.unsqueeze(1).expand(-1, self.cfg.num_obstacles, -1).reshape(self.num_envs * self.cfg.num_obstacles, 4),
            rel_vectors_w.reshape(self.num_envs * self.cfg.num_obstacles, 3),
        ).reshape(self.num_envs, self.cfg.num_obstacles, 3)
        # 依然使用到障碍物表面的距离。
        surface_distances = torch.sqrt(torch.sum(torch.square(rel_vectors_w), dim=2).clamp_min(1e-12)) - self._obstacle_radii.unsqueeze(0)
        # 只保留前方障碍物，其它位置统一赋成 detection range，相当于“看不见威胁”。
        forward_mask = rel_vectors_b[:, :, 0] > 0.0
        forward_distances = torch.where(
            forward_mask,
            surface_distances,
            torch.full_like(surface_distances, self.cfg.obstacle_detection_range),
        )
        return forward_distances.min(dim=1).values.clamp(0.0, self.cfg.obstacle_detection_range)

    def _compute_target_ray_obstacle_distance(self, drone_pos_w: torch.Tensor, target_direction_w: torch.Tensor) -> torch.Tensor:
        # 看“朝目标方向”的锥形区域里最近的障碍物距离，用于编码目标方向上的可通行性。
        rel_vectors_w = self._obstacle_positions_w.unsqueeze(0) - drone_pos_w.unsqueeze(1)
        rel_distances = torch.sqrt(torch.sum(torch.square(rel_vectors_w), dim=2).clamp_min(1e-12))
        # 用点积余弦来衡量障碍物是否位于“朝目标的前方锥形区域”里。
        target_alignment = torch.sum(rel_vectors_w * target_direction_w.unsqueeze(1), dim=2) / rel_distances.clamp_min(1e-6)
        in_target_cone = (target_alignment > 0.75) & (torch.sum(rel_vectors_w * target_direction_w.unsqueeze(1), dim=2) > 0.0)
        surface_distances = rel_distances - self._obstacle_radii.unsqueeze(0)
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

    def _update_prev_target_distance(self, env_ids: torch.Tensor) -> None:
        # 每次 reset 或 step 后，都维护一下“上一步到目标距离”，供 progress reward 使用。
        distances = torch.linalg.norm(self._target_positions_w[env_ids] - self.robot.data.root_pos_w[env_ids], dim=1)
        self._prev_dist_to_target[env_ids] = distances

    def _get_observations(self) -> dict[str, torch.Tensor]:
        # teacher 的观测是完整的特权低维状态，不包含图像。
        root_pos_w = self.robot.data.root_pos_w
        root_quat_w = self.robot.data.root_quat_w
        # 线速度、角速度都统一转到机体系，策略看到的是“自身坐标系下”的运动状态。
        root_lin_vel_b = quat_apply_inverse(root_quat_w, self.robot.data.root_lin_vel_w)
        root_ang_vel_b = quat_apply_inverse(root_quat_w, self.robot.data.root_ang_vel_w)
        # projected_gravity_b 可以理解成一种紧凑姿态表示：
        # 如果机体水平，它大约会接近固定方向；若有 roll/pitch 偏转，这个向量也会跟着偏。
        projected_gravity_b = quat_apply_inverse(root_quat_w, self._gravity_vec_w)
        # 目标相对位置先在世界系求差，再转到机体系。
        target_vec_w = self._target_positions_w - root_pos_w
        target_pos_b = quat_apply_inverse(root_quat_w, target_vec_w)
        # 归一化后的目标方向，后面给“目标方向障碍距离”用。
        target_direction_w = target_vec_w / torch.linalg.norm(target_vec_w, dim=1, keepdim=True).clamp_min(1e-6)

        # 障碍物相关观测全部由环境真值直接算出来，这也是 teacher 能强于 student 的关键。
        obstacle_directions, obstacle_distances = self._compute_closest_obstacles_directional()
        forward_obstacle_distance = self._compute_forward_obstacle_distance(root_pos_w, root_quat_w)
        target_ray_obstacle_distance = self._compute_target_ray_obstacle_distance(root_pos_w, target_direction_w)
        # 最终把所有低维量拼成一个大向量：
        # [自身速度, 自身角速度, 姿态, 目标相对位置, 当前速度命令, 最近障碍物方向/距离, 前向障碍距离, 目标方向障碍距离, target_reached]
        obs = torch.cat(
            [
                root_lin_vel_b,
                root_ang_vel_b,
                projected_gravity_b,
                target_pos_b,
                self._cmd_vel_b,
                obstacle_directions.reshape(self.num_envs, -1),
                obstacle_distances,
                (forward_obstacle_distance / self.cfg.obstacle_detection_range).unsqueeze(1),
                (target_ray_obstacle_distance / self.cfg.obstacle_detection_range).unsqueeze(1),
                self._target_reached.float().unsqueeze(1),
            ],
            dim=-1,
        )
        return {"policy": obs}

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
        # 只重置指定 env_ids，对并行环境更高效。
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
            min_separation=self.cfg.drone_spawn_min_separation,
            avoid_positions_xy=avoid_positions_xy,
        )
        # 先判断起点在哪一条边，再从对边采样目标点，让任务更像“横穿场地”。
        start_side_indices = self._infer_edge_side_indices(start_pos)
        target_pos = self._sample_opposite_edge_positions_with_clearance(
            start_side_indices,
            self.cfg.target_min_height,
            self.cfg.target_max_height,
            min_separation=self.cfg.drone_spawn_min_separation,
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

        # controller 也要和 reset 后的姿态/位置一起重置，否则会带着上一局的积分状态。
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
        self._actions[:] = actions
        # 前两个动作维度解释为机体系水平速度命令。
        self._cmd_vel_b[:, :2] = actions[:, :2] * self.cfg.cmd_body_vel_xy_max
        # 第三个维度解释为竖直速度命令。
        self._cmd_vel_b[:, 2] = actions[:, 2] * self.cfg.cmd_vel_z_max

        # 把速度指令送进 Crazyflie controller，控制器输出力和力矩。
        root_quat_w = self.robot.data.root_quat_w
        # 控制器需要角速度的机体系表达。
        root_ang_vel_b = quat_apply_inverse(root_quat_w, self.robot.data.root_ang_vel_w)
        self._controller.set_velocity_setpoint(
            vx=self._cmd_vel_b[:, 0],
            vy=self._cmd_vel_b[:, 1],
            vz=self._cmd_vel_b[:, 2],
            velocity_body=True,
        )
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
            self.robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)
            self.robot.write_data_to_sim()
            self.sim.step()
            self.robot.update(self.cfg.physics_dt)

        self.common_step_counter += 1
        self.episode_length_buf += 1

        # ==================== reward 计算 ====================
        root_quat_w = self.robot.data.root_quat_w
        root_lin_vel_b = quat_apply_inverse(root_quat_w, self.robot.data.root_lin_vel_w)
        root_ang_vel_b = quat_apply_inverse(root_quat_w, self.robot.data.root_ang_vel_w)
        target_vec_w = self._target_positions_w - self.robot.data.root_pos_w
        # 当前到目标的欧氏距离。
        distance_to_target = torch.linalg.norm(target_vec_w, dim=1)
        target_direction = target_vec_w / distance_to_target.unsqueeze(1).clamp_min(1e-6)

        # 注意这里的速度跟踪误差不是直接拿 root_lin_vel_w 和 cmd 比，
        # 而是把水平速度用机体系，竖直速度用世界系 z，和动作定义保持一致。
        velocity_cmd_frame = torch.stack(
            [
                root_lin_vel_b[:, 0],
                root_lin_vel_b[:, 1],
                self.robot.data.root_lin_vel_w[:, 2],
            ],
            dim=1,
        )
        vel_tracking_sq_error = torch.sum(torch.square(self._cmd_vel_b - velocity_cmd_frame), dim=1)
        vel_tracking_error = torch.linalg.norm(self._cmd_vel_b - velocity_cmd_frame, dim=1)
        # 用指数形式把速度误差转成 [0,1] 附近的奖励，误差越小奖励越高。
        vel_tracking = torch.exp(-self.cfg.vel_tracking_exp_scale * vel_tracking_sq_error)
        # 角速度惩罚：转得越凶，惩罚越大。
        ang_vel = torch.sum(torch.square(root_ang_vel_b), dim=1)
        # 距离奖励：离目标越近越高，但用 tanh 压缩，避免太远时梯度过大。
        distance_reward = 1.0 - torch.tanh(distance_to_target / 2.0)
        # 目标方向速度奖励：鼓励无人机真的朝目标方向前进，而不是只在原地稳定。
        target_velocity = torch.sum(self.robot.data.root_lin_vel_w * target_direction, dim=1).clamp(-1.0, 3.0)
        # progress 奖励：相比上一时刻是否更接近目标。
        progress_reward = (self._prev_dist_to_target - distance_to_target).clamp(-1.0, 1.0)
        self._prev_dist_to_target = distance_to_target.clone()

        # 第一次到达目标时给一个 bonus，并锁存 target_reached。
        newly_reached = (distance_to_target < self.cfg.target_reach_threshold) & (~self._target_reached)
        target_bonus = torch.zeros(self.num_envs, device=self.device)
        target_bonus[newly_reached] = self.cfg.target_reached_bonus
        self._target_reached |= newly_reached

        closest_obstacle_distance = self._compute_closest_obstacle_signed_distance(self.robot.data.root_pos_w)
        closest_wall_distance = self._compute_wall_signed_distance(self.robot.data.root_pos_w)
        # 障碍物和墙都属于 hazard，取两者里更危险的那个距离。
        closest_hazard_distance = torch.minimum(closest_obstacle_distance, closest_wall_distance)
        obstacle_proximity = torch.where(
            closest_hazard_distance < self.cfg.obstacle_proximity_trigger_distance,
            torch.exp(-closest_hazard_distance * 3.0),
            torch.zeros_like(closest_hazard_distance),
        )

        # 把各项奖励拆开记录，方便后面在 TensorBoard 中判断到底是哪一项在拉高/拉低。
        rewards = {
            "vel_tracking": vel_tracking * self.cfg.vel_tracking_reward_scale * self.step_dt,
            "ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
            "distance_to_target": distance_reward * self.cfg.distance_to_target_reward_scale * self.step_dt,
            "target_velocity": target_velocity.clamp(min=0.0) * self.cfg.target_velocity_reward_scale * self.step_dt,
            "target_bonus": target_bonus,
            "obstacle_proximity": obstacle_proximity * self.cfg.obstacle_proximity_reward_scale * self.step_dt,
            "progress": progress_reward * self.cfg.progress_reward_scale * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # 这里把每一项单独累到 episode_sums，后面 episode 结束时统一做统计。
        for key, value in rewards.items():
            self._episode_sums[key] += value
        self._episode_sums["vel_tracking_error"] += vel_tracking_error * self.step_dt
        self._rew_buf[:] = reward

        # ==================== 终止条件 ====================
        # 终止既包括“自然成功到达目标”，也包括各种死亡条件。
        timeout = self.episode_length_buf >= self.max_episode_length
        obstacle_collision = closest_obstacle_distance < self.cfg.obstacle_collision_margin
        wall_collision = closest_wall_distance < self.cfg.obstacle_collision_margin
        too_low = self.robot.data.root_pos_w[:, 2] < 0.1
        too_high = self.robot.data.root_pos_w[:, 2] > 2.5
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
                f"[DEBUG] Step {self.common_step_counter}: died={died.sum().item()}, "
                f"success={self._target_reached.sum().item()}, collision={(obstacle_collision | wall_collision).sum().item()}, "
                f"success_rate={success_rate:.3f}"
            )

        done_ids = torch.nonzero(self._done_buf, as_tuple=False).squeeze(-1)
        if done_ids.numel() > 0:
            log = {}
            for key in self._episode_sums:
                # 把积累到 episode 结束时的总奖励除以 episode 时长，得到平均强度。
                episodic_sum_avg = torch.mean(self._episode_sums[key][done_ids])
                if key == "vel_tracking_error":
                    log["Metrics/avg_vel_tracking_error"] = episodic_sum_avg / self.max_episode_length_s
                else:
                    log[f"Episode_Reward/{key}"] = episodic_sum_avg / self.max_episode_length_s
            log["Episode_Termination/died"] = torch.count_nonzero(died[done_ids]).item()
            log["Episode_Termination/time_out"] = torch.count_nonzero(timeout[done_ids]).item()
            log["Metrics/success_rate"] = self._target_reached[done_ids].float().mean().item()
            log["Metrics/success_rate_all_envs"] = self._target_reached.float().mean().item()
            log["Metrics/avg_closest_hazard_distance"] = closest_hazard_distance.mean().item()
            extras["log"] = log

        rew = self._rew_buf.clone()
        terminated_out = terminated.clone()
        truncated_out = timeout.clone()
        if done_ids.numel() > 0 and self.cfg.auto_reset_done:
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
        if self.sim.render_mode.value < self.sim.RenderMode.PARTIAL_RENDERING.value:
            raise RuntimeError(
                "Cannot render 'rgb_array' when the simulation render mode does not support rendering."
            )
        if self._rgb_annotator is None:
            # 首次 render 时再延迟创建 replicator annotator，避免训练时额外开销。
            import omni.replicator.core as rep

            self._render_product = rep.create.render_product(
                self.cfg.viewer_cam_prim_path,
                (1280, 720),
            )
            self._rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
            self._rgb_annotator.attach([self._render_product])
        rgb_data = self._rgb_annotator.get_data()
        rgb_data = np.frombuffer(rgb_data, dtype=np.uint8).reshape(*rgb_data.shape)
        if rgb_data.size == 0:
            # 在 annotator 还没真正产出图像前，返回一张全黑图，避免上层崩掉。
            return np.zeros((720, 1280, 3), dtype=np.uint8)
        return rgb_data[:, :, :3]

    def close(self):
        # 清理 annotator、timeline、callbacks 和仿真实例，避免反复启动时资源泄露。
        if self._sim is None:
            return
        if self._rgb_annotator is not None and self._render_product is not None:
            try:
                self._rgb_annotator.detach([self._render_product])
            except Exception:
                pass
        self._rgb_annotator = None
        self._render_product = None
        self._sim._timeline.stop()
        self._sim.clear_all_callbacks()
        self._sim.clear_instance()
        self._sim = None
        self._robot = None
        self._built = False
