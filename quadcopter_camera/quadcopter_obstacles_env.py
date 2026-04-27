

# Copyright (c) 2026 Alex Jauregui & Erik Eguskiza.
# Stage 2 v6: Obstacles + Single Target Navigation
# 第二阶段版本6：带障碍物和单目标点导航的无人机环境

from __future__ import annotations

import gymnasium as gym
import torch
# 导入IsaacLab核心模块
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sensors.camera import TiledCamera, TiledCameraCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms, quat_apply_inverse
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import CUBOID_MARKER_CFG

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


@configclass
class QuadcopterObstaclesEnvCfg(DirectRLEnvCfg):
    """带障碍物和方向观测的单目标点环境配置类"""

    # Episode配置
    episode_length_s = 90.0
    decimation = 2

    # 障碍物配置
    num_obstacles = 100
    num_closest_obstacles = 5

    # 目标点配置
    target_reach_threshold = 0.5

    # 深度相机基础配置
    depth_camera_width = 32
    depth_camera_height = 24
    num_stacked_depth_frames = 2

    # ────────────────────────────────────────────────────────
    # ← FIXED #3: depth 维度用 tuple 而非 list，
    #   确保 gym.spaces.Box(shape=tuple(...)) 在所有版本兼容
    # ────────────────────────────────────────────────────────
    observation_space = {
        "state": 16,
        "depth": (num_stacked_depth_frames, depth_camera_height, depth_camera_width),  # ← FIXED
    }
    action_space = 3
    state_space = 0
    debug_vis = False

    # 仿真配置
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # 地形配置
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # ────────────────────────────────────────────────────────
    # ← FIXED #1: env_spacing 必须 > 0
    #
    #   env_spacing=0 会导致所有 256 个环境在世界坐标原点
    #   完全重叠：
    #     • 深度相机同时看到 256 组障碍物和无人机
    #     • USD prim 坐标冲突
    #     • clone_environments 行为异常
    #
    #   设为 50.0（> 2 × map_half_extent）后：
    #     • 每个环境有独立的 50m×50m 区域
    #     • 共享相同的障碍物布局（通过 _shared_obstacle_positions_local）
    #     • 深度相机只看到自己环境内的物体
    #     • 相邻环境最近实体距离 50-20-20=10m > far_clip(8m)
    # ────────────────────────────────────────────────────────
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=256,
        env_spacing=50.0,       # ← FIXED (was 0.0)
        replicate_physics=True,
    )

    # 机器人配置
    robot: ArticulationCfg = OMNINXT_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    cmd_body_vel_xy_max = 0.6
    cmd_vel_z_max = 0.6
    ang_vel_obs_scale = 0.25
    target_pos_obs_scale = 0.1
    target_pos_obs_clip = 2.0
    debug_vis_env_limit = 1

    # 深度相机配置
    depth_camera_near_clip = 0.2
    depth_camera_far_clip = 8.0
    depth_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/body/front_depth_camera",
        update_period=0.0,
        height=24,
        width=32,
        data_types=["depth"],
        depth_clipping_behavior="max",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.10, 0.0, 0.03),
            rot=(0.5, -0.5, 0.5, -0.5),
            convention="ros",
        ),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=12.0,
            focus_distance=4.0,
            horizontal_aperture=20.955,
            clipping_range=(0.2, 8.0),
        ),
    )

    # 障碍物详细配置
    obstacle_height = 1.5
    obstacle_radius = 0.15
    map_half_extent = 20.0
    obstacle_spawn_range = 15.0
    obstacle_border_margin = 5.0
    obstacle_safe_zone = 1.0
    obstacle_min_separation = 0.6
    obstacle_detection_range = 4.0
    spawn_edge_distance = 20.0

    # 目标点详细配置
    target_spawn_range = 20.0
    target_min_height = 0.5
    target_max_height = 2.5
    target_obstacle_clearance = 2.0

    # 出生点详细配置
    spawn_min_height = 0.5
    spawn_max_height = 2.5

    # 奖励系数配置
    vel_tracking_reward_scale = 0.5
    vel_tracking_exp_scale = 4.0
    ang_vel_reward_scale = -0.01
    distance_to_target_reward_scale = 8.0
    target_velocity_reward_scale = 4.0
    target_reached_bonus = 30.0
    obstacle_proximity_reward_scale = -3.0
    progress_reward_scale = 15.0


class QuadcopterObstaclesEnv(DirectRLEnv):
    """带方向障碍物观测的单目标点无人机环境类"""

    cfg: QuadcopterObstaclesEnvCfg

    def __init__(self, cfg: QuadcopterObstaclesEnvCfg, render_mode: str | None = None, **kwargs):
        print("[DEBUG][DirectEnv] __init__ start", flush=True)
        super().__init__(cfg, render_mode, **kwargs)
        print("[DEBUG][DirectEnv] super().__init__ done", flush=True)

        # ========== 动作和力初始化 ==========
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._cmd_vel_b = torch.zeros(self.num_envs, 3, device=self.device)

        # ========== 目标点初始化 ==========
        self._target_positions_local = torch.zeros(self.num_envs, 3, device=self.device)
        self._target_reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._prev_dist_to_target = torch.zeros(self.num_envs, device=self.device)

        # ========== 障碍物初始化 ==========
        self._obstacle_positions_local = torch.zeros(
            self.num_envs, self.cfg.num_obstacles, 3, device=self.device
        )
        self._shared_obstacle_positions_local = torch.zeros(
            self.cfg.num_obstacles, 3, device=self.device
        )

        # ========== 环境重置 ==========
        all_env_ids = torch.arange(self.num_envs, device=self.device)
        self._randomize_obstacles(all_env_ids)
        self._randomize_targets(all_env_ids)

        # ========== 日志初始化 ==========
        self._episode_sums = {
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

        # ========== 机器人参数初始化 ==========
        self._body_id = self._robot.find_bodies("body")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum().item()
        self._controller = CrazyflieController(
            num_envs=self.num_envs,
            device=self.device,
            attitude_dt=self.step_dt,
            position_dt=self.step_dt,
        )
        self._depth_camera = self.scene.sensors["depth_camera"]
        self._cmd_obs_scale = torch.tensor(
            [1.0 / self.cfg.cmd_body_vel_xy_max, 1.0 / self.cfg.cmd_body_vel_xy_max, 1.0 / self.cfg.cmd_vel_z_max],
            device=self.device,
        )
        self._depth_frame_stack = torch.zeros(
            self.num_envs,
            self.cfg.num_stacked_depth_frames,
            self.cfg.depth_camera_height,
            self.cfg.depth_camera_width,
            device=self.device,
        )
        self._depth_stack_needs_fill = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_depth_stack_update_step = -1

        # 碰撞阈值
        self._collision_threshold = 0.05

        # 调试可视化
        self.set_debug_vis(self.cfg.debug_vis)

        # 打印环境信息
        print(f"[INFO] Quadcopter Obstacles v6 - Single Target Navigation")
        print(f"[INFO] Obstacles: {self.cfg.num_obstacles} (observing closest {self.cfg.num_closest_obstacles})")
        print(f"[INFO] env_spacing: {self.cfg.scene.env_spacing}m  |  map_half_extent: {self.cfg.map_half_extent}m")
        print(f"[INFO] Observation space: {self.cfg.observation_space}")
        print(
            f"[INFO] Depth stack: {self.cfg.num_stacked_depth_frames}x"
            f"{self.cfg.depth_camera_height}x{self.cfg.depth_camera_width}"
        )
        print(f"[INFO] OmniNxt mass: {self._robot_mass:.4f} kg  |  Controller CF_MASS: {controller_config.CF_MASS:.4f} kg")
        if abs(self._robot_mass - controller_config.CF_MASS) > 0.05:
            print("[WARN] Mass mismatch — retune controller/config.py if unstable.")

    # ──────────────────────── Scene ────────────────────────

    def _setup_scene(self):
        """设置仿真场景"""
        print("[DEBUG][DirectEnv] _setup_scene start", flush=True)
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        print("[DEBUG][DirectEnv] robot created", flush=True)

        self._depth_camera_sensor = TiledCamera(self.cfg.depth_camera)
        self.scene.sensors["depth_camera"] = self._depth_camera_sensor
        print("[DEBUG][DirectEnv] tiled camera created", flush=True)

        # 障碍物（视觉+刚体，碰撞关闭，用距离逻辑检测碰撞）
        self._obstacles: list[RigidObject] = []
        obstacle_spawn_cfg = sim_utils.CuboidCfg(
            size=(
                self.cfg.obstacle_radius * 2,
                self.cfg.obstacle_radius * 2,
                self.cfg.obstacle_height,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.8, 0.2, 0.2),
                metallic=0.0,
            ),
        )
        for obstacle_idx in range(self.cfg.num_obstacles):
            obstacle_cfg = RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/Obstacle_{obstacle_idx:03d}",
                spawn=obstacle_spawn_cfg,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -10.0)),
            )
            obstacle_obj = RigidObject(cfg=obstacle_cfg)
            self._obstacles.append(obstacle_obj)
            self.scene.rigid_objects[f"obstacle_{obstacle_idx:03d}"] = obstacle_obj
        print("[DEBUG][DirectEnv] obstacle objects created", flush=True)

        # 地形
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        print("[DEBUG][DirectEnv] terrain created", flush=True)

        # 克隆环境
        self.scene.clone_environments(copy_from_source=False)
        print("[DEBUG][DirectEnv] clone_environments done", flush=True)

        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # 灯光
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        print("[DEBUG][DirectEnv] _setup_scene complete", flush=True)

    # ──────────────────────── Obstacles ────────────────────────

    def _sync_render_obstacles(self, env_ids: torch.Tensor):
        """将逻辑障碍物坐标同步到 RigidObject 世界坐标。"""
        env_origins = self._terrain.env_origins[env_ids]          # ← FIXED: 取完整 xyz
        obstacle_pos_local = self._obstacle_positions_local[env_ids]

        obstacle_pos_world = obstacle_pos_local.clone()
        obstacle_pos_world[:, :, :2] += env_origins[:, :2].unsqueeze(1)  # 只偏移 xy

        for obstacle_idx, obstacle_obj in enumerate(self._obstacles):
            root_state = torch.zeros((len(env_ids), 13), device=self.device)
            root_state[:, :3] = obstacle_pos_world[:, obstacle_idx]
            root_state[:, 3] = 1.0  # 单位四元数 w
            obstacle_obj.write_root_state_to_sim(root_state, env_ids)

    def _randomize_obstacles(self, env_ids: torch.Tensor):
        """随机生成障碍物位置（共享布局）"""
        regenerate_shared_layout = not torch.any(self._shared_obstacle_positions_local)
        if len(env_ids) == self.num_envs:
            regenerate_shared_layout = True

        if regenerate_shared_layout:
            placed_xy = torch.zeros(self.cfg.num_obstacles, 2, device=self.device)
        else:
            placed_xy = self._shared_obstacle_positions_local[:, :2].clone()
        min_separation = self.cfg.obstacle_min_separation

        if regenerate_shared_layout:
            for obstacle_idx in range(self.cfg.num_obstacles):
                chosen_xy = None
                chosen_score = None

                for _ in range(64):
                    candidate_xy = torch.empty(2, device=self.device).uniform_(
                        -self.cfg.obstacle_spawn_range, self.cfg.obstacle_spawn_range
                    )
                    candidate_radius = torch.linalg.norm(candidate_xy, dim=0)
                    valid = candidate_radius >= (self.cfg.obstacle_safe_zone + self.cfg.obstacle_radius + 0.2)

                    if obstacle_idx == 0:
                        min_dist = torch.tensor(float("inf"), device=self.device)
                    else:
                        distances = torch.linalg.norm(candidate_xy.unsqueeze(0) - placed_xy[:obstacle_idx], dim=1)
                        min_dist = distances.min()
                        valid = bool(valid and (min_dist >= min_separation))

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

            self._shared_obstacle_positions_local[:, :2] = placed_xy
            self._shared_obstacle_positions_local[:, 2] = self.cfg.obstacle_height / 2

        self._obstacle_positions_local[env_ids] = self._shared_obstacle_positions_local.unsqueeze(0).expand(
            len(env_ids), -1, -1
        )
        if hasattr(self, "_obstacles"):
            self._sync_render_obstacles(env_ids)

    # ──────────────────────── Targets ────────────────────────

    def _randomize_targets(self, env_ids: torch.Tensor):
        """随机生成目标点位置"""
        num_envs_to_reset = len(env_ids)
        clearance = self.cfg.target_obstacle_clearance + self.cfg.obstacle_radius

        best_pos = None
        best_clearance = None

        for _ in range(64):
            candidate_pos = self._sample_edge_positions(
                num_envs_to_reset,
                self.cfg.target_spawn_range,
                self.cfg.target_min_height,
                self.cfg.target_max_height,
            )
            candidate_xy = candidate_pos[:, :2]
            distances = torch.linalg.norm(
                candidate_xy.unsqueeze(1) - self._obstacle_positions_local[env_ids, :, :2], dim=2
            )
            min_clearance = distances.min(dim=1).values
            valid_mask = min_clearance > clearance

            if best_pos is None:
                best_pos = candidate_pos
                best_clearance = min_clearance
            else:
                better_mask = min_clearance > best_clearance
                best_pos[better_mask] = candidate_pos[better_mask]
                best_clearance[better_mask] = min_clearance[better_mask]

            if valid_mask.all():
                best_pos = candidate_pos
                break

            unresolved_mask = min_clearance <= clearance
            if not unresolved_mask.any():
                best_pos = candidate_pos
                break

            if best_pos is not None:
                best_pos[valid_mask] = candidate_pos[valid_mask]

        self._target_positions_local[env_ids] = best_pos
        self._target_reached[env_ids] = False
        self._prev_dist_to_target[env_ids] = 0.0

    def _sample_edge_positions(
        self,
        num_samples: int,
        lateral_range: float,
        min_height: float,
        max_height: float,
    ) -> torch.Tensor:
        """在四条边上采样位置"""
        side_indices = torch.randint(0, 4, (num_samples,), device=self.device)
        side_signs = torch.where(
            side_indices % 2 == 0,
            torch.ones(num_samples, device=self.device),
            -torch.ones(num_samples, device=self.device),
        )
        lateral = torch.empty(num_samples, device=self.device).uniform_(-lateral_range, lateral_range)
        heights = torch.empty(num_samples, device=self.device).uniform_(min_height, max_height)

        positions = torch.zeros(num_samples, 3, device=self.device)
        x_side_mask = side_indices < 2
        y_side_mask = ~x_side_mask

        positions[x_side_mask, 0] = lateral[x_side_mask]
        positions[x_side_mask, 1] = side_signs[x_side_mask] * self.cfg.spawn_edge_distance
        positions[y_side_mask, 0] = side_signs[y_side_mask] * self.cfg.spawn_edge_distance
        positions[y_side_mask, 1] = lateral[y_side_mask]
        positions[:, 2] = heights
        return positions

    # ──────────────────────── Helpers ────────────────────────

    def _get_target_world(self) -> torch.Tensor:
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
        """计算机体坐标系下最近 N 个障碍物的方向和距离"""
        drone_pos_w = self._robot.data.root_pos_w
        drone_quat_w = self._robot.data.root_quat_w
        env_origins = self._terrain.env_origins

        drone_pos_local = drone_pos_w - env_origins
        drone_expanded = drone_pos_local.unsqueeze(1)
        to_obstacles = self._obstacle_positions_local - drone_expanded

        distances = torch.linalg.norm(to_obstacles, dim=2) - self.cfg.obstacle_radius
        _, closest_indices = torch.topk(distances, self.cfg.num_closest_obstacles, dim=1, largest=False)

        batch_indices = torch.arange(self.num_envs, device=self.device).unsqueeze(1).expand(
            -1, self.cfg.num_closest_obstacles
        )
        closest_vectors = to_obstacles[batch_indices, closest_indices]
        closest_distances = distances[batch_indices, closest_indices]

        closest_vectors_norm = closest_vectors / (torch.linalg.norm(closest_vectors, dim=2, keepdim=True) + 1e-6)

        num_closest = self.cfg.num_closest_obstacles
        closest_vectors_flat = closest_vectors_norm.reshape(self.num_envs * num_closest, 3)
        drone_quat_expanded = drone_quat_w.unsqueeze(1).expand(-1, num_closest, -1).reshape(
            self.num_envs * num_closest, 4
        )
        directions_body = quat_apply_inverse(drone_quat_expanded, closest_vectors_flat)

        distances_normalized = (closest_distances / self.cfg.obstacle_detection_range).clamp(0.0, 1.0)

        return directions_body, distances_normalized

    def _compute_min_obstacle_distance(self) -> torch.Tensor:
        """计算到最近圆柱障碍物表面的有符号距离。"""
        drone_pos_local = self._robot.data.root_pos_w - self._terrain.env_origins
        obstacle_pos_local = self._obstacle_positions_local

        radial_dist = torch.linalg.norm(
            drone_pos_local[:, None, :2] - obstacle_pos_local[:, :, :2], dim=2
        ) - self.cfg.obstacle_radius
        vertical_dist = torch.abs(
            drone_pos_local[:, None, 2] - obstacle_pos_local[:, :, 2]
        ) - (self.cfg.obstacle_height / 2.0)

        outside_radial = radial_dist.clamp_min(0.0)
        outside_vertical = vertical_dist.clamp_min(0.0)
        outside_dist = torch.sqrt(outside_radial.square() + outside_vertical.square())
        inside_dist = torch.minimum(torch.maximum(radial_dist, vertical_dist), torch.zeros_like(radial_dist))
        signed_dist = outside_dist + inside_dist

        return signed_dist.min(dim=1).values

    # ──────────────────────── Depth ────────────────────────

    def _get_depth_image(self) -> torch.Tensor:
        """读取深度图并转为接近度 [0,1]。"""
        if not hasattr(self, "_debug_depth_log_once"):
            self._debug_depth_log_once = False
        if not self._debug_depth_log_once:
            print("[DEBUG][DirectEnv] reading depth image", flush=True)
        depth = self._depth_camera.data.output["depth"].squeeze(-1)
        depth = torch.nan_to_num(
            depth,
            nan=self.cfg.depth_camera_far_clip,
            posinf=self.cfg.depth_camera_far_clip,
            neginf=self.cfg.depth_camera_near_clip,
        )
        depth = depth.clamp(self.cfg.depth_camera_near_clip, self.cfg.depth_camera_far_clip)

        depth_proximity = 1.0 - (
            (depth - self.cfg.depth_camera_near_clip)
            / (self.cfg.depth_camera_far_clip - self.cfg.depth_camera_near_clip)
        )
        if not self._debug_depth_log_once:
            print(f"[DEBUG][DirectEnv] depth image ready: {tuple(depth_proximity.shape)}", flush=True)
            self._debug_depth_log_once = True
        return depth_proximity.clamp(0.0, 1.0).unsqueeze(1)

    def _get_stacked_depth_images(self) -> torch.Tensor:
        """返回最近 N 帧深度图堆叠 [num_envs, N, H, W]。"""
        current_depth = self._get_depth_image()

        if self.common_step_counter != self._last_depth_stack_update_step:
            initialized_mask = ~self._depth_stack_needs_fill
            if initialized_mask.any():
                self._depth_frame_stack[initialized_mask] = torch.roll(
                    self._depth_frame_stack[initialized_mask], shifts=-1, dims=1
                )
                self._depth_frame_stack[initialized_mask, -1] = current_depth[initialized_mask, 0]
            self._last_depth_stack_update_step = self.common_step_counter

        if self._depth_stack_needs_fill.any():
            fill_mask = self._depth_stack_needs_fill
            self._depth_frame_stack[fill_mask] = current_depth[fill_mask].expand(
                -1, self.cfg.num_stacked_depth_frames, -1, -1
            )
            self._depth_stack_needs_fill[fill_mask] = False

        return self._depth_frame_stack

    # ──────────────────────── Core RL ────────────────────────

    def _pre_physics_step(self, actions: torch.Tensor):
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
        self._robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)

    # ────────────────────────────────────────────────────────
    # ← FIXED #2: 返回值必须包含 "policy" 键
    #
    #   IsaacLab 的 DirectRLEnv 将 observation_space dict
    #   自动包装在 {"policy": Dict(...)} 下。
    #   RSL-RL wrapper 通过 obs_buf["policy"] 读取观测，
    #   缺少此键会触发 KeyError。
    # ────────────────────────────────────────────────────────
    def _get_observations(self) -> dict:
        # 当前目标点在机体坐标系下的位置
        target_world = self._get_target_world()
        target_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_pos_w, self._robot.data.root_quat_w, target_world
        )
        ang_vel_obs = (self._robot.data.root_ang_vel_b * self.cfg.ang_vel_obs_scale).clamp(-2.0, 2.0)
        target_pos_obs = (target_pos_b * self.cfg.target_pos_obs_scale).clamp(
            -self.cfg.target_pos_obs_clip, self.cfg.target_pos_obs_clip
        )
        cmd_obs = (self._cmd_vel_b * self._cmd_obs_scale).clamp(-1.0, 1.0)

        state_obs = torch.cat(
            [
                self._robot.data.root_lin_vel_b,       # 3
                ang_vel_obs,                            # 3
                self._robot.data.projected_gravity_b,   # 3
                target_pos_obs,                         # 3
                cmd_obs,                                # 3
                self._target_reached.float().unsqueeze(1),  # 1
            ],
            dim=-1,
        )  # total = 16

        return {"policy": {"state": state_obs, "depth": self._get_stacked_depth_images()}}  # ← FIXED

    def _get_rewards(self) -> torch.Tensor:
        # 速度命令跟踪
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
        ang_vel = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)

        # 距离奖励
        target_world = self._get_target_world()
        distance_to_target = torch.linalg.norm(target_world - self._robot.data.root_pos_w, dim=1)
        distance_reward = 1 - torch.tanh(distance_to_target / 2.0)

        target_direction = target_world - self._robot.data.root_pos_w
        target_direction = target_direction / torch.linalg.norm(target_direction, dim=1, keepdim=True).clamp_min(1e-6)
        target_velocity = torch.sum(self._robot.data.root_lin_vel_w * target_direction, dim=1).clamp(-1.0, 1.5)

        # 进度奖励
        progress_reward = (self._prev_dist_to_target - distance_to_target).clamp(-1.0, 1.0)
        self._prev_dist_to_target = distance_to_target.clone()

        # 目标到达
        target_reached = (distance_to_target < self.cfg.target_reach_threshold) & (~self._target_reached)
        target_bonus = torch.zeros(self.num_envs, device=self.device)
        if target_reached.any():
            target_bonus[target_reached] = self.cfg.target_reached_bonus
            self._target_reached[target_reached] = True

        # 障碍物惩罚
        min_obstacle_dist = self._compute_min_obstacle_distance()
        obstacle_proximity = torch.where(
            min_obstacle_dist < 1.0,
            torch.exp(-min_obstacle_dist * 3.0),
            torch.zeros_like(min_obstacle_dist),
        )

        # 总奖励
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

        for key, value in rewards.items():
            self._episode_sums[key] += value
        self._episode_sums["vel_tracking_error"] += vel_tracking_error * self.step_dt

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        too_low = self._robot.data.root_pos_w[:, 2] < 0.1
        too_high = self._robot.data.root_pos_w[:, 2] > 2.5
        min_obstacle_dist = self._compute_min_obstacle_distance()
        collision = min_obstacle_dist < self._collision_threshold
        success = self._target_reached
        died = too_low | too_high | collision
        terminated = died | success
        return terminated, time_out

    # ──────────────────────── Reset ────────────────────────

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # 日志
        success_rate = self._target_reached[env_ids].float().mean().item()

        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            if key == "vel_tracking_error":
                extras["Metrics/avg_vel_tracking_error"] = episodic_sum_avg / self.max_episode_length_s
            else:
                extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)

        extras = dict()
        extras["Episode_Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        extras["Metrics/success_rate"] = success_rate
        self.extras["log"].update(extras)

        # 重置机器人
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        if len(env_ids) == self.num_envs:
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0
        self._cmd_vel_b[env_ids] = 0.0

        # 只重新随机目标点（地图/障碍物不变）
        self._randomize_targets(env_ids)

        # 生成初始姿态
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        spawn_pos_local = self._sample_edge_positions(
            len(env_ids),
            self.cfg.target_spawn_range,
            self.cfg.spawn_min_height,
            self.cfg.spawn_max_height,
        )
        spawn_pos_world = spawn_pos_local.clone()
        spawn_pos_world[:, :2] += self._terrain.env_origins[env_ids, :2]

        target_world = self._target_positions_local[env_ids].clone()
        target_world[:, :2] += self._terrain.env_origins[env_ids, :2]
        target_delta = target_world - spawn_pos_world
        facing_yaw = torch.atan2(target_delta[:, 1], target_delta[:, 0])
        half_yaw = facing_yaw * 0.5

        default_root_state = self._robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] = spawn_pos_world
        default_root_state[:, 3] = torch.cos(half_yaw)
        default_root_state[:, 4] = 0.0
        default_root_state[:, 5] = 0.0
        default_root_state[:, 6] = torch.sin(half_yaw)
        default_root_state[:, 7:13] = 0.0

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        controller_position = torch.zeros((self.num_envs, 3), device=self.device, dtype=default_root_state.dtype)
        controller_attitude = torch.zeros((self.num_envs, 3), device=self.device, dtype=default_root_state.dtype)
        controller_position[env_ids] = default_root_state[:, :3]
        controller_attitude[env_ids] = _quat_to_euler_deg(default_root_state[:, 3:7])
        controller_state = {
            "position": controller_position,
            "attitude": controller_attitude,
        }
        self._controller.reset(controller_state, env_ids=env_ids)
        self._controller.set_velocity_setpoint(
            vx=torch.zeros(len(env_ids), device=self.device),
            vy=torch.zeros(len(env_ids), device=self.device),
            vz=torch.zeros(len(env_ids), device=self.device),
            velocity_body=True,
            env_ids=env_ids,
        )
        self._depth_camera.reset(env_ids)
        self._depth_stack_needs_fill[env_ids] = True
        self._update_prev_target_distance(env_ids)

    # ──────────────────────── Debug Vis ────────────────────────

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "target_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.2, 0.2, 0.2)
                marker_cfg.markers["cuboid"].visual_material = sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 1.0, 0.0)
                )
                marker_cfg.prim_path = "/Visuals/Target"
                self.target_visualizer = VisualizationMarkers(marker_cfg)
            self.target_visualizer.set_visibility(True)

            if not hasattr(self, "obstacle_visualizer"):
                obs_marker_cfg = CUBOID_MARKER_CFG.copy()
                pillar_size = self.cfg.obstacle_radius * 2
                obs_marker_cfg.markers["cuboid"].size = (pillar_size, pillar_size, self.cfg.obstacle_height)
                obs_marker_cfg.markers["cuboid"].visual_material = sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.8, 0.2, 0.2)
                )
                obs_marker_cfg.prim_path = "/Visuals/Obstacles"
                self.obstacle_visualizer = VisualizationMarkers(obs_marker_cfg)
            self.obstacle_visualizer.set_visibility(True)
        else:
            if hasattr(self, "target_visualizer"):
                self.target_visualizer.set_visibility(False)
            if hasattr(self, "obstacle_visualizer"):
                self.obstacle_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        env_limit = min(self.cfg.debug_vis_env_limit, self.num_envs)
        env_slice = slice(0, env_limit)

        if hasattr(self, "target_visualizer"):
            target_world = self._get_target_world()[env_slice]
            self.target_visualizer.visualize(target_world)

        if hasattr(self, "obstacle_visualizer"):
            env_origins = self._terrain.env_origins[env_slice]
            env_origins_expanded = env_origins.unsqueeze(1).repeat(1, self.cfg.num_obstacles, 1)
            obstacle_pos_w = self._obstacle_positions_local[env_slice].clone()
            obstacle_pos_w[:, :, :2] += env_origins_expanded[:, :, :2]
            obstacle_pos_flat = obstacle_pos_w.reshape(-1, 3)
            self.obstacle_visualizer.visualize(obstacle_pos_flat)
