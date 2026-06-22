from __future__ import annotations

from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
import random
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import schemas as sim_schemas
from isaaclab.utils.math import quat_apply_inverse
from pxr import Gf, UsdGeom

from assets.omninxt.omninxt import OMNINXT_CFG
from controller import CrazyflieController, config as controller_config


DEFAULT_REPRO_SEED = 3407


def _quat_to_euler_deg(quat_w: torch.Tensor) -> torch.Tensor:
    """Convert quaternions in (w, x, y, z) format to Euler XYZ in degrees."""
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

    map_half_extent: float = 20.0
    wall_half_extent: float = 30.0
    wall_height: float = 5.0
    wall_thickness: float = 0.3

    num_obstacles: int = 240
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

    num_dynamic_obstacles: int = 80
    num_closest_dynamic_obstacles: int = 5
    dynamic_obstacle_max_width: float = 1.0
    dynamic_obstacle_3d_height: float = 1.0
    dynamic_obstacle_2d_height: float = 5.0
    dynamic_obstacle_vel_range: tuple[float, float] = (0.5, 1.5)
    dynamic_obstacle_local_range: tuple[float, float, float] = (5.0, 5.0, 4.5)
    dynamic_obstacle_goal_threshold: float = 0.5
    dynamic_obstacle_velocity_resample_s: float = 2.0
    dynamic_obstacle_collision_margin: float = 0.3
    dynamic_obstacle_visual_update_interval: int = 0

    spawn_edge_distance: float = 24.0
    target_spawn_range: float = 24.0
    spawn_min_height: float = 0.5
    spawn_max_height: float = 2.5
    target_min_height: float = 0.5
    target_max_height: float = 2.5
    flight_min_height: float = 0.1
    flight_max_height: float = 4.0
    target_reach_threshold: float = 0.5
    drone_spawn_min_separation: float = 1.0

    cmd_body_vel_xy_max: float = 2.0
    cmd_vel_z_max: float = 1.0
    cmd_vel_smoothing_alpha: float = 0.35
    cmd_body_vel_xy_rate_limit: float = 2.0
    cmd_vel_z_rate_limit: float = 4.0
    cmd_yaw_angle_max: float = 180.0
    cmd_yaw_smoothing_alpha: float = 0.35
    cmd_yaw_rate_limit: float = 90.0

    vel_tracking_reward_scale: float = 0.5
    vel_tracking_exp_scale: float = 4.0
    ang_vel_reward_scale: float = -0.01
    distance_to_target_reward_scale: float = 12.0
    target_velocity_reward_scale: float = 1.5
    target_reached_bonus: float = 100.0
    obstacle_proximity_reward_scale: float = -16.0
    progress_reward_scale: float = 8.0
    static_collision_penalty: float = -40.0
    dynamic_collision_penalty: float = -40.0
    wall_collision_penalty: float = -40.0
    height_violation_penalty: float = -40.0
    action_rate_penalty_scale: float = -1.0
    cmd_rate_penalty_scale: float = -1.5
    velocity_accel_penalty_scale: float = -0.8
    overspeed_penalty_scale: float = -0.20
    yaw_alignment_reward_scale: float = 1.0
    yaw_error_penalty_scale: float = -0.2

    observation_space: int = 0
    action_space: int = 4
    state_space: int = 0
    debug_vis: bool = False
    viewer_eye: tuple[float, float, float] = (-30.0, 0.0, 80.0)
    viewer_lookat: tuple[float, float, float] = (0.0, 0.0, 0.0)
    viewer_cam_prim_path: str = "/OmniverseKit_Persp"
    sim: SimCfg = field(default_factory=SimCfg)
    scene: SceneCfg = field(default_factory=SceneCfg)

    def policy_observation_dim(self) -> int:
        base_dim = 3 + 3 + 3 + 3 + 3
        closest_obstacle_dim = self.num_closest_obstacles * 3 + self.num_closest_obstacles
        dynamic_obstacle_dim = self.num_closest_dynamic_obstacles * 10
        extra_dim = 1 + 1 + 1
        return base_dim + closest_obstacle_dim + dynamic_obstacle_dim + extra_dim

    def __post_init__(self):
        self.scene.num_envs = int(self.scene.num_envs)
        self.sim.device = self.device
        self.observation_space = self.policy_observation_dim()


class QuadcopterObstaclesEnv(gym.Env):
    """Single-world multi-drone student environment with static and dynamic obstacles."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, cfg: QuadcopterObstaclesEnvCfg | None = None, render_mode: str | None = None, **kwargs):
        del kwargs
        self.cfg = cfg if cfg is not None else QuadcopterObstaclesEnvCfg()
        self.render_mode = render_mode
        if hasattr(self.cfg, "scene"):
            self.cfg.scene.num_envs = int(self.cfg.scene.num_envs)
        if hasattr(self.cfg, "sim"):
            self.cfg.device = self.cfg.sim.device

        self.device = torch.device(self.cfg.device)
        if self.cfg.seed is None:
            self.cfg.seed = DEFAULT_REPRO_SEED
        self._torch_rng = torch.Generator(device=self.device)
        self.seed(int(self.cfg.seed))
        self.num_envs = self.cfg.scene.num_envs
        self.step_dt = self.cfg.physics_dt * self.cfg.decimation
        cmd_xy_delta = (
            float(self.cfg.cmd_body_vel_xy_rate_limit) * self.step_dt
            if float(self.cfg.cmd_body_vel_xy_rate_limit) > 0.0
            else float("inf")
        )
        cmd_z_delta = (
            float(self.cfg.cmd_vel_z_rate_limit) * self.step_dt
            if float(self.cfg.cmd_vel_z_rate_limit) > 0.0
            else float("inf")
        )
        self._cmd_vel_max_delta = torch.tensor([cmd_xy_delta, cmd_xy_delta, cmd_z_delta], device=self.device)
        self._cmd_yaw_max_delta = (
            float(self.cfg.cmd_yaw_rate_limit) * self.step_dt
            if float(self.cfg.cmd_yaw_rate_limit) > 0.0
            else float("inf")
        )
        self.max_episode_length = int(round(self.cfg.episode_length_s / self.step_dt))
        self.max_episode_length_s = self.max_episode_length * self.step_dt
        self.num_states = 0
        self.common_step_counter = 0
        self.training = True

        self.single_action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.cfg.action_space,), dtype=float)
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)
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
        self._dynamic_obstacle_paths: list[str] = []
        self._dynamic_obstacle_translate_ops = []
        self._wall_paths: list[str] = []
        self._robot_mass = 0.0

        self._actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self._prev_actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self._target_cmd_vel_b = torch.zeros((self.num_envs, 3), device=self.device)
        self._cmd_vel_b = torch.zeros((self.num_envs, 3), device=self.device)
        self._prev_cmd_vel_b = torch.zeros((self.num_envs, 3), device=self.device)
        self._prev_velocity_cmd_frame = torch.zeros((self.num_envs, 3), device=self.device)
        self._target_cmd_yaw_deg = torch.zeros(self.num_envs, device=self.device)
        self._cmd_yaw_deg = torch.zeros(self.num_envs, device=self.device)
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
        self._prev_dist_to_target = torch.zeros(self.num_envs, device=self.device)
        self._obstacle_positions_w = torch.zeros((self.cfg.num_obstacles, 3), device=self.device)
        self._obstacle_heights = torch.full((self.cfg.num_obstacles,), self.cfg.obstacle_height, device=self.device)
        self._obstacle_radii = torch.full((self.cfg.num_obstacles,), self.cfg.obstacle_radius, device=self.device)
        self._obstacle_positions_xy = self._obstacle_positions_w[:, :2]
        self._dynamic_category_count = 8
        self._dynamic_num_per_category = self.cfg.num_dynamic_obstacles // self._dynamic_category_count
        self._num_dynamic_obstacles = self._dynamic_num_per_category * self._dynamic_category_count
        self._dynamic_obstacle_positions_w = torch.zeros((self._num_dynamic_obstacles, 3), device=self.device)
        self._dynamic_obstacle_origins_w = torch.zeros((self._num_dynamic_obstacles, 3), device=self.device)
        self._dynamic_obstacle_goals_w = torch.zeros((self._num_dynamic_obstacles, 3), device=self.device)
        self._dynamic_obstacle_velocities_w = torch.zeros((self._num_dynamic_obstacles, 3), device=self.device)
        self._dynamic_obstacle_sizes = torch.zeros((self._num_dynamic_obstacles, 3), device=self.device)
        self._dynamic_obstacle_step_count = 0
        self._dynamic_width_res = self.cfg.dynamic_obstacle_max_width / 4.0
        self._dynamic_2d_start_idx = self._num_dynamic_obstacles // 2
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
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "vel_tracking",
                "ang_vel",
                "distance_to_target",
                "target_velocity",
                "target_bonus",
                "obstacle_proximity",
                "progress",
                "static_collision_penalty",
                "dynamic_collision_penalty",
                "wall_collision_penalty",
                "height_violation_penalty",
                "action_rate",
                "cmd_rate",
                "velocity_accel",
                "overspeed",
                "yaw_alignment",
                "yaw_error",
                "vel_tracking_error",
            ]
        }

    @property
    def unwrapped(self) -> QuadcopterObstaclesEnv:
        return self

    def train(self):
        self.training = True
        return self

    def eval(self):
        self.training = False
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
        if seed >= 0:
            seed = int(seed)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            if hasattr(self, "_torch_rng"):
                self._torch_rng.manual_seed(seed)
            self.cfg.seed = seed
        return seed

    def _build(self) -> None:
        if self._built:
            return

        sim_cfg = sim_utils.SimulationCfg(dt=self.cfg.physics_dt, device=str(self.device))
        self._sim = sim_utils.SimulationContext(sim_cfg)
        self.sim.set_camera_view(
            eye=list(self.cfg.viewer_eye),
            target=list(self.cfg.viewer_lookat),
            camera_prim_path=self.cfg.viewer_cam_prim_path,
        )

        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_cfg.func("/World/Ground", ground_cfg)
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        self._spawn_shared_obstacles()
        self._spawn_dynamic_obstacles()
        self._spawn_boundary_walls()
        self._build_robot_assets()
        self._disable_robot_collisions()

        self.sim.reset()
        self._body_id = self.robot.find_bodies("body")[0]
        self._robot_mass = self.robot.root_physx_view.get_masses()[0].sum().item()
        self.robot.write_joint_state_to_sim(self.robot.data.default_joint_pos, self.robot.data.default_joint_vel)
        self.robot.update(self.cfg.physics_dt)
        for _ in range(5):
            self.sim.step()
            self.robot.update(self.cfg.physics_dt)

        print(
            f"[INFO] Quadcopter Obstacles Student - Single World with {self.num_envs} robots, "
            f"{self.cfg.num_obstacles} static obstacles, and {self._num_dynamic_obstacles} dynamic obstacles"
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
        side_indices = torch.randint(0, 4, (num_samples,), device=self.device, generator=self._torch_rng)
        side_signs = torch.where(
            side_indices % 2 == 0,
            torch.ones(num_samples, device=self.device),
            -torch.ones(num_samples, device=self.device),
        )
        lateral = torch.empty(num_samples, device=self.device).uniform_(
            -lateral_range, lateral_range, generator=self._torch_rng
        )
        heights = torch.empty(num_samples, device=self.device).uniform_(
            min_height, max_height, generator=self._torch_rng
        )

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
        positions = torch.zeros((num_samples, 3), device=self.device)
        obstacle_clearance = self.cfg.target_obstacle_clearance if obstacle_clearance is None else obstacle_clearance
        obstacle_clearance_sq = obstacle_clearance**2
        for idx in range(num_samples):
            best_candidate = None
            best_score = None
            for _ in range(128):
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
        num_samples = len(source_side_indices)
        opposite_side_indices = torch.where(source_side_indices % 2 == 0, source_side_indices + 1, source_side_indices - 1)
        lateral = torch.empty(num_samples, device=self.device).uniform_(
            -self.cfg.spawn_edge_distance, self.cfg.spawn_edge_distance, generator=self._torch_rng
        )
        heights = torch.empty(num_samples, device=self.device).uniform_(
            min_height, max_height, generator=self._torch_rng
        )

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
        placed_xy = torch.zeros((self.cfg.num_obstacles, 2), device=self.device)
        for obstacle_idx in range(self.cfg.num_obstacles):
            chosen_xy = None
            chosen_score = None
            for _ in range(64):
                candidate_xy = torch.empty(2, device=self.device).uniform_(
                    -self.cfg.obstacle_spawn_range,
                    self.cfg.obstacle_spawn_range,
                    generator=self._torch_rng,
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

    def _sample_dynamic_obstacle_origins(self) -> torch.Tensor:
        if self._num_dynamic_obstacles == 0:
            return torch.zeros((0, 3), device=self.device)

        positions = torch.zeros((self._num_dynamic_obstacles, 3), device=self.device)
        preferred_dist = 2.0 * (self.cfg.map_half_extent**2 / max(self._num_dynamic_obstacles, 1)) ** 0.5
        for idx in range(self._num_dynamic_obstacles):
            is_2d = idx >= self._dynamic_2d_start_idx
            best_candidate = None
            best_score = None
            for _ in range(128):
                candidate = torch.empty(3, device=self.device)
                candidate[0].uniform_(-self.cfg.map_half_extent, self.cfg.map_half_extent, generator=self._torch_rng)
                candidate[1].uniform_(-self.cfg.map_half_extent, self.cfg.map_half_extent, generator=self._torch_rng)
                if is_2d:
                    candidate[2] = self.cfg.dynamic_obstacle_2d_height * 0.5
                else:
                    candidate[2].uniform_(0.0, self.cfg.flight_max_height, generator=self._torch_rng)

                if idx == 0:
                    score = torch.tensor(float("inf"), device=self.device)
                else:
                    dist_sq = torch.sum(torch.square(candidate[:2].unsqueeze(0) - positions[:idx, :2]), dim=1)
                    score = dist_sq.min()

                if best_candidate is None or score > best_score:
                    best_candidate = candidate
                    best_score = score
                if idx == 0 or score >= preferred_dist**2:
                    best_candidate = candidate
                    break
            positions[idx] = best_candidate
        return positions

    def _spawn_dynamic_obstacles(self) -> None:
        self._dynamic_obstacle_paths.clear()
        self._dynamic_obstacle_translate_ops.clear()
        if self._num_dynamic_obstacles == 0:
            return

        self._dynamic_obstacle_origins_w[:] = self._sample_dynamic_obstacle_origins()
        self._dynamic_obstacle_positions_w[:] = self._dynamic_obstacle_origins_w
        self._dynamic_obstacle_goals_w[:] = self._dynamic_obstacle_origins_w
        self._dynamic_obstacle_velocities_w.zero_()
        self._dynamic_obstacle_step_count = 0

        for idx in range(self._num_dynamic_obstacles):
            category_idx = idx // self._dynamic_num_per_category
            width_category = category_idx % 4
            width = (width_category + 1) * self._dynamic_width_res
            is_2d = category_idx >= 4
            height = self.cfg.dynamic_obstacle_2d_height if is_2d else self.cfg.dynamic_obstacle_3d_height
            self._dynamic_obstacle_sizes[idx] = torch.tensor([width, width, height], device=self.device)

            prim_path = f"/World/DynamicObstacles/Obstacle_{idx:03d}"
            if is_2d:
                obstacle_cfg = sim_utils.CylinderCfg(
                    radius=width * 0.5,
                    height=height,
                    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.8, 0.2), metallic=0.0),
                )
            else:
                obstacle_cfg = sim_utils.CuboidCfg(
                    size=(width, width, height),
                    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.8, 0.2), metallic=0.0),
                )
            obstacle_cfg.func(prim_path, obstacle_cfg, translation=tuple(self._dynamic_obstacle_positions_w[idx].tolist()))
            self._dynamic_obstacle_paths.append(prim_path)
            self._dynamic_obstacle_translate_ops.append(self._get_or_add_translate_op(prim_path))

        self._sample_dynamic_obstacle_goals(torch.ones(self._num_dynamic_obstacles, dtype=torch.bool, device=self.device))
        self._resample_dynamic_obstacle_velocities()

    def _get_or_add_translate_op(self, prim_path: str):
        prim = self.sim.stage.GetPrimAtPath(prim_path)
        xformable = UsdGeom.Xformable(prim)
        for op in xformable.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                return op
        return xformable.AddTranslateOp()

    def _update_dynamic_obstacle_visuals(self) -> None:
        for idx, translate_op in enumerate(self._dynamic_obstacle_translate_ops):
            pos = self._dynamic_obstacle_positions_w[idx].detach().cpu().tolist()
            translate_op.Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))

    def _sample_dynamic_obstacle_goals(self, new_goal_mask: torch.Tensor) -> None:
        num_new_goals = int(new_goal_mask.sum().item())
        if num_new_goals == 0:
            return

        local_range = self.cfg.dynamic_obstacle_local_range
        sample = torch.empty((num_new_goals, 3), device=self.device)
        sample[:, 0].uniform_(-local_range[0], local_range[0], generator=self._torch_rng)
        sample[:, 1].uniform_(-local_range[1], local_range[1], generator=self._torch_rng)
        sample[:, 2].uniform_(-local_range[2], local_range[2], generator=self._torch_rng)

        goals = self._dynamic_obstacle_origins_w[new_goal_mask] + sample
        goals[:, 0].clamp_(-self.cfg.map_half_extent, self.cfg.map_half_extent)
        goals[:, 1].clamp_(-self.cfg.map_half_extent, self.cfg.map_half_extent)
        goals[:, 2].clamp_(0.0, self.cfg.flight_max_height)
        new_indices = torch.nonzero(new_goal_mask, as_tuple=False).squeeze(-1)
        is_2d = new_indices >= self._dynamic_2d_start_idx
        goals[is_2d, 2] = self.cfg.dynamic_obstacle_2d_height * 0.5
        self._dynamic_obstacle_goals_w[new_goal_mask] = goals

    def _resample_dynamic_obstacle_velocities(self) -> None:
        if self._num_dynamic_obstacles == 0:
            return

        goal_delta = self._dynamic_obstacle_goals_w - self._dynamic_obstacle_positions_w
        goal_dir = goal_delta / torch.linalg.norm(goal_delta, dim=1, keepdim=True).clamp_min(1e-6)
        speed_min, speed_max = self.cfg.dynamic_obstacle_vel_range
        speeds = torch.empty((self._num_dynamic_obstacles, 1), device=self.device).uniform_(
            speed_min, speed_max, generator=self._torch_rng
        )
        self._dynamic_obstacle_velocities_w[:] = speeds * goal_dir

    def _move_dynamic_obstacles(self) -> None:
        if self._num_dynamic_obstacles == 0:
            return

        goal_dist = torch.linalg.norm(self._dynamic_obstacle_positions_w - self._dynamic_obstacle_goals_w, dim=1)
        new_goal_mask = (goal_dist < self.cfg.dynamic_obstacle_goal_threshold) | (self._dynamic_obstacle_step_count == 0)
        self._sample_dynamic_obstacle_goals(new_goal_mask)

        resample_interval = max(1, int(round(self.cfg.dynamic_obstacle_velocity_resample_s / self.cfg.physics_dt)))
        if self._dynamic_obstacle_step_count % resample_interval == 0:
            self._resample_dynamic_obstacle_velocities()

        self._dynamic_obstacle_positions_w += self._dynamic_obstacle_velocities_w * self.cfg.physics_dt
        self._dynamic_obstacle_positions_w[:, 0].clamp_(-self.cfg.map_half_extent, self.cfg.map_half_extent)
        self._dynamic_obstacle_positions_w[:, 1].clamp_(-self.cfg.map_half_extent, self.cfg.map_half_extent)
        self._dynamic_obstacle_positions_w[: self._dynamic_2d_start_idx, 2].clamp_(0.0, self.cfg.flight_max_height)
        self._dynamic_obstacle_positions_w[self._dynamic_2d_start_idx :, 2] = self.cfg.dynamic_obstacle_2d_height * 0.5
        visual_update_interval = int(self.cfg.dynamic_obstacle_visual_update_interval)
        if visual_update_interval > 0 and self._dynamic_obstacle_step_count % visual_update_interval == 0:
            self._update_dynamic_obstacle_visuals()
        self._dynamic_obstacle_step_count += 1

    def _spawn_boundary_walls(self) -> None:
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

        robot_view_cfg = robot_spawn_cfg.copy()
        robot_view_cfg.spawn = None
        self._robot = Articulation(robot_view_cfg)

    def _disable_robot_collisions(self) -> None:
        collision_cfg = sim_utils.CollisionPropertiesCfg(collision_enabled=False)
        for idx in range(self.num_envs):
            sim_schemas.modify_collision_properties(f"/World/Robot_{idx:02d}/OmniNxt", collision_cfg)

    def _compute_closest_obstacles_directional(self) -> tuple[torch.Tensor, torch.Tensor]:
        drone_pos_w = self.robot.data.root_pos_w
        drone_quat_w = self.robot.data.root_quat_w
        to_obstacles = self._obstacle_positions_w.unsqueeze(0) - drone_pos_w.unsqueeze(1)
        distances_sq = torch.sum(torch.square(to_obstacles), dim=2)

        _, closest_indices = torch.topk(distances_sq, self.cfg.num_closest_obstacles, dim=1, largest=False)
        batch_indices = torch.arange(self.num_envs, device=self.device).unsqueeze(1).expand(-1, self.cfg.num_closest_obstacles)
        closest_vectors = to_obstacles[batch_indices, closest_indices]
        closest_radii = self._obstacle_radii[closest_indices]
        closest_distances = torch.sqrt(distances_sq[batch_indices, closest_indices].clamp_min(1e-12)) - closest_radii
        closest_vectors_norm = closest_vectors / torch.sqrt(
            torch.sum(torch.square(closest_vectors), dim=2, keepdim=True).clamp_min(1e-12)
        )

        num_closest = self.cfg.num_closest_obstacles
        closest_vectors_flat = closest_vectors_norm.reshape(self.num_envs * num_closest, 3)
        drone_quat_expanded = drone_quat_w.unsqueeze(1).expand(-1, num_closest, -1).reshape(self.num_envs * num_closest, 4)
        directions_body = quat_apply_inverse(drone_quat_expanded, closest_vectors_flat).reshape(
            self.num_envs, num_closest, 3
        )
        distances_normalized = (closest_distances / self.cfg.obstacle_detection_range).clamp(0.0, 1.0)
        return directions_body, distances_normalized

    def _compute_closest_obstacle_signed_distance(self, drone_pos_w: torch.Tensor) -> torch.Tensor:
        rel_xy = drone_pos_w[:, None, :2] - self._obstacle_positions_w[None, :, :2]
        rel_z = torch.abs(drone_pos_w[:, None, 2] - self._obstacle_positions_w[None, :, 2])
        radial = torch.linalg.norm(rel_xy, dim=2) - self._obstacle_radii[None, :]
        vertical = rel_z - self._obstacle_heights[None, :] * 0.5
        outside = torch.stack((radial, vertical), dim=-1).clamp_min(0.0)
        inside = torch.maximum(radial, vertical).clamp_max(0.0)
        signed_distance = torch.linalg.norm(outside, dim=-1) + inside
        return signed_distance.min(dim=1).values

    def _compute_forward_obstacle_distance(self, drone_pos_w: torch.Tensor, drone_quat_w: torch.Tensor) -> torch.Tensor:
        rel_vectors_w = self._obstacle_positions_w.unsqueeze(0) - drone_pos_w.unsqueeze(1)
        rel_vectors_b = quat_apply_inverse(
            drone_quat_w.unsqueeze(1).expand(-1, self.cfg.num_obstacles, -1).reshape(self.num_envs * self.cfg.num_obstacles, 4),
            rel_vectors_w.reshape(self.num_envs * self.cfg.num_obstacles, 3),
        ).reshape(self.num_envs, self.cfg.num_obstacles, 3)
        surface_distances = torch.sqrt(torch.sum(torch.square(rel_vectors_w), dim=2).clamp_min(1e-12)) - self._obstacle_radii.unsqueeze(0)
        forward_mask = rel_vectors_b[:, :, 0] > 0.0
        forward_distances = torch.where(
            forward_mask,
            surface_distances,
            torch.full_like(surface_distances, self.cfg.obstacle_detection_range),
        )
        return forward_distances.min(dim=1).values.clamp(0.0, self.cfg.obstacle_detection_range)

    def _compute_target_ray_obstacle_distance(self, drone_pos_w: torch.Tensor, target_direction_w: torch.Tensor) -> torch.Tensor:
        rel_vectors_w = self._obstacle_positions_w.unsqueeze(0) - drone_pos_w.unsqueeze(1)
        rel_distances = torch.sqrt(torch.sum(torch.square(rel_vectors_w), dim=2).clamp_min(1e-12))
        target_alignment = torch.sum(rel_vectors_w * target_direction_w.unsqueeze(1), dim=2) / rel_distances.clamp_min(1e-6)
        in_target_cone = (target_alignment > 0.75) & (torch.sum(rel_vectors_w * target_direction_w.unsqueeze(1), dim=2) > 0.0)
        surface_distances = rel_distances - self._obstacle_radii.unsqueeze(0)
        target_ray_distances = torch.where(
            in_target_cone,
            surface_distances,
            torch.full_like(surface_distances, self.cfg.obstacle_detection_range),
        )
        return target_ray_distances.min(dim=1).values.clamp(0.0, self.cfg.obstacle_detection_range)

    def _compute_closest_dynamic_obstacle_features(
        self, drone_pos_w: torch.Tensor, target_direction_w: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feature_shape = (self.num_envs, self.cfg.num_closest_dynamic_obstacles, 10)
        if self._num_dynamic_obstacles == 0:
            features = torch.zeros(feature_shape, device=self.device)
            collision = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            distance = torch.full((self.num_envs,), self.cfg.obstacle_detection_range, device=self.device)
            return features, collision, distance

        rel_pos = self._dynamic_obstacle_positions_w.unsqueeze(0) - drone_pos_w.unsqueeze(1)
        rel_pos[:, self._dynamic_2d_start_idx :, 2] = 0.0
        distance_2d_all = torch.linalg.norm(rel_pos[:, :, :2], dim=2)
        closest_count = min(self.cfg.num_closest_dynamic_obstacles, self._num_dynamic_obstacles)
        _, closest_idx = torch.topk(distance_2d_all, closest_count, dim=1, largest=False)
        range_mask = distance_2d_all.gather(1, closest_idx) > self.cfg.obstacle_detection_range

        gather_idx = closest_idx.unsqueeze(-1).expand(-1, -1, 3)
        closest_rel_pos = torch.gather(rel_pos, 1, gather_idx)
        target_yaw = torch.atan2(target_direction_w[:, 1], target_direction_w[:, 0])
        cos_yaw = torch.cos(target_yaw).unsqueeze(1)
        sin_yaw = torch.sin(target_yaw).unsqueeze(1)
        rel_x = cos_yaw * closest_rel_pos[:, :, 0] + sin_yaw * closest_rel_pos[:, :, 1]
        rel_y = -sin_yaw * closest_rel_pos[:, :, 0] + cos_yaw * closest_rel_pos[:, :, 1]
        rel_goal_frame = torch.stack((rel_x, rel_y, closest_rel_pos[:, :, 2]), dim=-1)
        rel_goal_frame[range_mask] = 0.0

        closest_distance = torch.linalg.norm(closest_rel_pos, dim=-1, keepdim=True).clamp_min(1e-6)
        rel_goal_frame_norm = rel_goal_frame / closest_distance
        distance_2d = torch.linalg.norm(rel_goal_frame[:, :, :2], dim=-1, keepdim=True)
        distance_z = rel_goal_frame[:, :, 2].unsqueeze(-1)

        closest_vel = self._dynamic_obstacle_velocities_w[closest_idx]
        closest_vel[range_mask] = 0.0
        vel_x = cos_yaw * closest_vel[:, :, 0] + sin_yaw * closest_vel[:, :, 1]
        vel_y = -sin_yaw * closest_vel[:, :, 0] + cos_yaw * closest_vel[:, :, 1]
        vel_goal_frame = torch.stack((vel_x, vel_y, closest_vel[:, :, 2]), dim=-1)

        closest_size = self._dynamic_obstacle_sizes[closest_idx]
        width = closest_size[:, :, 0].unsqueeze(-1)
        height = closest_size[:, :, 2].unsqueeze(-1)
        width_category = width / self._dynamic_width_res - 1.0
        height_category = torch.where(
            height > self.cfg.dynamic_obstacle_3d_height,
            torch.zeros_like(height),
            height,
        )
        width_category[range_mask] = 0.0
        height_category[range_mask] = 0.0

        features = torch.cat(
            (
                rel_goal_frame_norm,
                distance_2d,
                distance_z,
                vel_goal_frame,
                width_category,
                height_category,
            ),
            dim=-1,
        )

        if closest_count < self.cfg.num_closest_dynamic_obstacles:
            padded = torch.zeros(feature_shape, device=self.device)
            padded[:, :closest_count] = features
            features = padded

        collision_distance_2d = torch.linalg.norm(closest_rel_pos[:, :, :2], dim=-1, keepdim=True)
        collision_distance_z = torch.abs(closest_rel_pos[:, :, 2]).unsqueeze(-1)
        collision_distance_2d[range_mask] = float("inf")
        collision_distance_z[range_mask] = float("inf")
        collision_2d = collision_distance_2d <= (width * 0.5 + self.cfg.dynamic_obstacle_collision_margin)
        collision_z = collision_distance_z <= (height * 0.5 + self.cfg.dynamic_obstacle_collision_margin)
        collision = torch.any((collision_2d & collision_z).squeeze(-1), dim=1)

        reward_distance = torch.linalg.norm(closest_rel_pos, dim=-1) - closest_size[:, :, 0] * 0.5
        reward_distance[range_mask] = self.cfg.obstacle_detection_range
        closest_reward_distance = reward_distance.min(dim=1).values
        return features, collision, closest_reward_distance

    def _compute_wall_signed_distance(self, drone_pos_w: torch.Tensor) -> torch.Tensor:
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
        distances = torch.linalg.norm(self._target_positions_w[env_ids] - self.robot.data.root_pos_w[env_ids], dim=1)
        self._prev_dist_to_target[env_ids] = distances

    def _get_observations(self) -> dict[str, torch.Tensor]:
        root_pos_w = self.robot.data.root_pos_w
        root_quat_w = self.robot.data.root_quat_w
        root_lin_vel_b = quat_apply_inverse(root_quat_w, self.robot.data.root_lin_vel_w)
        root_ang_vel_b = quat_apply_inverse(root_quat_w, self.robot.data.root_ang_vel_w)
        projected_gravity_b = quat_apply_inverse(root_quat_w, self._gravity_vec_w)
        target_vec_w = self._target_positions_w - root_pos_w
        target_pos_b = quat_apply_inverse(root_quat_w, target_vec_w)
        target_direction_w = target_vec_w / torch.linalg.norm(target_vec_w, dim=1, keepdim=True).clamp_min(1e-6)

        obstacle_directions, obstacle_distances = self._compute_closest_obstacles_directional()
        dynamic_obstacle_features, _, _ = self._compute_closest_dynamic_obstacle_features(root_pos_w, target_direction_w)
        forward_obstacle_distance = self._compute_forward_obstacle_distance(root_pos_w, root_quat_w)
        target_ray_obstacle_distance = self._compute_target_ray_obstacle_distance(root_pos_w, target_direction_w)
        obs = torch.cat(
            [
                root_lin_vel_b,
                root_ang_vel_b,
                projected_gravity_b,
                target_pos_b,
                self._cmd_vel_b,
                obstacle_directions.reshape(self.num_envs, -1),
                obstacle_distances,
                dynamic_obstacle_features.reshape(self.num_envs, -1),
                (forward_obstacle_distance / self.cfg.obstacle_detection_range).unsqueeze(1),
                (target_ray_obstacle_distance / self.cfg.obstacle_detection_range).unsqueeze(1),
                self._target_reached.float().unsqueeze(1),
            ],
            dim=-1,
        )
        return {"policy": obs}

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        del options
        if seed is not None:
            self.seed(seed)
        if not self._built:
            self._build()
        env_ids = torch.arange(self.num_envs, device=self.device)
        obs = self._reset_idx(env_ids)
        return obs, {}

    def _reset_idx(self, env_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        if env_ids.numel() == 0:
            return self._get_observations()

        active_ids = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        active_ids[env_ids] = False
        avoid_positions_xy = self.robot.data.root_pos_w[active_ids, :2] if self._built else None

        start_pos = self._sample_edge_positions_with_clearance(
            len(env_ids),
            self.cfg.spawn_edge_distance,
            self.cfg.spawn_min_height,
            self.cfg.spawn_max_height,
            min_separation=self.cfg.drone_spawn_min_separation,
            avoid_positions_xy=avoid_positions_xy,
        )
        # Match NavRL: the target is sampled independently from the same four map edges,
        # instead of being forced onto the edge opposite the drone spawn.
        target_pos = self._sample_edge_positions_with_clearance(
            len(env_ids),
            self.cfg.target_spawn_range,
            self.cfg.target_min_height,
            self.cfg.target_max_height,
            min_separation=self.cfg.drone_spawn_min_separation,
        )

        root_state = self.robot.data.default_root_state.clone()
        diff = target_pos - start_pos
        yaw = torch.atan2(diff[:, 1], diff[:, 0])
        cy = torch.cos(yaw * 0.5)
        sy = torch.sin(yaw * 0.5)

        root_state[env_ids, :3] = start_pos
        root_state[env_ids, 3] = cy
        root_state[env_ids, 4] = 0.0
        root_state[env_ids, 5] = 0.0
        root_state[env_ids, 6] = sy
        root_state[env_ids, 7:] = 0.0

        self.robot.write_root_state_to_sim(root_state[env_ids], env_ids=env_ids)
        self.robot.write_joint_state_to_sim(
            self.robot.data.default_joint_pos[env_ids],
            self.robot.data.default_joint_vel[env_ids],
            env_ids=env_ids,
        )

        self._target_positions_w[env_ids] = target_pos
        self._target_reached[env_ids] = False
        self._done_buf[env_ids] = False
        self._rew_buf[env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0
        self._cmd_vel_b[env_ids] = 0.0
        self._target_cmd_vel_b[env_ids] = 0.0
        initial_yaw_deg = torch.rad2deg(yaw)
        self._cmd_yaw_deg[env_ids] = initial_yaw_deg
        self._target_cmd_yaw_deg[env_ids] = initial_yaw_deg
        self._actions[env_ids] = 0.0
        self._prev_actions[env_ids] = 0.0
        self._prev_cmd_vel_b[env_ids] = 0.0
        self._prev_velocity_cmd_frame[env_ids] = 0.0

        controller_state = {
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
        self._controller.yaw_setpoint[env_ids] = initial_yaw_deg

        # Refresh local robot buffers after writing the reset state without advancing physics.
        self.robot.update(0.0)
        self._update_prev_target_distance(env_ids)
        for key in self._episode_sums:
            self._episode_sums[key][env_ids] = 0.0
        return self._get_observations()

    def _update_command_from_actions(self, actions: torch.Tensor) -> None:
        self._target_cmd_vel_b[:, :2] = actions[:, :2] * self.cfg.cmd_body_vel_xy_max
        self._target_cmd_vel_b[:, 2] = actions[:, 2] * self.cfg.cmd_vel_z_max

        alpha = max(0.0, min(float(self.cfg.cmd_vel_smoothing_alpha), 1.0))
        if alpha >= 1.0:
            next_cmd = self._target_cmd_vel_b
        else:
            next_cmd = self._cmd_vel_b + alpha * (self._target_cmd_vel_b - self._cmd_vel_b)

        xy_rate_limit = float(self.cfg.cmd_body_vel_xy_rate_limit)
        z_rate_limit = float(self.cfg.cmd_vel_z_rate_limit)
        if xy_rate_limit > 0.0 or z_rate_limit > 0.0:
            delta = (next_cmd - self._cmd_vel_b).clamp(-self._cmd_vel_max_delta, self._cmd_vel_max_delta)
            next_cmd = self._cmd_vel_b + delta

        self._cmd_vel_b[:] = next_cmd

        self._target_cmd_yaw_deg[:] = self._wrap_angle_deg(actions[:, 3] * self.cfg.cmd_yaw_angle_max)
        yaw_alpha = max(0.0, min(float(self.cfg.cmd_yaw_smoothing_alpha), 1.0))
        yaw_delta = self._wrap_angle_deg(self._target_cmd_yaw_deg - self._cmd_yaw_deg)
        if yaw_alpha >= 1.0:
            next_yaw_delta = yaw_delta
        else:
            next_yaw_delta = yaw_alpha * yaw_delta
        if float(self.cfg.cmd_yaw_rate_limit) > 0.0:
            next_yaw_delta = next_yaw_delta.clamp(-self._cmd_yaw_max_delta, self._cmd_yaw_max_delta)
        self._cmd_yaw_deg[:] = self._wrap_angle_deg(self._cmd_yaw_deg + next_yaw_delta)

    def _wrap_angle_deg(self, angle_deg: torch.Tensor) -> torch.Tensor:
        return torch.remainder(angle_deg + 180.0, 360.0) - 180.0

    def step(self, actions: torch.Tensor):
        if not self._built:
            raise RuntimeError("Call reset() before step().")

        actions = actions.to(self.device).clamp(-1.0, 1.0)
        prev_actions = self._prev_actions.clone()
        prev_cmd_vel_b = self._prev_cmd_vel_b.clone()
        prev_velocity_cmd_frame = self._prev_velocity_cmd_frame.clone()
        self._actions[:] = actions
        self._update_command_from_actions(actions)

        root_quat_w = self.robot.data.root_quat_w
        root_ang_vel_b = quat_apply_inverse(root_quat_w, self.robot.data.root_ang_vel_w)
        self._controller.set_velocity_setpoint(
            vx=self._cmd_vel_b[:, 0],
            vy=self._cmd_vel_b[:, 1],
            vz=self._cmd_vel_b[:, 2],
            velocity_body=True,
        )
        self._controller.yaw_setpoint[:] = self._cmd_yaw_deg
        force, torque = self._controller.compute(
            {
                "position": self.robot.data.root_pos_w,
                "velocity": self.robot.data.root_lin_vel_w,
                "attitude": _quat_to_euler_deg(root_quat_w),
                "angular_velocity": torch.rad2deg(root_ang_vel_b),
            }
        )
        self._thrust[:, 0, :] = force
        self._moment[:, 0, :] = torque

        for _ in range(self.cfg.decimation):
            self._move_dynamic_obstacles()
            self.robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)
            self.robot.write_data_to_sim()
            self.sim.step()
            self.robot.update(self.cfg.physics_dt)

        self.common_step_counter += 1
        self.episode_length_buf += 1

        root_quat_w = self.robot.data.root_quat_w
        root_rpy_deg = _quat_to_euler_deg(root_quat_w)
        root_lin_vel_b = quat_apply_inverse(root_quat_w, self.robot.data.root_lin_vel_w)
        root_ang_vel_b = quat_apply_inverse(root_quat_w, self.robot.data.root_ang_vel_w)
        target_vec_w = self._target_positions_w - self.robot.data.root_pos_w
        distance_to_target = torch.linalg.norm(target_vec_w, dim=1)
        target_direction = target_vec_w / distance_to_target.unsqueeze(1).clamp_min(1e-6)
        target_yaw_deg = torch.rad2deg(torch.atan2(target_vec_w[:, 1], target_vec_w[:, 0]))
        yaw_error_deg = self._wrap_angle_deg(target_yaw_deg - root_rpy_deg[:, 2])
        yaw_alignment = 0.5 * (torch.cos(torch.deg2rad(yaw_error_deg)) + 1.0)
        yaw_error_norm = torch.abs(yaw_error_deg) / 180.0

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
        vel_tracking = torch.exp(-self.cfg.vel_tracking_exp_scale * vel_tracking_sq_error)
        action_rate = torch.sum(torch.square(actions - prev_actions), dim=1)
        cmd_rate = torch.sum(torch.square(self._cmd_vel_b - prev_cmd_vel_b), dim=1)
        velocity_accel = torch.sum(torch.square(velocity_cmd_frame - prev_velocity_cmd_frame), dim=1)
        speed_xy = torch.linalg.norm(self.robot.data.root_lin_vel_w[:, :2], dim=1)
        overspeed = torch.square((speed_xy - self.cfg.cmd_body_vel_xy_max).clamp_min(0.0))
        ang_vel = torch.sum(torch.square(root_ang_vel_b), dim=1)
        distance_reward = 1.0 - torch.tanh(distance_to_target / 2.0)
        target_velocity = torch.sum(self.robot.data.root_lin_vel_w * target_direction, dim=1).clamp(-1.0, 2.0)
        progress_reward = (self._prev_dist_to_target - distance_to_target).clamp(-1.0, 1.0)
        self._prev_dist_to_target = distance_to_target.clone()

        newly_reached = (distance_to_target < self.cfg.target_reach_threshold) & (~self._target_reached)
        target_bonus = torch.zeros(self.num_envs, device=self.device)
        target_bonus[newly_reached] = self.cfg.target_reached_bonus
        self._target_reached |= newly_reached

        _, dynamic_obstacle_collision, closest_dynamic_obstacle_distance = self._compute_closest_dynamic_obstacle_features(
            self.robot.data.root_pos_w, target_direction
        )
        closest_obstacle_distance = self._compute_closest_obstacle_signed_distance(self.robot.data.root_pos_w)
        closest_wall_distance = self._compute_wall_signed_distance(self.robot.data.root_pos_w)
        closest_hazard_distance = torch.minimum(
            torch.minimum(closest_obstacle_distance, closest_wall_distance),
            closest_dynamic_obstacle_distance,
        )
        trigger_distance = max(float(self.cfg.obstacle_proximity_trigger_distance), 1.0e-6)
        obstacle_proximity = ((trigger_distance - closest_hazard_distance) / trigger_distance).clamp(0.0, 1.0)

        timeout = self.episode_length_buf >= self.max_episode_length
        obstacle_collision = closest_obstacle_distance < self.cfg.obstacle_collision_margin
        wall_collision = closest_wall_distance < self.cfg.obstacle_collision_margin
        too_low = self.robot.data.root_pos_w[:, 2] < self.cfg.flight_min_height
        too_high = self.robot.data.root_pos_w[:, 2] > self.cfg.flight_max_height
        height_violation = too_low | too_high

        rewards = {
            "vel_tracking": vel_tracking * self.cfg.vel_tracking_reward_scale * self.step_dt,
            "ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
            "distance_to_target": distance_reward * self.cfg.distance_to_target_reward_scale * self.step_dt,
            "target_velocity": target_velocity.clamp(min=0.0) * self.cfg.target_velocity_reward_scale * self.step_dt,
            "target_bonus": target_bonus,
            "obstacle_proximity": obstacle_proximity * self.cfg.obstacle_proximity_reward_scale * self.step_dt,
            "progress": progress_reward * self.cfg.progress_reward_scale * self.step_dt,
            "static_collision_penalty": obstacle_collision.float() * self.cfg.static_collision_penalty,
            "dynamic_collision_penalty": dynamic_obstacle_collision.float() * self.cfg.dynamic_collision_penalty,
            "wall_collision_penalty": wall_collision.float() * self.cfg.wall_collision_penalty,
            "height_violation_penalty": height_violation.float() * self.cfg.height_violation_penalty,
            "action_rate": action_rate * self.cfg.action_rate_penalty_scale * self.step_dt,
            "cmd_rate": cmd_rate * self.cfg.cmd_rate_penalty_scale * self.step_dt,
            "velocity_accel": velocity_accel * self.cfg.velocity_accel_penalty_scale * self.step_dt,
            "overspeed": overspeed * self.cfg.overspeed_penalty_scale * self.step_dt,
            "yaw_alignment": yaw_alignment * self.cfg.yaw_alignment_reward_scale * self.step_dt,
            "yaw_error": yaw_error_norm * self.cfg.yaw_error_penalty_scale * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        for key, value in rewards.items():
            self._episode_sums[key] += value
        self._episode_sums["vel_tracking_error"] += vel_tracking_error * self.step_dt
        self._rew_buf[:] = reward
        self._prev_actions[:] = actions
        self._prev_cmd_vel_b[:] = self._cmd_vel_b
        self._prev_velocity_cmd_frame[:] = velocity_cmd_frame

        died = too_low | too_high | obstacle_collision | wall_collision | dynamic_obstacle_collision
        terminated = self._target_reached | died
        self._done_buf[:] = terminated | timeout

        extras = {
            "time_outs": timeout.clone(),
            "target_reached": self._target_reached.clone(),
            "terminated": terminated.clone(),
            "obstacle_collision": obstacle_collision.clone(),
            "dynamic_obstacle_collision": dynamic_obstacle_collision.clone(),
            "wall_collision": wall_collision.clone(),
            "too_low": too_low.clone(),
            "too_high": too_high.clone(),
            "closest_obstacle_distance": closest_obstacle_distance.clone(),
            "closest_dynamic_obstacle_distance": closest_dynamic_obstacle_distance.clone(),
            "closest_wall_distance": closest_wall_distance.clone(),
            "distance_to_target": distance_to_target.clone(),
            "yaw_error_deg": yaw_error_deg.clone(),
            "cmd_yaw_deg": self._cmd_yaw_deg.clone(),
        }

        done_ids = torch.nonzero(self._done_buf, as_tuple=False).squeeze(-1)
        if done_ids.numel() > 0:
            log = {}
            for key in self._episode_sums:
                episodic_sum_avg = torch.mean(self._episode_sums[key][done_ids])
                if key == "vel_tracking_error":
                    log["Metrics/avg_vel_tracking_error"] = episodic_sum_avg / self.max_episode_length_s
                else:
                    log[f"Episode_Reward/{key}"] = episodic_sum_avg / self.max_episode_length_s
            log["Episode_Termination/died"] = torch.count_nonzero(died[done_ids]).item()
            log["Episode_Termination/time_out"] = torch.count_nonzero(timeout[done_ids]).item()
            log["Episode_Termination/too_low"] = torch.count_nonzero(too_low[done_ids]).item()
            log["Episode_Termination/too_high"] = torch.count_nonzero(too_high[done_ids]).item()
            log["Episode_Termination/obstacle_collision"] = torch.count_nonzero(obstacle_collision[done_ids]).item()
            log["Episode_Termination/static_obstacle_collision"] = torch.count_nonzero(
                obstacle_collision[done_ids]
            ).item()
            log["Episode_Termination/dynamic_obstacle_collision"] = torch.count_nonzero(
                dynamic_obstacle_collision[done_ids]
            ).item()
            log["Episode_Termination/wall_collision"] = torch.count_nonzero(wall_collision[done_ids]).item()
            log["Metrics/success_rate"] = self._target_reached[done_ids].float().mean().item()
            log["Metrics/failure_rate"] = (died[done_ids] & (~self._target_reached[done_ids])).float().mean().item()
            log["Metrics/too_low_rate"] = too_low[done_ids].float().mean().item()
            log["Metrics/too_high_rate"] = too_high[done_ids].float().mean().item()
            log["Metrics/obstacle_collision_rate"] = obstacle_collision[done_ids].float().mean().item()
            log["Metrics/static_obstacle_collision_rate"] = obstacle_collision[done_ids].float().mean().item()
            log["Metrics/dynamic_obstacle_collision_rate"] = dynamic_obstacle_collision[done_ids].float().mean().item()
            log["Metrics/total_obstacle_collision_rate"] = (
                obstacle_collision[done_ids] | dynamic_obstacle_collision[done_ids]
            ).float().mean().item()
            log["Metrics/wall_collision_rate"] = wall_collision[done_ids].float().mean().item()
            log["Metrics/success_rate_all_envs"] = self._target_reached.float().mean().item()
            log["Metrics/avg_closest_hazard_distance"] = closest_hazard_distance.mean().item()
            extras["log"] = log

        rew = self._rew_buf.clone()
        terminated_out = terminated.clone()
        truncated_out = timeout.clone()
        if done_ids.numel() > 0:
            self._reset_idx(done_ids)
        obs = self._get_observations()
        return obs, rew, terminated_out, truncated_out, extras

    def render(self):
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
            return np.zeros((720, 1280, 3), dtype=np.uint8)
        return rgb_data[:, :, :3]

    def close(self):
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
