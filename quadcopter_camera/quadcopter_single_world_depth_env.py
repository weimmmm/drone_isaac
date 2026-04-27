from __future__ import annotations

"""Single-world multi-drone depth environment.

This file is the forward path for keeping real depth cameras while avoiding the
`num_envs`/env-cloning assumptions that kept crashing the Direct/ManagerBased
variants. It does not register into gym yet. The intent is:

- one real 40x40 world
- N manually spawned OmniNxt robots
- one forward depth camera per robot
- shared static obstacle map
- per-robot observations / rewards / done flags
"""

from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
import torch
from PIL import Image, ImageDraw
import omni.replicator.core as rep

import isaaclab.sim as sim_utils
from isaaclab.sim import schemas as sim_schemas
import omni.kit.commands
import omni.usd
from isaaclab.assets import Articulation
from isaaclab.sensors.camera import TiledCamera, TiledCameraCfg
from isaaclab.utils.math import quat_apply_inverse
from pxr import Sdf, UsdGeom

from assets.omninxt.omninxt import OMNINXT_CFG
from controller import CrazyflieController, config as controller_config


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
class QuadcopterSingleWorldDepthEnvCfg:
    @dataclass
    class SimCfg:
        device: str = "cuda:0"

    @dataclass
    class SceneCfg:
        num_envs: int = 8

    num_robots: int = 8
    map_half_extent: float = 20.0
    wall_half_extent: float = 30.0
    wall_height: float = 5.0
    wall_thickness: float = 0.3
    num_obstacles: int = 100
    obstacle_height: float = 1.5
    obstacle_radius: float = 0.15
    obstacle_spawn_range: float = 20.0
    obstacle_safe_zone: float = 1.0
    obstacle_min_separation: float = 0.6
    obstacle_collision_margin: float = 0.10
    target_obstacle_clearance: float = 2.0

    depth_camera_width: int = 32
    depth_camera_height: int = 24
    num_stacked_depth_frames: int = 2
    depth_camera_near_clip: float = 0.2
    depth_camera_far_clip: float = 8.0

    spawn_edge_distance: float = 23.0
    target_spawn_range: float = 23.0
    spawn_min_height: float = 0.5
    spawn_max_height: float = 2.5
    target_min_height: float = 0.5
    target_max_height: float = 2.5
    target_reach_threshold: float = 0.5

    cmd_body_vel_xy_max: float = 2.0
    cmd_vel_z_max: float = 0.5
    drone_spawn_min_separation: float = 1.0
    ang_vel_obs_scale: float = 0.25
    target_pos_obs_scale: float = 0.1
    target_pos_obs_clip: float = 2.0

    progress_reward_scale: float = 15.0
    target_velocity_reward_scale: float = 4.0
    distance_to_target_reward_scale: float = 8.0
    target_reached_bonus: float = 80.0
    obstacle_proximity_reward_scale: float = -6.0
    ang_vel_reward_scale: float = -0.01
    vel_tracking_reward_scale: float = 0.5
    vel_tracking_exp_scale: float = 4.0
    safety_static_reward_scale: float = 0.5
    smoothness_penalty_scale: float = 0.0
    height_penalty_scale: float = 0.0
    depth_sector_reward_scale: float = 0.0

    max_episode_steps: int = 4500
    physics_dt: float = 1.0 / 100.0
    control_decimation: int = 2
    device: str = "cuda:0"
    is_finite_horizon: bool = False
    viewer_eye: tuple[float, float, float] = (-30.0, 0.0, 80.0)
    viewer_lookat: tuple[float, float, float] = (0.0, 0.0, 0.0)
    viewer_resolution: tuple[int, int] = (1280, 720)
    viewer_cam_prim_path: str = "/OmniverseKit_Persp"
    sim: SimCfg = field(default_factory=SimCfg)
    scene: SceneCfg = field(default_factory=SceneCfg)

    observation_space: dict = field(
        default_factory=lambda: {
            "state": 16,
            "depth": (2, 24, 32),
        }
    )
    action_space: int = 3

    def __post_init__(self):
        self.scene.num_envs = self.num_robots
        self.sim.device = self.device


class QuadcopterSingleWorldDepthEnv:
    """Standalone single-world environment.

    This is not a gym.Env yet. The focus is to make the single-world scene and
    real depth cameras work first before attaching PPO wrappers.
    """

    def __init__(self, cfg: QuadcopterSingleWorldDepthEnvCfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.num_robots = cfg.num_robots
        self.step_dt = cfg.physics_dt * cfg.control_decimation
        self._built = False

        self.actions = torch.zeros((self.num_robots, cfg.action_space), device=self.device)
        self.cmd_vel_b = torch.zeros((self.num_robots, 3), device=self.device)
        self.rew_buf = torch.zeros(self.num_robots, device=self.device)
        self.done_buf = torch.zeros(self.num_robots, dtype=torch.bool, device=self.device)
        self.episode_length_buf = torch.zeros(self.num_robots, dtype=torch.long, device=self.device)
        self.common_step_counter = 0
        self.max_episode_length_s = self.cfg.max_episode_steps * self.step_dt

        self.target_positions_w = torch.zeros((self.num_robots, 3), device=self.device)
        self.target_reached = torch.zeros(self.num_robots, dtype=torch.bool, device=self.device)
        self.prev_dist_to_target = torch.zeros(self.num_robots, device=self.device)
        self.height_range = torch.zeros((self.num_robots, 2), device=self.device)
        self.prev_lin_vel_w = torch.zeros((self.num_robots, 3), device=self.device)
        self._robot_mass = 0.0

        self.obstacle_positions_w = torch.zeros((cfg.num_obstacles, 3), device=self.device)
        self.depth_frame_stack = torch.zeros(
            (
                self.num_robots,
                cfg.num_stacked_depth_frames,
                cfg.depth_camera_height,
                cfg.depth_camera_width,
            ),
            device=self.device,
        )
        self.depth_stack_needs_fill = torch.ones(self.num_robots, dtype=torch.bool, device=self.device)

        self._sim: sim_utils.SimulationContext | None = None
        self._robot: Articulation | None = None
        self._depth_camera: TiledCamera | None = None
        self._body_id: torch.Tensor | None = None
        self._obstacle_paths: list[str] = []
        self._wall_paths: list[str] = []
        self._thrust = torch.zeros(self.num_robots, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_robots, 1, 3, device=self.device)
        self._controller = CrazyflieController(
            num_envs=self.num_robots,
            device=self.device,
            attitude_dt=self.step_dt,
            position_dt=self.step_dt,
        )
        self._cmd_obs_scale = torch.tensor(
            [
                1.0 / self.cfg.cmd_body_vel_xy_max,
                1.0 / self.cfg.cmd_body_vel_xy_max,
                1.0 / self.cfg.cmd_vel_z_max,
            ],
            device=self.device,
        )
        self._episode_sums = {
            key: torch.zeros(self.num_robots, dtype=torch.float, device=self.device)
            for key in [
                "progress",
                "target_velocity",
                "target_bonus",
                "safety_static",
            ]
        }

    @property
    def robot(self) -> Articulation:
        if self._robot is None:
            raise RuntimeError("Environment not built yet.")
        return self._robot

    @property
    def sim(self) -> sim_utils.SimulationContext:
        if self._sim is None:
            raise RuntimeError("Environment not built yet.")
        return self._sim

    @property
    def depth_camera(self) -> TiledCamera:
        if self._depth_camera is None:
            raise RuntimeError("Environment not built yet.")
        return self._depth_camera

    def _sample_edge_positions(
        self, num_samples: int, lateral_range: float, min_height: float, max_height: float
    ) -> torch.Tensor:
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

    def _sample_edge_positions_with_clearance(
        self,
        num_samples: int,
        lateral_range: float,
        min_height: float,
        max_height: float,
        min_separation: float = 0.0,
        avoid_obstacles: bool = False,
        obstacle_clearance: float | None = None,
    ) -> torch.Tensor:
        positions = torch.zeros(num_samples, 3, device=self.device)
        obstacle_clearance = self.cfg.target_obstacle_clearance if obstacle_clearance is None else obstacle_clearance
        for idx in range(num_samples):
            best_candidate = None
            best_score = None
            for _ in range(128):
                candidate = self._sample_edge_positions(1, lateral_range, min_height, max_height)[0]
                valid = True
                score = torch.tensor(float("inf"), device=self.device)
                if min_separation > 0.0 and idx > 0:
                    d_prev = torch.linalg.norm(candidate.unsqueeze(0) - positions[:idx], dim=1)
                    min_prev_dist = d_prev.min()
                    valid = bool(valid and (min_prev_dist >= min_separation))
                    score = torch.minimum(score, min_prev_dist)
                if avoid_obstacles and self.cfg.num_obstacles > 0:
                    d_obs = torch.linalg.norm(candidate[:2].unsqueeze(0) - self.obstacle_positions_w[:, :2], dim=1)
                    min_obs_dist = d_obs.min()
                    valid = bool(valid and (min_obs_dist >= obstacle_clearance))
                    score = torch.minimum(score, min_obs_dist)
                if best_candidate is None or score > best_score:
                    best_candidate = candidate
                    best_score = score
                if valid:
                    best_candidate = candidate
                    break
            positions[idx] = best_candidate
        return positions

    def _sample_obstacles(self) -> torch.Tensor:
        placed_xy = torch.zeros(self.cfg.num_obstacles, 2, device=self.device)
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
                    valid = bool(valid and (min_dist >= self.cfg.obstacle_min_separation))

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

        positions = torch.zeros(self.cfg.num_obstacles, 3, device=self.device)
        positions[:, :2] = placed_xy
        positions[:, 2] = self.cfg.obstacle_height / 2
        return positions

    def _compute_closest_obstacle_signed_distance(self, drone_pos_w: torch.Tensor) -> torch.Tensor:
        rel_xy = drone_pos_w[:, None, :2] - self.obstacle_positions_w[None, :, :2]
        rel_z = torch.abs(drone_pos_w[:, None, 2] - self.obstacle_positions_w[None, :, 2])
        radial = torch.linalg.norm(rel_xy, dim=2) - self.cfg.obstacle_radius
        vertical = rel_z - self.cfg.obstacle_height * 0.5
        outside = torch.stack((radial, vertical), dim=-1).clamp_min(0.0)
        inside = torch.maximum(radial, vertical).clamp_max(0.0)
        signed_distance = torch.linalg.norm(outside, dim=-1) + inside
        return signed_distance.min(dim=1).values

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

    def _spawn_shared_obstacles(self) -> None:
        print("[DEBUG][SingleWorld] spawning shared obstacles", flush=True)
        self.obstacle_positions_w[:] = self._sample_obstacles()
        obstacle_cfg = sim_utils.CuboidCfg(
            size=(
                self.cfg.obstacle_radius * 2,
                self.cfg.obstacle_radius * 2,
                self.cfg.obstacle_height,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2), metallic=0.0),
        )
        self._obstacle_paths.clear()
        for idx in range(self.cfg.num_obstacles):
            prim_path = f"/World/SharedObstacles/Obstacle_{idx:03d}"
            translation = tuple(self.obstacle_positions_w[idx].tolist())
            obstacle_cfg.func(prim_path, obstacle_cfg, translation=translation)
            self._obstacle_paths.append(prim_path)
        print("[DEBUG][SingleWorld] shared obstacles spawned", flush=True)

    def _spawn_boundary_walls(self) -> None:
        print("[DEBUG][SingleWorld] spawning boundary walls", flush=True)
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
        print("[DEBUG][SingleWorld] boundary walls spawned", flush=True)

    def _build_robot_assets(self) -> None:
        print("[DEBUG][SingleWorld] spawning robots", flush=True)
        robot_spawn_cfg = OMNINXT_CFG.replace(prim_path="/World/Robot_.*/OmniNxt")
        initial_spawn = self._sample_edge_positions_with_clearance(
            self.num_robots,
            self.cfg.target_spawn_range,
            self.cfg.spawn_min_height,
            self.cfg.spawn_max_height,
            min_separation=self.cfg.drone_spawn_min_separation,
        )
        for idx in range(self.num_robots):
            prim_path = f"/World/Robot_{idx:02d}/OmniNxt"
            robot_spawn_cfg.spawn.func(prim_path, robot_spawn_cfg.spawn, translation=tuple(initial_spawn[idx].tolist()))

        # Build the articulation view over the already-spawned robots.
        robot_view_cfg = robot_spawn_cfg.copy()
        robot_view_cfg.spawn = None
        print("[DEBUG][SingleWorld] creating articulation view", flush=True)
        self._robot = Articulation(robot_view_cfg)
        print("[DEBUG][SingleWorld] articulation view created", flush=True)
        print("[DEBUG][SingleWorld] robot articulation view created", flush=True)

    def _hide_robots_from_depth(self) -> None:
        """Mark robot visual geometry invisible to secondary rays such as depth images."""
        print("[DEBUG][SingleWorld] hiding robot geometry from depth rays", flush=True)
        stage = omni.usd.get_context().get_stage()
        for idx in range(self.num_robots):
            root_prim = stage.GetPrimAtPath(f"/World/Robot_{idx:02d}/OmniNxt")
            if not root_prim.IsValid():
                continue
            all_prims = list(root_prim.GetChildren())
            while all_prims:
                child_prim = all_prims.pop(0)
                if child_prim.IsA(UsdGeom.Gprim):
                    omni.kit.commands.execute(
                        "ChangePropertyCommand",
                        prop_path=Sdf.Path(
                            f"{child_prim.GetPrimPath().pathString}.primvars:invisibleToSecondaryRays"
                        ),
                        value=True,
                        prev=None,
                        type_to_create_if_not_exist=Sdf.ValueTypeNames.Bool,
                    )
                all_prims += child_prim.GetChildren()
        print("[DEBUG][SingleWorld] robot geometry hidden from depth rays", flush=True)

    def _disable_robot_collisions(self) -> None:
        """Disable collisions for all robot collider prims.

        The task uses logical obstacle collision/termination, so the robots do not
        need physical contacts. Disabling robot collisions prevents inter-robot
        interference while keeping them in the same world.
        """
        print("[DEBUG][SingleWorld] disabling robot collisions", flush=True)
        collision_cfg = sim_utils.CollisionPropertiesCfg(collision_enabled=False)
        for idx in range(self.num_robots):
            sim_schemas.modify_collision_properties(f"/World/Robot_{idx:02d}/OmniNxt", collision_cfg)
        print("[DEBUG][SingleWorld] robot collisions disabled", flush=True)

    def _build_depth_cameras(self) -> None:
        print("[DEBUG][SingleWorld] creating tiled camera", flush=True)
        camera_cfg = TiledCameraCfg(
            prim_path="/World/Robot_.*/OmniNxt/body/front_depth_camera",
            update_period=0.0,
            height=self.cfg.depth_camera_height,
            width=self.cfg.depth_camera_width,
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
                clipping_range=(self.cfg.depth_camera_near_clip, self.cfg.depth_camera_far_clip),
            ),
        )
        self._depth_camera = TiledCamera(camera_cfg)
        print("[DEBUG][SingleWorld] tiled camera created", flush=True)

    def build(self) -> None:
        if self._built:
            return

        print("[DEBUG][SingleWorld] build start", flush=True)
        sim_cfg = sim_utils.SimulationCfg(dt=self.cfg.physics_dt, device=str(self.device))
        self._sim = sim_utils.SimulationContext(sim_cfg)
        print("[DEBUG][SingleWorld] simulation context created", flush=True)
        self.sim.set_camera_view(
            eye=list(self.cfg.viewer_eye),
            target=list(self.cfg.viewer_lookat),
            camera_prim_path=self.cfg.viewer_cam_prim_path,
        )

        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_cfg.func("/World/Ground", ground_cfg)
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        print("[DEBUG][SingleWorld] ground and light spawned", flush=True)

        self._spawn_shared_obstacles()
        self._spawn_boundary_walls()
        self._build_robot_assets()
        self._disable_robot_collisions()
        self._hide_robots_from_depth()
        self._build_depth_cameras()

        print("[DEBUG][SingleWorld] calling sim.reset()", flush=True)
        self.sim.reset()
        print("[DEBUG][SingleWorld] sim.reset() done", flush=True)
        print("[DEBUG][SingleWorld] resolving body ids", flush=True)
        self._body_id = self.robot.find_bodies("body")[0]
        print("[DEBUG][SingleWorld] body ids resolved", flush=True)
        self._robot_mass = self.robot.root_physx_view.get_masses()[0].sum().item()
        self.robot.write_joint_state_to_sim(self.robot.data.default_joint_pos, self.robot.data.default_joint_vel)
        self.robot.update(self.cfg.physics_dt)
        print("[DEBUG][SingleWorld] robot joints initialized", flush=True)
        for _ in range(5):
            self.sim.step()
            self.robot.update(self.cfg.physics_dt)
        print("[DEBUG][SingleWorld] warmup steps done", flush=True)
        self.depth_camera.update(self.cfg.physics_dt)
        print("[DEBUG][SingleWorld] first depth update done", flush=True)

        print(
            f"[INFO][SingleWorld] Built scene with {self.num_robots} robots, "
            f"{self.cfg.num_obstacles} shared obstacles, depth shape "
            f"{self.cfg.num_stacked_depth_frames}x{self.cfg.depth_camera_height}x{self.cfg.depth_camera_width}",
            flush=True,
        )
        print("[INFO] Quadcopter Camera - Single World Depth Navigation", flush=True)
        print(f"[INFO] Obstacles: {self.cfg.num_obstacles}", flush=True)
        print("[INFO] Task: reach one random target, then reset", flush=True)
        print("[INFO] Policy action: 3-axis velocity command -> OmniNxt controller", flush=True)
        print(
            f"[INFO] Observation space: state={self.cfg.observation_space['state']}, "
            f"depth={self.cfg.observation_space['depth']}",
            flush=True,
        )
        print(
            f"[INFO] Map/obstacle region: 40x40 core area, spawn/target boundary: "
            f"{self.cfg.spawn_edge_distance * 2:.0f}x{self.cfg.spawn_edge_distance * 2:.0f}, "
            f"wall boundary: {self.cfg.wall_half_extent * 2:.0f}x{self.cfg.wall_half_extent * 2:.0f}",
            flush=True,
        )
        print(f"[INFO] OmniNxt mass from asset: {self._robot_mass:.4f} kg", flush=True)
        print(f"[INFO] Controller configured mass: {controller_config.CF_MASS:.4f} kg", flush=True)
        if abs(self._robot_mass - controller_config.CF_MASS) > 0.05:
            print(
                "[WARN] OmniNxt mass and controller CF_MASS differ significantly. "
                "Retune controller/config.py if flight is unstable.",
                flush=True,
            )
        self._built = True
        self.reset()

    def _get_depth_image(self) -> torch.Tensor:
        self.depth_camera.update(self.cfg.physics_dt)
        depth = self.depth_camera.data.output["depth"].squeeze(-1).to(self.device)
        depth = torch.nan_to_num(
            depth,
            nan=self.cfg.depth_camera_far_clip,
            posinf=self.cfg.depth_camera_far_clip,
            neginf=self.cfg.depth_camera_near_clip,
        )
        depth = depth.clamp(self.cfg.depth_camera_near_clip, self.cfg.depth_camera_far_clip)
        depth = 1.0 - (depth - self.cfg.depth_camera_near_clip) / (
            self.cfg.depth_camera_far_clip - self.cfg.depth_camera_near_clip
        )
        return depth

    def _update_depth_stack(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_robots, device=self.device)
        current_depth = self._get_depth_image()[env_ids]
        if self.depth_stack_needs_fill[env_ids].any():
            fill_ids = env_ids[self.depth_stack_needs_fill[env_ids]]
            self.depth_frame_stack[fill_ids] = current_depth[self.depth_stack_needs_fill[env_ids]].unsqueeze(1).repeat(
                1, self.cfg.num_stacked_depth_frames, 1, 1
            )
            self.depth_stack_needs_fill[fill_ids] = False
        normal_ids = env_ids[~self.depth_stack_needs_fill[env_ids]]
        if len(normal_ids) > 0:
            self.depth_frame_stack[normal_ids] = torch.roll(self.depth_frame_stack[normal_ids], shifts=-1, dims=1)
            self.depth_frame_stack[normal_ids, -1] = current_depth[~self.depth_stack_needs_fill[env_ids]]

    def _get_observations(self) -> dict[str, torch.Tensor]:
        root_pos_w = self.robot.data.root_pos_w
        root_quat_w = self.robot.data.root_quat_w
        root_lin_vel_w = self.robot.data.root_lin_vel_w
        root_ang_vel_w = self.robot.data.root_ang_vel_w

        root_lin_vel_b = quat_apply_inverse(root_quat_w, root_lin_vel_w)
        root_ang_vel_b = quat_apply_inverse(root_quat_w, root_ang_vel_w)
        gravity_w = torch.tensor([0.0, 0.0, -1.0], device=self.device).repeat(self.num_robots, 1)
        projected_gravity_b = quat_apply_inverse(root_quat_w, gravity_w)

        target_pos_b = quat_apply_inverse(root_quat_w, self.target_positions_w - root_pos_w)
        target_pos_b = target_pos_b * self.cfg.target_pos_obs_scale
        target_pos_b = target_pos_b.clamp(-self.cfg.target_pos_obs_clip, self.cfg.target_pos_obs_clip)
        cmd_obs = self.cmd_vel_b * self._cmd_obs_scale

        state = torch.cat(
            [
                root_lin_vel_b,
                root_ang_vel_b * self.cfg.ang_vel_obs_scale,
                projected_gravity_b,
                target_pos_b,
                cmd_obs,
                self.target_reached.float().unsqueeze(-1),
            ],
            dim=1,
        )
        return {"state": state, "depth": self.depth_frame_stack.clone()}

    def _compute_depth_sector_reward(self, depth_proximity: torch.Tensor) -> torch.Tensor:
        """Encourage commands toward the safest 3x3 depth-image sector.

        depth_proximity is in [0, 1], where larger means closer obstacle. We
        convert it to free-space score using (1 - proximity).
        """
        latest = 1.0 - depth_proximity
        row_bins = torch.tensor_split(latest, 3, dim=1)
        sector_scores = []
        for row in row_bins:
            col_bins = torch.tensor_split(row, 3, dim=2)
            for col in col_bins:
                sector_scores.append(col.mean(dim=(1, 2)))
        sector_scores = torch.stack(sector_scores, dim=1)
        best_sector = sector_scores.argmax(dim=1)

        # Forward-facing body-frame directions for the 3x3 image sectors.
        sector_dirs = torch.tensor(
            [
                [1.0, 1.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, -1.0, 1.0],
                [1.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, -1.0, 0.0],
                [1.0, 1.0, -1.0],
                [1.0, 0.0, -1.0],
                [1.0, -1.0, -1.0],
            ],
            device=self.device,
        )
        sector_dirs = sector_dirs / torch.linalg.norm(sector_dirs, dim=1, keepdim=True).clamp_min(1e-6)
        desired_dir = sector_dirs[best_sector]

        cmd_dir = self.cmd_vel_b / torch.linalg.norm(self.cmd_vel_b, dim=1, keepdim=True).clamp_min(1e-6)
        alignment = torch.sum(cmd_dir * desired_dir, dim=1).clamp(-1.0, 1.0)
        cmd_speed = torch.linalg.norm(self.cmd_vel_b, dim=1).clamp(max=1.0)
        best_score = sector_scores.gather(1, best_sector.unsqueeze(1)).squeeze(1)
        return alignment * cmd_speed * best_score * self.cfg.depth_sector_reward_scale

    def reset(self, env_ids: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if not self._built:
            self.build()
        if env_ids is None:
            env_ids = torch.arange(self.num_robots, device=self.device)

        start_pos = self._sample_edge_positions_with_clearance(
            len(env_ids),
            self.cfg.target_spawn_range,
            self.cfg.spawn_min_height,
            self.cfg.spawn_max_height,
            min_separation=self.cfg.drone_spawn_min_separation,
        )
        target_pos = self._sample_edge_positions_with_clearance(
            len(env_ids),
            self.cfg.target_spawn_range,
            self.cfg.target_min_height,
            self.cfg.target_max_height,
            avoid_obstacles=True,
            obstacle_clearance=self.cfg.target_obstacle_clearance,
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

        self.target_positions_w[env_ids] = target_pos
        self.target_reached[env_ids] = False
        self.done_buf[env_ids] = False
        self.rew_buf[env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0
        self.cmd_vel_b[env_ids] = 0.0
        self.actions[env_ids] = 0.0
        self.depth_stack_needs_fill[env_ids] = True
        self.height_range[env_ids, 0] = torch.minimum(start_pos[:, 2], target_pos[:, 2])
        self.height_range[env_ids, 1] = torch.maximum(start_pos[:, 2], target_pos[:, 2])
        self.prev_lin_vel_w[env_ids] = 0.0

        controller_state = {
            "position": root_state[:, :3].clone(),
            "attitude": _quat_to_euler_deg(root_state[:, 3:7]),
        }
        self._controller.reset(controller_state, env_ids=env_ids)

        self.sim.step()
        self.robot.update(self.cfg.physics_dt)
        dist_to_target = torch.linalg.norm(self.target_positions_w[env_ids] - self.robot.data.root_pos_w[env_ids], dim=1)
        self.prev_dist_to_target[env_ids] = dist_to_target
        for key in self._episode_sums:
            self._episode_sums[key][env_ids] = 0.0
        self._update_depth_stack(env_ids)
        return self._get_observations()

    def step(self, actions: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, dict]:
        if not self._built:
            raise RuntimeError("Call build() before step().")

        actions = actions.to(self.device).clamp(-1.0, 1.0)
        self.actions[:] = actions
        self.cmd_vel_b[:, 0] = actions[:, 0] * self.cfg.cmd_body_vel_xy_max
        self.cmd_vel_b[:, 1] = actions[:, 1] * self.cfg.cmd_body_vel_xy_max
        self.cmd_vel_b[:, 2] = actions[:, 2] * self.cfg.cmd_vel_z_max

        root_pos_w = self.robot.data.root_pos_w
        root_quat_w = self.robot.data.root_quat_w
        root_lin_vel_w = self.robot.data.root_lin_vel_w
        root_ang_vel_b = quat_apply_inverse(root_quat_w, self.robot.data.root_ang_vel_w)
        attitude_deg = _quat_to_euler_deg(root_quat_w)

        self._controller.set_velocity_setpoint(
            self.cmd_vel_b[:, 0],
            self.cmd_vel_b[:, 1],
            self.cmd_vel_b[:, 2],
            yaw_rate=torch.zeros(self.num_robots, device=self.device),
            velocity_body=True,
        )
        force, torque = self._controller.compute(
            {
                "position": root_pos_w,
                "velocity": root_lin_vel_w,
                "attitude": attitude_deg,
                "angular_velocity": torch.rad2deg(root_ang_vel_b),
            }
        )
        self._thrust[:, 0, :] = force
        self._moment[:, 0, :] = torque

        for _ in range(self.cfg.control_decimation):
            self.robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)
            self.robot.write_data_to_sim()
            self.sim.step()
            self.robot.update(self.cfg.physics_dt)

        self.common_step_counter += 1
        self.episode_length_buf += 1
        self._update_depth_stack()

        target_vec_w = self.target_positions_w - self.robot.data.root_pos_w
        dist_to_target = torch.linalg.norm(target_vec_w, dim=1)
        progress = (self.prev_dist_to_target - dist_to_target).clamp(-1.0, 1.0)
        self.prev_dist_to_target = dist_to_target
        newly_reached = (dist_to_target < self.cfg.target_reach_threshold) & (~self.target_reached)
        self.target_reached |= newly_reached

        closest_obstacle_distance = self._compute_closest_obstacle_signed_distance(self.robot.data.root_pos_w)
        closest_wall_distance = self._compute_wall_signed_distance(self.robot.data.root_pos_w)
        current_depth = self.depth_frame_stack[:, -1]
        depth_metric = self.cfg.depth_camera_near_clip + (1.0 - current_depth) * (
            self.cfg.depth_camera_far_clip - self.cfg.depth_camera_near_clip
        )
        reward_safety_static = torch.log(
            depth_metric.clamp(min=1e-6, max=self.cfg.depth_camera_far_clip)
        ).mean(dim=(1, 2))
        target_direction = target_vec_w / dist_to_target.unsqueeze(1).clamp_min(1e-6)
        reward_vel = torch.sum(self.robot.data.root_lin_vel_w * target_direction, dim=1)
        penalty_smooth = torch.linalg.norm(self.robot.data.root_lin_vel_w - self.prev_lin_vel_w, dim=1)
        z_pos = self.robot.data.root_pos_w[:, 2]
        buffer = 0.2
        penalty_height = torch.zeros_like(z_pos)
        mask_high = z_pos > (self.height_range[:, 1] + buffer)
        penalty_height[mask_high] = (z_pos[mask_high] - self.height_range[mask_high, 1] - buffer) ** 2
        mask_low = z_pos < (self.height_range[:, 0] - buffer)
        penalty_height[mask_low] = (self.height_range[mask_low, 0] - buffer - z_pos[mask_low]) ** 2
        target_bonus = newly_reached.float() * self.cfg.target_reached_bonus
        depth_sector_reward = self._compute_depth_sector_reward(current_depth) * self.step_dt
        rewards = {
            "progress": progress * self.cfg.progress_reward_scale * self.step_dt,
            "target_velocity": reward_vel.clamp(min=0.0) * self.cfg.target_velocity_reward_scale * self.step_dt,
            "target_bonus": target_bonus,
            "safety_static": reward_safety_static * self.cfg.safety_static_reward_scale,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        for key, value in rewards.items():
            self._episode_sums[key] += value
        self.rew_buf[:] = reward
        self.prev_lin_vel_w[:] = self.robot.data.root_lin_vel_w

        timeout = self.episode_length_buf >= self.cfg.max_episode_steps
        obstacle_collision = closest_obstacle_distance < self.cfg.obstacle_collision_margin
        wall_collision = closest_wall_distance < self.cfg.obstacle_collision_margin
        too_low = self.robot.data.root_pos_w[:, 2] < 0.1
        too_high = self.robot.data.root_pos_w[:, 2] > 2.5
        died = too_low | too_high | obstacle_collision | wall_collision
        terminated = self.target_reached | died
        self.done_buf[:] = terminated | timeout

        extras = {
            "target_reached": self.target_reached.clone(),
            "timeout": timeout.clone(),
            "terminated": terminated.clone(),
            "obstacle_collision": obstacle_collision.clone(),
            "wall_collision": wall_collision.clone(),
            "closest_obstacle_distance": closest_obstacle_distance.clone(),
            "closest_wall_distance": closest_wall_distance.clone(),
            "distance_to_target": dist_to_target.clone(),
        }
        if self.common_step_counter % 500 == 0:
            success_rate = self.target_reached.float().mean().item()
            print(
                f"[DEBUG] Step {self.common_step_counter}: died={died.sum().item()}, "
                f"success={self.target_reached.sum().item()}, collision={(obstacle_collision | wall_collision).sum().item()}, "
                f"success_rate={success_rate:.3f}",
                flush=True,
            )
        done_ids = torch.nonzero(self.done_buf, as_tuple=False).squeeze(-1)
        if done_ids.numel() > 0:
            log = {}
            for key in self._episode_sums.keys():
                episodic_sum_avg = torch.mean(self._episode_sums[key][done_ids])
                log[f"Episode_Reward/{key}"] = episodic_sum_avg / self.max_episode_length_s
            log["Episode_Termination/died"] = torch.count_nonzero(died[done_ids]).item()
            log["Episode_Termination/time_out"] = torch.count_nonzero(timeout[done_ids]).item()
            log["Metrics/success_rate"] = self.target_reached[done_ids].float().mean().item()
            extras["log"] = log
        rew = self.rew_buf.clone()
        done = self.done_buf.clone()
        if done_ids.numel() > 0:
            self.reset(done_ids)
        obs = self._get_observations()
        return obs, rew, done, extras

    def close(self) -> None:
        if self._sim is None:
            return
        self._sim._timeline.stop()
        self._sim.clear_all_callbacks()
        self._sim.clear_instance()
        self._sim = None
        self._robot = None
        self._depth_camera = None
        self._built = False
        print("[INFO][SingleWorld] Simulation closed.", flush=True)


class QuadcopterSingleWorldDepthGymEnv(gym.Env):
    """Gym wrapper exposing the single-world environment as PPO-compatible batch env."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, cfg: QuadcopterSingleWorldDepthEnvCfg | None = None, render_mode: str | None = None, **kwargs):
        del kwargs
        self.cfg = cfg if cfg is not None else QuadcopterSingleWorldDepthEnvCfg()
        self.render_mode = render_mode
        # Keep IsaacLab train.py compatibility: treat scene.num_envs as robot count.
        if hasattr(self.cfg, "scene"):
            self.cfg.num_robots = self.cfg.scene.num_envs
        if hasattr(self.cfg, "sim"):
            self.cfg.device = self.cfg.sim.device

        self._env = QuadcopterSingleWorldDepthEnv(self.cfg)
        self._record_robot_index = 0
        self._render_product = None
        self._rgb_annotator = None
        self.num_envs = self.cfg.num_robots
        self.device = self._env.device
        self.max_episode_length = self.cfg.max_episode_steps
        self.num_states = 0

        self.single_action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.cfg.action_space,), dtype=float)
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)
        self.single_observation_space = gym.spaces.Dict(
            {
                "policy": gym.spaces.Dict(
                    {
                        "state": gym.spaces.Box(
                            low=-float("inf"), high=float("inf"), shape=(self.cfg.observation_space["state"],), dtype=float
                        ),
                        "depth": gym.spaces.Box(
                            low=0.0,
                            high=1.0,
                            shape=self.cfg.observation_space["depth"],
                            dtype=float,
                        ),
                    }
                )
            }
        )
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space, self.num_envs)
        self.episode_length_buf = self._env.episode_length_buf

    def seed(self, seed: int = -1) -> int:
        if seed >= 0:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        return seed

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        del options
        if seed is not None:
            self.seed(seed)
        obs = self._env.reset()
        self.episode_length_buf = self._env.episode_length_buf
        return {"policy": obs}, {}

    def _get_observations(self):
        return {"policy": self._env._get_observations()}

    def step(self, actions):
        obs, rew, done, extras = self._env.step(actions)
        self.episode_length_buf = self._env.episode_length_buf
        terminated = extras.get("terminated", done).clone()
        truncated = extras.get("timeout", torch.zeros_like(terminated)).clone()
        return {"policy": obs}, rew, terminated, truncated, extras

    def set_record_robot_index(self, robot_index: int):
        self._record_robot_index = int(np.clip(robot_index, 0, self.num_envs - 1))

    def render_depth(self, robot_index: int | None = None):
        if self.render_mode != "rgb_array":
            return None
        if not self._env._built:
            self._env.build()
        if robot_index is None:
            robot_index = self._record_robot_index
        depth_frame = self._env.depth_frame_stack[robot_index, -1].detach().cpu().numpy()
        depth_rgb = np.clip(depth_frame * 255.0, 0.0, 255.0).astype(np.uint8)
        return np.repeat(depth_rgb[:, :, None], 3, axis=2)

    def render_topdown(self, robot_index: int | None = None):
        if self.render_mode != "rgb_array":
            return None
        if not self._env._built:
            self._env.build()
        if robot_index is None:
            robot_index = self._record_robot_index

        image_size = 512
        pad = 16
        extent = self.cfg.spawn_edge_distance + 2.0
        scale = (image_size - 2 * pad) / (2 * extent)

        def to_px(x: float, y: float) -> tuple[int, int]:
            px = int(round((x + extent) * scale + pad))
            py = int(round((extent - y) * scale + pad))
            return px, py

        img = Image.new("RGB", (image_size, image_size), (18, 22, 28))
        draw = ImageDraw.Draw(img)

        wall_min = to_px(-self.cfg.wall_half_extent, -self.cfg.wall_half_extent)
        wall_max = to_px(self.cfg.wall_half_extent, self.cfg.wall_half_extent)
        draw.rectangle([wall_min[0], wall_max[1], wall_max[0], wall_min[1]], outline=(180, 180, 180), width=3)

        core_min = to_px(-self.cfg.map_half_extent, -self.cfg.map_half_extent)
        core_max = to_px(self.cfg.map_half_extent, self.cfg.map_half_extent)
        draw.rectangle([core_min[0], core_max[1], core_max[0], core_min[1]], outline=(90, 90, 90), width=2)

        obstacle_r = max(2, int(round(self.cfg.obstacle_radius * scale)))
        for obstacle_pos in self._env.obstacle_positions_w.detach().cpu().numpy():
            ox, oy = to_px(float(obstacle_pos[0]), float(obstacle_pos[1]))
            draw.ellipse((ox - obstacle_r, oy - obstacle_r, ox + obstacle_r, oy + obstacle_r), fill=(196, 62, 62))

        robot_r = 4
        target_r = 4
        robot_positions = self._env.robot.data.root_pos_w.detach().cpu().numpy()
        target_positions = self._env.target_positions_w.detach().cpu().numpy()
        for idx in range(self.num_envs):
            tx, ty = to_px(float(target_positions[idx, 0]), float(target_positions[idx, 1]))
            draw.rectangle((tx - target_r, ty - target_r, tx + target_r, ty + target_r), fill=(60, 220, 90))

            rx, ry = to_px(float(robot_positions[idx, 0]), float(robot_positions[idx, 1]))
            robot_color = (255, 210, 60) if idx == robot_index else (60, 210, 230)
            draw.ellipse((rx - robot_r, ry - robot_r, rx + robot_r, ry + robot_r), fill=robot_color)

        return np.asarray(img, dtype=np.uint8)

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if not self._env._built:
            self._env.build()
        self._env.sim.render()
        if self._env.sim.render_mode.value < self._env.sim.RenderMode.PARTIAL_RENDERING.value:
            raise RuntimeError(
                "Cannot render 'rgb_array' when the simulation render mode does not support rendering."
            )
        if self._rgb_annotator is None:
            self._render_product = rep.create.render_product(
                self.cfg.viewer_cam_prim_path, self.cfg.viewer_resolution
            )
            self._rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
            self._rgb_annotator.attach([self._render_product])
        rgb_data = self._rgb_annotator.get_data()
        rgb_data = np.frombuffer(rgb_data, dtype=np.uint8).reshape(*rgb_data.shape)
        if rgb_data.size == 0:
            return np.zeros((self.cfg.viewer_resolution[1], self.cfg.viewer_resolution[0], 3), dtype=np.uint8)
        return rgb_data[:, :, :3]

    def close(self):
        if self._rgb_annotator is not None and self._render_product is not None:
            try:
                self._rgb_annotator.detach([self._render_product])
            except Exception:
                pass
        return self._env.close()
