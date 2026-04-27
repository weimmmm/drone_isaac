# Shared-map quadcopter environment with depth camera perception.
# All drones share the same obstacle layout in a large 40x40 map.
# Based on the proven DirectRLEnv pattern from quadcopter_obstacles_env.py.

from __future__ import annotations

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sensors.camera import TiledCamera, TiledCameraCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

from assets.omninxt.omninxt import OMNINXT_CFG
from controller import CrazyflieController, config as controller_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------------

@configclass
class QuadcopterSharedMapEnvCfg(DirectRLEnvCfg):
    """Shared-map quadcopter environment configuration."""

    # -- Episode --
    episode_length_s = 90.0
    decimation = 2

    # -- Obstacles --
    num_obstacles = 20

    # -- Target --
    target_reach_threshold = 0.5

    # -- Depth camera base dims (must precede observation_space) --
    depth_camera_width = 32
    depth_camera_height = 24
    num_stacked_depth_frames = 2

    # -- Spaces --
    # state: lin_vel_b(3) + ang_vel_b(3) + gravity_b(3) + target_pos_b(3) + cmd(3) + reached(1) = 16
    observation_space = {
        "state": 16,
        "depth": [num_stacked_depth_frames, depth_camera_height, depth_camera_width],
    }
    action_space = 3
    state_space = 0
    debug_vis = False
    is_finite_horizon = False

    # -- Simulation --
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

    # -- Terrain --
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

    # -- Scene (shared map: env_spacing=0, replicate_physics keeps drones independent) --
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=8,
        env_spacing=0.0,
        replicate_physics=True,
    )

    # -- Robot --
    robot: ArticulationCfg = OMNINXT_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    cmd_body_vel_xy_max = 0.6
    cmd_vel_z_max = 0.6
    ang_vel_obs_scale = 0.25
    target_pos_obs_scale = 0.1
    target_pos_obs_clip = 2.0

    # -- Depth camera --
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

    # -- Obstacle geometry --
    obstacle_height = 1.5
    obstacle_radius = 0.15
    obstacle_spawn_range = 15.0
    obstacle_safe_zone = 1.0
    obstacle_min_separation = 0.6

    # -- Spawn / target --
    spawn_edge_distance = 20.0
    target_spawn_range = 20.0
    target_min_height = 0.5
    target_max_height = 2.5
    target_obstacle_clearance = 2.0
    spawn_min_height = 0.5
    spawn_max_height = 2.5

    # -- Rewards --
    vel_tracking_reward_scale = 0.5
    vel_tracking_exp_scale = 4.0
    ang_vel_reward_scale = -0.01
    distance_to_target_reward_scale = 8.0
    target_velocity_reward_scale = 4.0
    target_reached_bonus = 30.0
    obstacle_proximity_reward_scale = -3.0
    progress_reward_scale = 15.0


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class QuadcopterSharedMapEnv(DirectRLEnv):
    """Shared-map quadcopter environment with depth camera.

    All drones train in the same 40x40 map with shared obstacles.
    ``env_spacing=0`` ensures all environment origins overlap, so a single
    obstacle layout is visually shared across all parallel environments.
    ``replicate_physics=True`` keeps each drone's physics independent.
    """

    cfg: QuadcopterSharedMapEnvCfg

    def __init__(self, cfg: QuadcopterSharedMapEnvCfg, render_mode: str | None = None, **kwargs):
        print("[DEBUG][SharedMapEnv] __init__ start", flush=True)
        super().__init__(cfg, render_mode, **kwargs)
        print("[DEBUG][SharedMapEnv] super().__init__ done", flush=True)

        # -- Actions / forces --
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._cmd_vel_b = torch.zeros(self.num_envs, 3, device=self.device)

        # -- Targets --
        self._target_positions_local = torch.zeros(self.num_envs, 3, device=self.device)
        self._target_reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._prev_dist_to_target = torch.zeros(self.num_envs, device=self.device)

        # -- Obstacles (shared layout, stored per-env for API compatibility) --
        self._obstacle_positions_local = torch.zeros(
            self.num_envs, self.cfg.num_obstacles, 3, device=self.device,
        )
        self._shared_obstacle_positions_local = torch.zeros(
            self.cfg.num_obstacles, 3, device=self.device,
        )
        print("[DEBUG][SharedMapEnv] obstacle tensors allocated", flush=True)

        # -- Initial reset --
        all_env_ids = torch.arange(self.num_envs, device=self.device)
        print("[DEBUG][SharedMapEnv] randomizing obstacles", flush=True)
        self._randomize_obstacles(all_env_ids)
        print("[DEBUG][SharedMapEnv] obstacles ready", flush=True)
        print("[DEBUG][SharedMapEnv] randomizing targets", flush=True)
        self._randomize_targets(all_env_ids)
        print("[DEBUG][SharedMapEnv] targets ready", flush=True)

        # -- Episode logging --
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

        # -- Robot helpers --
        self._body_id = self._robot.find_bodies("body")[0]
        print("[DEBUG][SharedMapEnv] robot body resolved", flush=True)
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum().item()
        print("[DEBUG][SharedMapEnv] robot mass queried", flush=True)
        self._controller = CrazyflieController(
            num_envs=self.num_envs,
            device=self.device,
            attitude_dt=self.step_dt,
            position_dt=self.step_dt,
        )
        print("[DEBUG][SharedMapEnv] controller created", flush=True)

        # -- Depth camera --
        self._depth_camera = self.scene.sensors["depth_camera"]
        print("[DEBUG][SharedMapEnv] depth camera handle acquired", flush=True)
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

        # -- Collision threshold --
        self._collision_threshold = 0.05

        print("[INFO] Shared-map quadcopter env with depth camera (DirectRLEnv)")
        print(f"[INFO] Map: 40x40, env_spacing=0, obstacles={self.cfg.num_obstacles}")
        print(f"[INFO] OmniNxt mass from asset: {self._robot_mass:.4f} kg")
        print(f"[INFO] Controller configured mass: {controller_config.CF_MASS:.4f} kg")
        if abs(self._robot_mass - controller_config.CF_MASS) > 0.05:
            print("[WARN] Mass mismatch - consider retuning controller/config.py")

    # ------------------------------------------------------------------
    # Scene setup
    # ------------------------------------------------------------------

    def _setup_scene(self):
        """Build the USD scene: robot, camera, obstacles, terrain, lights."""
        print("[DEBUG][SharedMapEnv] _setup_scene start", flush=True)

        # Robot
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        print("[DEBUG][SharedMapEnv] robot created", flush=True)

        # Depth camera
        self._depth_camera_sensor = TiledCamera(self.cfg.depth_camera)
        self.scene.sensors["depth_camera"] = self._depth_camera_sensor
        print("[DEBUG][SharedMapEnv] tiled camera created", flush=True)

        # Obstacles - per-env prims (visually shared because env_spacing=0)
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
        for idx in range(self.cfg.num_obstacles):
            obstacle_cfg = RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/Obstacle_{idx:03d}",
                spawn=obstacle_spawn_cfg,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -10.0)),
            )
            obstacle_obj = RigidObject(cfg=obstacle_cfg)
            self._obstacles.append(obstacle_obj)
            self.scene.rigid_objects[f"obstacle_{idx:03d}"] = obstacle_obj
        print(f"[DEBUG][SharedMapEnv] {self.cfg.num_obstacles} obstacle objects created", flush=True)

        # Terrain
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        print("[DEBUG][SharedMapEnv] terrain created", flush=True)

        # Clone environments
        self.scene.clone_environments(copy_from_source=False)
        print("[DEBUG][SharedMapEnv] clone_environments done", flush=True)

        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # Lighting
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        print("[DEBUG][SharedMapEnv] _setup_scene complete", flush=True)

    # ------------------------------------------------------------------
    # Obstacle helpers
    # ------------------------------------------------------------------

    def _sync_render_obstacles(self, env_ids: torch.Tensor):
        """Write obstacle positions to the USD prims for rendering."""
        env_origins = self._terrain.env_origins[env_ids, :2]
        obstacle_pos_local = self._obstacle_positions_local[env_ids]
        obstacle_pos_world = obstacle_pos_local.clone()
        obstacle_pos_world[:, :, :2] += env_origins.unsqueeze(1)
        for idx, obstacle_obj in enumerate(self._obstacles):
            root_state = torch.zeros((len(env_ids), 13), device=self.device)
            root_state[:, :3] = obstacle_pos_world[:, idx]
            root_state[:, 3] = 1.0  # w of quaternion
            obstacle_obj.write_root_state_to_sim(root_state, env_ids)

    def _randomize_obstacles(self, env_ids: torch.Tensor):
        """Generate a shared obstacle layout (identical for all envs)."""
        regenerate = not torch.any(self._shared_obstacle_positions_local)
        if len(env_ids) == self.num_envs:
            regenerate = True

        if regenerate:
            placed_xy = torch.zeros(self.cfg.num_obstacles, 2, device=self.device)
            min_sep = self.cfg.obstacle_min_separation

            for obs_idx in range(self.cfg.num_obstacles):
                chosen_xy = None
                chosen_score = None
                for _ in range(64):
                    cand = torch.empty(2, device=self.device).uniform_(
                        -self.cfg.obstacle_spawn_range, self.cfg.obstacle_spawn_range,
                    )
                    cand_r = torch.linalg.norm(cand, dim=0)
                    valid = cand_r >= (self.cfg.obstacle_safe_zone + self.cfg.obstacle_radius + 0.2)
                    if obs_idx == 0:
                        md = torch.tensor(float("inf"), device=self.device)
                    else:
                        dists = torch.linalg.norm(cand.unsqueeze(0) - placed_xy[:obs_idx], dim=1)
                        md = dists.min()
                        valid = bool(valid and (md >= min_sep))
                    if chosen_xy is None:
                        chosen_xy = cand
                        chosen_score = md if valid else torch.tensor(-1.0, device=self.device)
                    else:
                        cs = md if valid else torch.tensor(-1.0, device=self.device)
                        if cs > chosen_score:
                            chosen_xy = cand
                            chosen_score = cs
                    if valid:
                        chosen_xy = cand
                        break
                placed_xy[obs_idx] = chosen_xy

            self._shared_obstacle_positions_local[:, :2] = placed_xy
            self._shared_obstacle_positions_local[:, 2] = self.cfg.obstacle_height / 2

        self._obstacle_positions_local[env_ids] = self._shared_obstacle_positions_local.unsqueeze(0).expand(
            len(env_ids), -1, -1,
        )
        if hasattr(self, "_obstacles"):
            self._sync_render_obstacles(env_ids)

    # ------------------------------------------------------------------
    # Target helpers
    # ------------------------------------------------------------------

    def _sample_edge_positions(
        self, num_samples: int, lateral_range: float, min_height: float, max_height: float,
    ) -> torch.Tensor:
        """Sample positions along the four edges of the 40x40 map."""
        side_idx = torch.randint(0, 4, (num_samples,), device=self.device)
        side_sign = torch.where(
            side_idx % 2 == 0,
            torch.ones(num_samples, device=self.device),
            -torch.ones(num_samples, device=self.device),
        )
        lateral = torch.empty(num_samples, device=self.device).uniform_(-lateral_range, lateral_range)
        heights = torch.empty(num_samples, device=self.device).uniform_(min_height, max_height)

        positions = torch.zeros(num_samples, 3, device=self.device)
        x_mask = side_idx < 2
        y_mask = ~x_mask
        positions[x_mask, 0] = lateral[x_mask]
        positions[x_mask, 1] = side_sign[x_mask] * self.cfg.spawn_edge_distance
        positions[y_mask, 0] = side_sign[y_mask] * self.cfg.spawn_edge_distance
        positions[y_mask, 1] = lateral[y_mask]
        positions[:, 2] = heights
        return positions

    def _randomize_targets(self, env_ids: torch.Tensor):
        """Sample target positions that are far enough from obstacles."""
        n = len(env_ids)
        clearance = self.cfg.target_obstacle_clearance + self.cfg.obstacle_radius
        best_pos = None
        best_clr = None

        for _ in range(64):
            cand = self._sample_edge_positions(
                n, self.cfg.target_spawn_range, self.cfg.target_min_height, self.cfg.target_max_height,
            )
            dists = torch.linalg.norm(
                cand[:, :2].unsqueeze(1) - self._obstacle_positions_local[env_ids, :, :2], dim=2,
            )
            min_clr = dists.min(dim=1).values
            ok = min_clr > clearance

            if best_pos is None:
                best_pos = cand
                best_clr = min_clr
            else:
                better = min_clr > best_clr
                best_pos[better] = cand[better]
                best_clr[better] = min_clr[better]

            if ok.all():
                best_pos = cand
                break

        self._target_positions_local[env_ids] = best_pos
        self._target_reached[env_ids] = False
        self._prev_dist_to_target[env_ids] = 0.0

    def _get_target_world(self) -> torch.Tensor:
        t = self._target_positions_local.clone()
        t[:, :2] += self._terrain.env_origins[:, :2]
        return t

    def _update_prev_target_distance(self, env_ids: torch.Tensor):
        tw = self._target_positions_local[env_ids].clone()
        tw[:, :2] += self._terrain.env_origins[env_ids, :2]
        self._prev_dist_to_target[env_ids] = torch.linalg.norm(
            tw - self._robot.data.root_pos_w[env_ids], dim=1,
        )

    # ------------------------------------------------------------------
    # Obstacle distance
    # ------------------------------------------------------------------

    def _compute_min_obstacle_distance(self) -> torch.Tensor:
        """Signed distance to nearest cylinder obstacle surface."""
        drone_local = self._robot.data.root_pos_w - self._terrain.env_origins
        obs_local = self._obstacle_positions_local
        radial = torch.linalg.norm(
            drone_local[:, None, :2] - obs_local[:, :, :2], dim=2,
        ) - self.cfg.obstacle_radius
        vertical = torch.abs(
            drone_local[:, None, 2] - obs_local[:, :, 2],
        ) - (self.cfg.obstacle_height / 2.0)
        out_r = radial.clamp_min(0.0)
        out_v = vertical.clamp_min(0.0)
        outside = torch.sqrt(out_r.square() + out_v.square())
        inside = torch.minimum(torch.maximum(radial, vertical), torch.zeros_like(radial))
        return (outside + inside).min(dim=1).values

    # ------------------------------------------------------------------
    # Depth camera
    # ------------------------------------------------------------------

    def _get_depth_image(self) -> torch.Tensor:
        if not hasattr(self, "_debug_depth_log_once"):
            self._debug_depth_log_once = False
        if not self._debug_depth_log_once:
            print("[DEBUG][SharedMapEnv] reading depth image", flush=True)
        depth = self._depth_camera.data.output["depth"].squeeze(-1)
        depth = torch.nan_to_num(
            depth,
            nan=self.cfg.depth_camera_far_clip,
            posinf=self.cfg.depth_camera_far_clip,
            neginf=self.cfg.depth_camera_near_clip,
        )
        depth = depth.clamp(self.cfg.depth_camera_near_clip, self.cfg.depth_camera_far_clip)
        proximity = 1.0 - (
            (depth - self.cfg.depth_camera_near_clip)
            / (self.cfg.depth_camera_far_clip - self.cfg.depth_camera_near_clip)
        )
        if not self._debug_depth_log_once:
            print(f"[DEBUG][SharedMapEnv] depth image ready: {tuple(proximity.shape)}", flush=True)
            self._debug_depth_log_once = True
        return proximity.clamp(0.0, 1.0).unsqueeze(1)

    def _get_stacked_depth_images(self) -> torch.Tensor:
        current = self._get_depth_image()
        if self.common_step_counter != self._last_depth_stack_update_step:
            init_mask = ~self._depth_stack_needs_fill
            if init_mask.any():
                self._depth_frame_stack[init_mask] = torch.roll(
                    self._depth_frame_stack[init_mask], shifts=-1, dims=1,
                )
                self._depth_frame_stack[init_mask, -1] = current[init_mask, 0]
            self._last_depth_stack_update_step = self.common_step_counter
        if self._depth_stack_needs_fill.any():
            fill = self._depth_stack_needs_fill
            self._depth_frame_stack[fill] = current[fill].expand(-1, self.cfg.num_stacked_depth_frames, -1, -1)
            self._depth_stack_needs_fill[fill] = False
        return self._depth_frame_stack

    # ------------------------------------------------------------------
    # DirectRLEnv interface
    # ------------------------------------------------------------------

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

    def _get_observations(self) -> dict:
        target_world = self._get_target_world()
        target_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_pos_w, self._robot.data.root_quat_w, target_world,
        )
        ang_vel_obs = (self._robot.data.root_ang_vel_b * self.cfg.ang_vel_obs_scale).clamp(-2.0, 2.0)
        target_pos_obs = (target_pos_b * self.cfg.target_pos_obs_scale).clamp(
            -self.cfg.target_pos_obs_clip, self.cfg.target_pos_obs_clip,
        )
        cmd_obs = (self._cmd_vel_b * self._cmd_obs_scale).clamp(-1.0, 1.0)

        state_obs = torch.cat(
            [
                self._robot.data.root_lin_vel_b,        # 3
                ang_vel_obs,                             # 3
                self._robot.data.projected_gravity_b,    # 3
                target_pos_obs,                          # 3
                cmd_obs,                                 # 3
                self._target_reached.float().unsqueeze(1),  # 1
            ],
            dim=-1,
        )
        return {
            "state": state_obs,
            "depth": self._get_stacked_depth_images(),
        }

    def _get_rewards(self) -> torch.Tensor:
        # -- Velocity tracking --
        vel_cmd_frame = torch.stack(
            [
                self._robot.data.root_lin_vel_b[:, 0],
                self._robot.data.root_lin_vel_b[:, 1],
                self._robot.data.root_lin_vel_w[:, 2],
            ],
            dim=1,
        )
        vel_sq_err = torch.sum(torch.square(self._cmd_vel_b - vel_cmd_frame), dim=1)
        vel_err = torch.linalg.norm(self._cmd_vel_b - vel_cmd_frame, dim=1)
        vel_tracking = torch.exp(-self.cfg.vel_tracking_exp_scale * vel_sq_err)
        ang_vel = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)

        # -- Distance to target --
        target_w = self._get_target_world()
        dist = torch.linalg.norm(target_w - self._robot.data.root_pos_w, dim=1)
        distance_reward = 1 - torch.tanh(dist / 2.0)

        tdir = target_w - self._robot.data.root_pos_w
        tdir = tdir / torch.linalg.norm(tdir, dim=1, keepdim=True).clamp_min(1e-6)
        target_vel = torch.sum(self._robot.data.root_lin_vel_w * tdir, dim=1).clamp(-1.0, 1.5)

        # -- Progress --
        progress = (self._prev_dist_to_target - dist).clamp(-1.0, 1.0)
        self._prev_dist_to_target = dist.clone()

        # -- Target bonus --
        reached_now = (dist < self.cfg.target_reach_threshold) & (~self._target_reached)
        target_bonus = torch.zeros(self.num_envs, device=self.device)
        if reached_now.any():
            target_bonus[reached_now] = self.cfg.target_reached_bonus
            self._target_reached[reached_now] = True

        # -- Obstacle proximity --
        min_obs_dist = self._compute_min_obstacle_distance()
        obstacle_prox = torch.where(
            min_obs_dist < 1.0,
            torch.exp(-min_obs_dist * 3.0),
            torch.zeros_like(min_obs_dist),
        )

        # -- Sum --
        rewards = {
            "vel_tracking": vel_tracking * self.cfg.vel_tracking_reward_scale * self.step_dt,
            "ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
            "distance_to_target": distance_reward * self.cfg.distance_to_target_reward_scale * self.step_dt,
            "target_velocity": target_vel.clamp(min=0.0) * self.cfg.target_velocity_reward_scale * self.step_dt,
            "target_bonus": target_bonus,
            "obstacle_proximity": obstacle_prox * self.cfg.obstacle_proximity_reward_scale * self.step_dt,
            "progress": progress * self.cfg.progress_reward_scale * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        for k, v in rewards.items():
            self._episode_sums[k] += v
        self._episode_sums["vel_tracking_error"] += vel_err * self.step_dt
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        too_low = self._robot.data.root_pos_w[:, 2] < 0.1
        too_high = self._robot.data.root_pos_w[:, 2] > 2.5
        collision = self._compute_min_obstacle_distance() < self._collision_threshold
        died = too_low | too_high | collision
        terminated = died | self._target_reached
        return terminated, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        print(
            "[DEBUG][SharedMapEnv] _reset_idx "
            + ("all" if env_ids is None else f"count={len(env_ids)}"),
            flush=True,
        )
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # -- Logging --
        success_rate = self._target_reached[env_ids].float().mean().item()
        extras = {}
        for key in self._episode_sums:
            avg = torch.mean(self._episode_sums[key][env_ids])
            if key == "vel_tracking_error":
                extras["Metrics/avg_vel_tracking_error"] = avg / self.max_episode_length_s
            else:
                extras["Episode_Reward/" + key] = avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = {}
        self.extras["log"].update(extras)
        term_extras = {
            "Episode_Termination/died": torch.count_nonzero(self.reset_terminated[env_ids]).item(),
            "Episode_Termination/time_out": torch.count_nonzero(self.reset_time_outs[env_ids]).item(),
            "Metrics/success_rate": success_rate,
        }
        self.extras["log"].update(term_extras)

        # -- Robot reset --
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        if len(env_ids) == self.num_envs:
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        # -- Actions reset --
        self._actions[env_ids] = 0.0
        self._cmd_vel_b[env_ids] = 0.0

        # -- Targets --
        self._randomize_targets(env_ids)

        # -- Spawn position --
        spawn_local = self._sample_edge_positions(
            len(env_ids), self.cfg.target_spawn_range, self.cfg.spawn_min_height, self.cfg.spawn_max_height,
        )
        spawn_world = spawn_local.clone()
        spawn_world[:, :2] += self._terrain.env_origins[env_ids, :2]

        target_world = self._target_positions_local[env_ids].clone()
        target_world[:, :2] += self._terrain.env_origins[env_ids, :2]

        facing = torch.atan2(
            target_world[:, 1] - spawn_world[:, 1],
            target_world[:, 0] - spawn_world[:, 0],
        )
        half = facing * 0.5

        root_state = self._robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] = spawn_world
        root_state[:, 3] = torch.cos(half)
        root_state[:, 4] = 0.0
        root_state[:, 5] = 0.0
        root_state[:, 6] = torch.sin(half)
        root_state[:, 7:13] = 0.0

        self._robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(
            self._robot.data.default_joint_pos[env_ids],
            self._robot.data.default_joint_vel[env_ids],
            None,
            env_ids,
        )

        # -- Controller reset --
        ctrl_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=root_state.dtype)
        ctrl_att = torch.zeros((self.num_envs, 3), device=self.device, dtype=root_state.dtype)
        ctrl_pos[env_ids] = root_state[:, :3]
        ctrl_att[env_ids] = _quat_to_euler_deg(root_state[:, 3:7])
        self._controller.reset({"position": ctrl_pos, "attitude": ctrl_att}, env_ids=env_ids)
        self._controller.set_velocity_setpoint(
            vx=torch.zeros(len(env_ids), device=self.device),
            vy=torch.zeros(len(env_ids), device=self.device),
            vz=torch.zeros(len(env_ids), device=self.device),
            velocity_body=True,
            env_ids=env_ids,
        )

        # -- Depth camera reset --
        self._depth_camera.reset(env_ids)
        self._depth_stack_needs_fill[env_ids] = True
        self._update_prev_target_distance(env_ids)
        print("[DEBUG][SharedMapEnv] _reset_idx complete", flush=True)
