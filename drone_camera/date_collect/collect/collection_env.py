from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import dataclass

import gymnasium as gym
import torch

COLLECT_DIR = os.path.abspath(os.path.dirname(__file__))
DRONE_CAMERA_DIR = os.path.abspath(os.path.join(COLLECT_DIR, "..", ".."))
DRONE_ISAAC_DIR = os.path.abspath(os.path.join(DRONE_CAMERA_DIR, ".."))
ROOT_DIR = os.path.abspath(os.path.join(DRONE_ISAAC_DIR, ".."))
LOCAL_RSL_RL_DIR = os.path.join(DRONE_ISAAC_DIR, "local_rsl_rl")

for path in (ROOT_DIR, DRONE_ISAAC_DIR, DRONE_CAMERA_DIR, LOCAL_RSL_RL_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

import isaaclab.sim as sim_utils
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_mul
from pxr import Usd, UsdGeom

from controller import config as controller_config
from quadcopter_obstacles_student.quadcopter_obstacles_env import QuadcopterObstaclesEnv, QuadcopterObstaclesEnvCfg
from quadcopter_obstacles_student import agents


TASK_ID = "Isaac-Quadcopter-Obstacles-DepthCollect-v0"


@dataclass
class QuadcopterObstaclesDepthCollectEnvCfg(QuadcopterObstaclesEnvCfg):
    """Student environment plus a lightweight geometric depth-image observation."""

    policy_state_space: int = 16
    depth_image_height: int = 32
    depth_image_width: int = 48
    depth_camera_horizontal_fov_deg: float = 87.0
    depth_camera_vertical_fov_deg: float = 58.0
    depth_camera_min_range: float = 0.15
    depth_camera_max_range: float = 8.0
    depth_camera_offset_pos: tuple[float, float, float] = (0.12, 0.0, 0.0)
    depth_camera_offset_rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    depth_source: str = "geometric"

    def privileged_observation_dim(self) -> int:
        return self.policy_observation_dim()

    def __post_init__(self):
        super().__post_init__()
        self.policy_state_space = 16


class QuadcopterObstaclesDepthCollectEnv(QuadcopterObstaclesEnv):
    """Camera-style collection env that also exposes the privileged teacher observation."""

    def __init__(
        self,
        cfg: QuadcopterObstaclesDepthCollectEnvCfg | None = None,
        render_mode: str | None = None,
        **kwargs,
    ):
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)
        self.single_observation_space = gym.spaces.Dict(
            {
                "policy": gym.spaces.Box(
                    low=-float("inf"),
                    high=float("inf"),
                    shape=(self.cfg.privileged_observation_dim(),),
                    dtype=float,
                ),
                "policy_state": gym.spaces.Box(
                    low=-float("inf"),
                    high=float("inf"),
                    shape=(self.cfg.policy_state_space,),
                    dtype=float,
                ),
                "policy_image": gym.spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(1, self.cfg.depth_image_height, self.cfg.depth_image_width),
                    dtype=float,
                ),
            }
        )
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space, self.num_envs)
        self.num_states = self.cfg.privileged_observation_dim()
        self._depth_camera = None
        self._depth_dirs_b: torch.Tensor | None = None
        self._skip_depth_observation = False

    def _log_build_stage(self, name: str, start_time: float | None = None) -> float:
        if start_time is None:
            print(f"[INFO] DepthCollect build: {name}...", flush=True)
            return time.perf_counter()
        print(f"[INFO] DepthCollect build: {name} done in {time.perf_counter() - start_time:.2f}s", flush=True)
        return start_time

    def _build(self) -> None:
        if self._built:
            return

        start = self._log_build_stage("create simulation context")
        sim_cfg = sim_utils.SimulationCfg(dt=self.cfg.physics_dt, device=str(self.device))
        self._sim = sim_utils.SimulationContext(sim_cfg)
        self.sim.set_camera_view(
            eye=list(self.cfg.viewer_eye),
            target=list(self.cfg.viewer_lookat),
            camera_prim_path=self.cfg.viewer_cam_prim_path,
        )
        self._log_build_stage("create simulation context", start)

        start = self._log_build_stage("spawn ground and light")
        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_cfg.func("/World/Ground", ground_cfg)
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        self._log_build_stage("spawn ground and light", start)

        self._spawn_shared_obstacles()
        self._spawn_dynamic_obstacles()
        self._spawn_boundary_walls()
        self._build_robot_assets()
        self._disable_robot_collisions()

        start = self._log_build_stage("sim reset")
        self.sim.reset()
        self._log_build_stage("sim reset", start)

        start = self._log_build_stage("initialize robot view")
        self._body_id = self.robot.find_bodies("body")[0]
        self._robot_mass = self.robot.root_physx_view.get_masses()[0].sum().item()
        self.robot.write_joint_state_to_sim(self.robot.data.default_joint_pos, self.robot.data.default_joint_vel)
        self.robot.update(self.cfg.physics_dt)
        self._log_build_stage("initialize robot view", start)

        start = self._log_build_stage("warmup simulation")
        for _ in range(5):
            self.sim.step()
            self.robot.update(self.cfg.physics_dt)
        self._log_build_stage("warmup simulation", start)

        print(
            f"[INFO] Quadcopter Obstacles DepthCollect - Single World with {self.num_envs} robots, "
            f"{self.cfg.num_obstacles} static obstacles, and {self._num_dynamic_obstacles} dynamic obstacles",
            flush=True,
        )
        print(f"[INFO] Observation space: {self.cfg.observation_space}", flush=True)
        print(f"[INFO] Depth source: {self.cfg.depth_source}", flush=True)
        print(f"[INFO] OmniNxt mass from asset: {self._robot_mass:.4f} kg", flush=True)
        print(f"[INFO] Controller configured mass: {controller_config.CF_MASS:.4f} kg", flush=True)
        if abs(self._robot_mass - controller_config.CF_MASS) > 0.05:
            print("[WARN] OmniNxt mass and controller CF_MASS differ significantly. Retune controller/config.py.", flush=True)

        self._built = True

    def _spawn_shared_obstacles(self) -> None:
        start = self._log_build_stage("spawn static obstacles")
        super()._spawn_shared_obstacles()
        self._log_build_stage("spawn static obstacles", start)

    def _spawn_dynamic_obstacles(self) -> None:
        start = self._log_build_stage("spawn dynamic obstacles")
        super()._spawn_dynamic_obstacles()
        self._log_build_stage("spawn dynamic obstacles", start)

    def _spawn_boundary_walls(self) -> None:
        start = self._log_build_stage("spawn boundary walls")
        super()._spawn_boundary_walls()
        self._log_build_stage("spawn boundary walls", start)

    def _build_robot_assets(self) -> None:
        start = self._log_build_stage("spawn robots")
        super()._build_robot_assets()
        self._build_depth_camera()
        self._hide_robot_visual_geometry()
        self._log_build_stage("spawn robots", start)

    def _disable_robot_collisions(self) -> None:
        start = self._log_build_stage("disable robot collisions")
        super()._disable_robot_collisions()
        self._log_build_stage("disable robot collisions", start)

    def _build_depth_camera(self) -> None:
        if self.cfg.depth_source == "geometric":
            self._depth_camera = None
            return
        if self.cfg.depth_source != "camera":
            raise ValueError(f"Unknown depth_source: {self.cfg.depth_source}")

        import isaaclab.sim as sim_utils
        from isaaclab.sensors import TiledCamera, TiledCameraCfg

        horizontal_aperture = 20.955
        half_fov_rad = math.radians(self.cfg.depth_camera_horizontal_fov_deg) * 0.5
        focal_length = horizontal_aperture / (2.0 * math.tan(half_fov_rad))
        camera_cfg = TiledCameraCfg(
            prim_path="/World/Robot_.*/OmniNxt/DepthCamera",
            offset=TiledCameraCfg.OffsetCfg(
                pos=self.cfg.depth_camera_offset_pos,
                rot=self.cfg.depth_camera_offset_rot,
                convention="world",
            ),
            data_types=["depth"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=focal_length,
                focus_distance=400.0,
                horizontal_aperture=horizontal_aperture,
                vertical_aperture=2.0
                * focal_length
                * math.tan(math.radians(self.cfg.depth_camera_vertical_fov_deg) * 0.5),
                clipping_range=(self.cfg.depth_camera_min_range, self.cfg.depth_camera_max_range),
            ),
            width=self.cfg.depth_image_width,
            height=self.cfg.depth_image_height,
            depth_clipping_behavior="max",
        )
        self._depth_camera = TiledCamera(camera_cfg)

    def _hide_robot_visual_geometry(self) -> None:
        """Prevent exported depth images from including parallel drone visuals."""
        stage = self.sim.stage
        for env_idx in range(self.num_envs):
            root_path = f"/World/Robot_{env_idx:02d}/OmniNxt"
            root_prim = stage.GetPrimAtPath(root_path)
            if not root_prim.IsValid():
                continue
            for prim in Usd.PrimRange(root_prim):
                prim_path = prim.GetPath().pathString
                if "DepthCamera" in prim_path:
                    continue
                if prim.IsA(UsdGeom.Gprim):
                    UsdGeom.Imageable(prim).MakeInvisible()

    def _sync_depth_camera_to_robot(self) -> None:
        if self._depth_camera is None:
            return
        offset_pos_b = torch.tensor(
            self.cfg.depth_camera_offset_pos,
            device=self.device,
            dtype=self.robot.data.root_pos_w.dtype,
        ).unsqueeze(0).expand(self.num_envs, -1)
        offset_quat_b = torch.tensor(
            self.cfg.depth_camera_offset_rot,
            device=self.device,
            dtype=self.robot.data.root_quat_w.dtype,
        ).unsqueeze(0).expand(self.num_envs, -1)
        camera_pos_w = self.robot.data.root_pos_w + quat_apply(self.robot.data.root_quat_w, offset_pos_b)
        camera_quat_w = quat_mul(self.robot.data.root_quat_w, offset_quat_b)
        self._depth_camera.set_world_poses(camera_pos_w, camera_quat_w, convention="world")

    def capture_depth_camera_image(self) -> torch.Tensor:
        return self._get_depth_camera_image()

    def _get_depth_camera_image(self) -> torch.Tensor:
        if self.cfg.depth_source == "geometric":
            return self._get_geometric_depth_image()
        if self._depth_camera is None:
            raise RuntimeError("Depth camera has not been built yet.")

        self._depth_camera.update(0.0, force_recompute=True)
        depth = self._depth_camera.data.output["depth"].to(self.device)
        depth = torch.nan_to_num(depth, nan=self.cfg.depth_camera_max_range, posinf=self.cfg.depth_camera_max_range)
        depth = depth.clamp(self.cfg.depth_camera_min_range, self.cfg.depth_camera_max_range)
        depth = (depth - self.cfg.depth_camera_min_range) / (
            self.cfg.depth_camera_max_range - self.cfg.depth_camera_min_range
        )
        return depth.permute(0, 3, 1, 2).contiguous()

    def _get_depth_dirs_b(self) -> torch.Tensor:
        height = int(self.cfg.depth_image_height)
        width = int(self.cfg.depth_image_width)
        expected_shape = (height * width, 3)
        if self._depth_dirs_b is not None and tuple(self._depth_dirs_b.shape) == expected_shape:
            return self._depth_dirs_b

        rows = torch.arange(height, device=self.device, dtype=torch.float32)
        cols = torch.arange(width, device=self.device, dtype=torch.float32)
        grid_v, grid_u = torch.meshgrid(rows, cols, indexing="ij")
        u = ((grid_u + 0.5) / float(width)) * 2.0 - 1.0
        v = 1.0 - ((grid_v + 0.5) / float(height)) * 2.0
        tan_h = torch.tan(torch.tensor(math.radians(self.cfg.depth_camera_horizontal_fov_deg) * 0.5, device=self.device))
        tan_v = torch.tan(torch.tensor(math.radians(self.cfg.depth_camera_vertical_fov_deg) * 0.5, device=self.device))
        dirs = torch.stack(
            [
                torch.ones_like(u),
                -u * tan_h,
                v * tan_v,
            ],
            dim=-1,
        ).reshape(-1, 3)
        self._depth_dirs_b = dirs / torch.linalg.norm(dirs, dim=-1, keepdim=True).clamp_min(1.0e-6)
        return self._depth_dirs_b

    def _aabb_depth(
        self,
        origins_w: torch.Tensor,
        dirs_w: torch.Tensor,
        box_mins_w: torch.Tensor,
        box_maxs_w: torch.Tensor,
        depth: torch.Tensor,
        *,
        chunk_size: int = 48,
    ) -> torch.Tensor:
        if box_mins_w.numel() == 0:
            return depth

        min_range = float(self.cfg.depth_camera_min_range)
        eps = 1.0e-6
        origins = origins_w[:, None, None, :]
        dirs = dirs_w[:, :, None, :]
        inv_dirs = torch.where(torch.abs(dirs) > eps, 1.0 / dirs, torch.full_like(dirs, 1.0 / eps))

        for start in range(0, box_mins_w.shape[0], chunk_size):
            box_min = box_mins_w[start : start + chunk_size][None, None, :, :]
            box_max = box_maxs_w[start : start + chunk_size][None, None, :, :]
            t1 = (box_min - origins) * inv_dirs
            t2 = (box_max - origins) * inv_dirs
            t_near = torch.minimum(t1, t2).amax(dim=-1)
            t_far = torch.maximum(t1, t2).amin(dim=-1)
            t = torch.where(t_near >= min_range, t_near, t_far)
            hit = (t_far >= torch.maximum(t_near, torch.full_like(t_near, min_range))) & (t > min_range)
            t = torch.where(hit, t, torch.full_like(t, float("inf")))
            depth = torch.minimum(depth, t.amin(dim=-1))
        return depth

    def _get_geometric_depth_image(self) -> torch.Tensor:
        height = int(self.cfg.depth_image_height)
        width = int(self.cfg.depth_image_width)
        num_pixels = height * width
        min_range = float(self.cfg.depth_camera_min_range)
        max_range = float(self.cfg.depth_camera_max_range)

        dirs_b = self._get_depth_dirs_b()
        dirs_w = quat_apply(
            self.robot.data.root_quat_w[:, None, :].expand(-1, num_pixels, -1).reshape(-1, 4),
            dirs_b[None, :, :].expand(self.num_envs, -1, -1).reshape(-1, 3),
        ).reshape(self.num_envs, num_pixels, 3)

        offset_pos_b = torch.tensor(
            self.cfg.depth_camera_offset_pos,
            device=self.device,
            dtype=self.robot.data.root_pos_w.dtype,
        ).unsqueeze(0).expand(self.num_envs, -1)
        origins_w = self.robot.data.root_pos_w + quat_apply(self.robot.data.root_quat_w, offset_pos_b)
        depth = torch.full((self.num_envs, num_pixels), max_range, device=self.device, dtype=torch.float32)

        static_half = torch.stack(
            [
                self._obstacle_radii,
                self._obstacle_radii,
                self._obstacle_heights * 0.5,
            ],
            dim=-1,
        )
        box_mins = self._obstacle_positions_w - static_half
        box_maxs = self._obstacle_positions_w + static_half
        if self._num_dynamic_obstacles > 0:
            dynamic_half = self._dynamic_obstacle_sizes * 0.5
            box_mins = torch.cat([box_mins, self._dynamic_obstacle_positions_w - dynamic_half], dim=0)
            box_maxs = torch.cat([box_maxs, self._dynamic_obstacle_positions_w + dynamic_half], dim=0)

        half = float(self.cfg.wall_half_extent)
        thickness = float(self.cfg.wall_thickness)
        wall_height = float(self.cfg.wall_height)
        wall_specs = torch.tensor(
            [
                [-half - thickness * 0.5, half - thickness * 0.5, 0.0, half + thickness * 0.5, half + thickness * 0.5, wall_height],
                [-half - thickness * 0.5, -half - thickness * 0.5, 0.0, half + thickness * 0.5, -half + thickness * 0.5, wall_height],
                [half - thickness * 0.5, -half - thickness * 0.5, 0.0, half + thickness * 0.5, half + thickness * 0.5, wall_height],
                [-half - thickness * 0.5, -half - thickness * 0.5, 0.0, -half + thickness * 0.5, half + thickness * 0.5, wall_height],
            ],
            device=self.device,
            dtype=torch.float32,
        )
        box_mins = torch.cat([box_mins, wall_specs[:, :3]], dim=0)
        box_maxs = torch.cat([box_maxs, wall_specs[:, 3:]], dim=0)
        depth = self._aabb_depth(origins_w, dirs_w, box_mins, box_maxs, depth)

        down = dirs_w[..., 2] < -1.0e-6
        t_ground = (0.0 - origins_w[:, None, 2]) / dirs_w[..., 2].clamp_max(-1.0e-6)
        ground_xy = origins_w[:, None, :2] + t_ground[..., None] * dirs_w[..., :2]
        ground_hit = (
            down
            & (t_ground > min_range)
            & (torch.abs(ground_xy[..., 0]) <= half)
            & (torch.abs(ground_xy[..., 1]) <= half)
        )
        depth = torch.minimum(depth, torch.where(ground_hit, t_ground, torch.full_like(t_ground, float("inf"))))
        depth = depth.clamp(min_range, max_range)
        depth = (depth - min_range) / (max_range - min_range)
        return depth.reshape(self.num_envs, height, width).unsqueeze(1).contiguous()

    def _get_observations(self) -> dict[str, torch.Tensor]:
        root_pos_w = self.robot.data.root_pos_w
        root_quat_w = self.robot.data.root_quat_w
        root_lin_vel_b = quat_apply_inverse(root_quat_w, self.robot.data.root_lin_vel_w)
        root_ang_vel_b = quat_apply_inverse(root_quat_w, self.robot.data.root_ang_vel_w)
        projected_gravity_b = quat_apply_inverse(root_quat_w, self._gravity_vec_w)
        target_vec_w = self._target_positions_w - root_pos_w
        target_pos_b = quat_apply_inverse(root_quat_w, target_vec_w)

        target_distance = torch.linalg.norm(target_vec_w, dim=1)
        target_direction_w = target_vec_w / target_distance.unsqueeze(1).clamp_min(1.0e-6)
        closest_dirs_b, closest_dists = self._compute_closest_obstacles_directional()
        dynamic_features, _, _ = self._compute_closest_dynamic_obstacle_features(root_pos_w, target_direction_w)
        forward_distance = self._compute_forward_obstacle_distance(root_pos_w, root_quat_w)
        target_ray_distance = self._compute_target_ray_obstacle_distance(root_pos_w, target_direction_w)

        base_state = torch.cat(
            [
                root_lin_vel_b,
                root_ang_vel_b,
                projected_gravity_b,
                target_pos_b,
                self._cmd_vel_b,
            ],
            dim=-1,
        )
        policy = torch.cat(
            [
                base_state,
                closest_dirs_b.reshape(self.num_envs, -1),
                closest_dists,
                dynamic_features.reshape(self.num_envs, -1),
                (forward_distance / self.cfg.obstacle_detection_range).unsqueeze(1),
                (target_ray_distance / self.cfg.obstacle_detection_range).unsqueeze(1),
                self._target_reached.float().unsqueeze(1),
            ],
            dim=-1,
        )
        policy_state = torch.cat([base_state, self._target_reached.float().unsqueeze(1)], dim=-1)
        if self._skip_depth_observation:
            policy_image = torch.ones(
                (self.num_envs, 1, self.cfg.depth_image_height, self.cfg.depth_image_width),
                device=self.device,
                dtype=policy.dtype,
            )
        else:
            policy_image = self._get_depth_camera_image()
        return {
            "policy": policy,
            "policy_state": policy_state,
            "policy_image": policy_image,
        }


try:
    gym.spec(TASK_ID)
except gym.error.NameNotFound:
    gym.register(
        id=TASK_ID,
        entry_point=f"{__name__}:QuadcopterObstaclesDepthCollectEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}:QuadcopterObstaclesDepthCollectEnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QuadcopterObstaclesPPORunnerCfg",
        },
    )
