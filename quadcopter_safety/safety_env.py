from __future__ import annotations

import gymnasium as gym
import omni.replicator.core as rep
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import CUBOID_MARKER_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.camera import TiledCamera, TiledCameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply

from assets.omninxt.omninxt import OMNINXT_CFG
from controller import CrazyflieController, config as controller_config


def _quat_to_euler_deg(quat_w: torch.Tensor) -> torch.Tensor:
    """Convert (w, x, y, z) quaternions to Euler XYZ degrees."""
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


def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Multiply quaternions in (w, x, y, z) format."""
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)
    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )


@configclass
class QuadcopterSafetyEnvCfg(DirectRLEnvCfg):
    """GPU-compatible phase-1 safety command task."""

    episode_length_s = 30.0
    decimation = 2
    observation_space = 98
    action_space = 3
    state_space = 0
    debug_vis = False

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

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=45.0,
        replicate_physics=True,
    )

    robot: ArticulationCfg = OMNINXT_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    cmd_body_vel_xy_max = 0.8
    cmd_vel_z_max = 0.5
    forward_body_axis_sign = -1.0
    enforce_forward_commands = True

    num_obstacles = 100
    num_closest_obstacles = 5
    obstacle_height = 1.5
    obstacle_radius = 0.15
    obstacle_spawn_range = 20.0
    obstacle_safe_zone = 1.0
    obstacle_min_separation = 0.6
    obstacle_detection_range = 4.0

    depth_camera_width = 32
    depth_camera_height = 24
    depth_camera_near_clip = 0.2
    depth_camera_far_clip = 8.0
    depth_sector_count = 32
    depth_slice_half_thickness = 0.30
    use_first_person_view = True
    show_rgb_preview = True
    show_depth_preview = True
    depth_preview_env_index = 0
    # Display-only correction for the follow camera.
    # The training camera uses ROS convention; for the human-facing viewport we
    # add a 180 deg roll around the optical axis so the world is upright.
    follow_camera_display_correction = (0.0, 0.0, 0.0, 1.0)
    
    # 【注意】：在运行一次后，如果发现相机仍然没有跟随无人机，
    # 请根据终端打印出的 "[DEBUG] 你的 OmniNxt 拥有的物理连杆包含: [...]" 
    # 将下方 prim_path 中的 "body" 替换为正确的连杆名称（例如 "base_link"）
    depth_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/body/front_depth_camera",
        update_period=0.0,
        height=24,
        width=32,
        data_types=["rgb", "depth"],
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

    vel_tracking_exp_scale = 4.0
    desired_cmd_resample_min_steps = 100
    desired_cmd_resample_max_steps = 251
    cmd_tracking_reward_scale = 4.0
    obstacle_proximity_reward_scale = -8.0
    safety_margin_reward_scale = 0.6
    cmd_ttc_reward_scale = 0.8
    vel_ttc_penalty_scale = -1.0
    unsafe_cmd_penalty_scale = -4.0
    stall_penalty_scale = -2.0
    backward_cmd_penalty_scale = -3.0
    action_smoothness_penalty_scale = -0.05
    ang_vel_reward_scale = -0.01
    height_penalty_scale = -8.0


class QuadcopterSafetyEnv(DirectRLEnv):
    """High-level safety command policy task."""

    cfg: QuadcopterSafetyEnvCfg

    def __init__(self, cfg: QuadcopterSafetyEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._cmd_vel_b = torch.zeros(self.num_envs, 3, device=self.device)
        self._desired_cmd_b = torch.zeros(self.num_envs, 3, device=self.device)
        self._prev_cmd_vel_b = torch.zeros(self.num_envs, 3, device=self.device)
        self._obstacle_positions_local = torch.zeros(
            self.num_envs, self.cfg.num_obstacles, 3, device=self.device
        )
        self._latest_sector_distances = torch.full(
            (self.num_envs, self.cfg.depth_sector_count),
            self.cfg.depth_camera_far_clip,
            device=self.device,
        )

        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "cmd_tracking",
                "ang_vel",
                "obstacle_proximity",
                "safety_margin",
                "cmd_ttc",
                "vel_ttc",
                "unsafe_cmd",
                "stall",
                "front_risk",
                "action_smoothness",
                "height",
                "vel_tracking_error",
            ]
        }

        # 尝试寻找 body 连杆。如果找不到，可能会索引越界。
        try:
            self._body_id = self._robot.find_bodies("body")[0]
        except IndexError:
            print(f"[ERROR] 找不到名为 'body' 的连杆！")
            
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum().item()
        self._controller = CrazyflieController(
            num_envs=self.num_envs,
            device=self.device,
            attitude_dt=self.step_dt,
            position_dt=self.step_dt,
        )
        self._depth_camera = self.scene.sensors["depth_camera"]
        self._collision_threshold = 0.05

        self._cmd_resample_interval_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._cmd_step_age = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._debug_reward_logged = False
        self._debug_done_logged = False
        self._debug_reset_logged = False
        self._depth_preview_window = None
        self._depth_preview_provider = None
        self._depth_preview_widget = None
        self._rgb_preview_window = None
        self._rgb_preview_provider = None
        self._rgb_preview_widget = None
        self._viewport_follow_cam_path = None
        self._viewport_rgb_annotator = None
        self._viewport_depth_annotator = None
        self._viewport_render_product = None

        all_env_ids = torch.arange(self.num_envs, device=self.device)
        self._randomize_obstacles(all_env_ids)
        self._sample_desired_cmd(all_env_ids)

        self._setup_first_person_view()
        self._setup_rgb_preview_ui()
        self._setup_depth_preview_ui()
        self.set_debug_vis(self.cfg.debug_vis)

        print("[INFO] Quadcopter Safety - Phase 1 High-Level Safety Commands")
        print(f"[INFO] Obstacles: {self.cfg.num_obstacles} | depth sectors: {self.cfg.depth_sector_count}")
        print("[INFO] Task: track desired velocity while avoiding collisions")
        print(f"[INFO] OmniNxt mass from asset: {self._robot_mass:.4f} kg")
        print(f"[INFO] Controller configured mass: {controller_config.CF_MASS:.4f} kg")
        
        # 调试输出：打印当前资产中所有的合法物理连杆名称
        print("\n=======================================================")
        print(f"[DEBUG] 你的 OmniNxt 拥有的物理连杆包含: {self._robot.body_names}")
        print("请检查上方列表，如果其中没有 'body'，请将 Config 中相机的 prim_path 以及本文件内的 find_bodies('body') 替换为正确的名称！")
        print("=======================================================\n")

    def _setup_first_person_view(self):
        """Use an independent camera that is updated from the UAV body pose every frame."""
        if not self.cfg.use_first_person_view:
            return
        try:
            from omni.kit.viewport.utility import get_active_viewport
            import omni.usd
            from pxr import UsdGeom
        except ImportError:
            return

        viewport = get_active_viewport()
        if viewport is None:
            return

        env_index = int(min(max(self.cfg.depth_preview_env_index, 0), self.num_envs - 1))
        viewport_cam_path = f"/World/ViewportFollowCamera"

        stage = omni.usd.get_context().get_stage()
        cam_prim = UsdGeom.Camera.Define(stage, viewport_cam_path)
        cam_prim.GetFocalLengthAttr().Set(12.0)
        cam_prim.GetHorizontalApertureAttr().Set(20.955)

        try:
            viewport.set_active_camera(viewport_cam_path)
            print(f"[INFO] Viewport using independent follow camera (not sensor camera)", flush=True)
        except Exception as exc:
            print(f"[WARN] Failed to set viewport camera: {exc}", flush=True)

        self._viewport_follow_cam_path = viewport_cam_path
        self._viewport_follow_env_index = env_index
        self._viewport_render_product = rep.create.render_product(
            viewport_cam_path,
            (self.cfg.depth_camera_width, self.cfg.depth_camera_height),
        )
        self._viewport_rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        self._viewport_rgb_annotator.attach([self._viewport_render_product])
        self._viewport_depth_annotator = rep.AnnotatorRegistry.get_annotator("distance_to_camera", device="cpu")
        self._viewport_depth_annotator.attach([self._viewport_render_product])

    def _update_follow_camera_pose(self):
        if not self.cfg.use_first_person_view or self._viewport_follow_cam_path is None:
            return
        try:
            import omni.usd
            from pxr import Gf, UsdGeom
        except ImportError:
            return

        env_index = self._viewport_follow_env_index
        root_pos_w = self._robot.data.root_pos_w[env_index : env_index + 1]
        root_quat_w = self._robot.data.root_quat_w[env_index : env_index + 1]
        cam_pos_b = torch.tensor(self.cfg.depth_camera.offset.pos, device=self.device).unsqueeze(0)
        cam_quat_b = torch.tensor(self.cfg.depth_camera.offset.rot, device=self.device).unsqueeze(0)
        cam_display_correction = torch.tensor(
            self.cfg.follow_camera_display_correction, device=self.device
        ).unsqueeze(0)

        cam_pos_w = root_pos_w + quat_apply(root_quat_w, cam_pos_b)
        cam_quat_w = _quat_mul(root_quat_w, _quat_mul(cam_quat_b, cam_display_correction))

        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(self._viewport_follow_cam_path)
        if not prim.IsValid():
            return
        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()
        translate_op = xform.AddTranslateOp()
        orient_op = xform.AddOrientOp()
        pos = cam_pos_w[0]
        quat = cam_quat_w[0]
        translate_op.Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
        orient_op.Set(Gf.Quatf(float(quat[0]), Gf.Vec3f(float(quat[1]), float(quat[2]), float(quat[3]))))

    def _setup_rgb_preview_ui(self):
        if not self.cfg.show_rgb_preview:
            return
        try:
            import omni.ui as ui
        except ImportError:
            return

        self._rgb_preview_provider = ui.ByteImageProvider()
        env_index = int(min(max(self.cfg.depth_preview_env_index, 0), self.num_envs - 1))
        self._rgb_preview_window = ui.Window(
            f"UAV Front RGB (env_{env_index})",
            width=max(220, self.cfg.depth_camera_width * 8),
            height=max(180, self.cfg.depth_camera_height * 8 + 24),
        )
        with self._rgb_preview_window.frame:
            with ui.VStack(spacing=4):
                ui.Label(f"Front RGB camera view from env_{env_index}")
                self._rgb_preview_widget = ui.ImageWithProvider(
                    self._rgb_preview_provider,
                    width=self.cfg.depth_camera_width * 8,
                    height=self.cfg.depth_camera_height * 8,
                )
        print(f"[INFO] RGB preview window created for UAV front camera: env_{env_index}", flush=True)

    def _setup_depth_preview_ui(self):
        if not self.cfg.show_depth_preview:
            return
        try:
            import omni.ui as ui
        except ImportError:
            return

        self._depth_preview_provider = ui.ByteImageProvider()
        env_index = int(min(max(self.cfg.depth_preview_env_index, 0), self.num_envs - 1))
        self._depth_preview_window = ui.Window(
            f"UAV Front Depth (env_{env_index})",
            width=max(220, self.cfg.depth_camera_width * 8),
            height=max(180, self.cfg.depth_camera_height * 8 + 24),
        )
        with self._depth_preview_window.frame:
            with ui.VStack(spacing=4):
                ui.Label(f"Front depth camera view from env_{env_index}")
                self._depth_preview_widget = ui.ImageWithProvider(
                    self._depth_preview_provider,
                    width=self.cfg.depth_camera_width * 8,
                    height=self.cfg.depth_camera_height * 8,
                )
        print(f"[INFO] Depth preview window created for UAV front depth camera: env_{env_index}", flush=True)

    def _update_rgb_preview_ui(self):
        if self._rgb_preview_provider is None or self._viewport_rgb_annotator is None:
            return
        rgb = self._viewport_rgb_annotator.get_data()
        if rgb is None or getattr(rgb, "size", 0) == 0:
            return
        rgb_cpu = torch.from_numpy(rgb[:, :, :3].copy()).to(torch.uint8)
        if rgb_cpu.shape[-1] == 3:
            alpha = torch.full((*rgb_cpu.shape[:2], 1), 255, dtype=torch.uint8)
            rgba = torch.cat([rgb_cpu, alpha], dim=-1)
        elif rgb_cpu.shape[-1] == 4:
            rgba = rgb_cpu
        else:
            return
            
        height, width = rgba.shape[:2]
        self._rgb_preview_provider.set_bytes_data(rgba.contiguous().numpy().flatten().data, [width, height])

    def _update_depth_preview_ui(self, depth_meters: torch.Tensor):
        if self._depth_preview_provider is None:
            return
        if self._viewport_depth_annotator is not None:
            depth_raw = self._viewport_depth_annotator.get_data()
            if depth_raw is not None and getattr(depth_raw, "size", 0) != 0:
                if depth_raw.ndim == 2:
                    depth_img = torch.from_numpy(depth_raw.copy())
                elif depth_raw.ndim == 3:
                    depth_img = torch.from_numpy(depth_raw[:, :, 0].copy())
                else:
                    env_index = int(min(max(self.cfg.depth_preview_env_index, 0), self.num_envs - 1))
                    depth_img = depth_meters[env_index].detach().cpu()
            else:
                env_index = int(min(max(self.cfg.depth_preview_env_index, 0), self.num_envs - 1))
                depth_img = depth_meters[env_index].detach().cpu()
        else:
            env_index = int(min(max(self.cfg.depth_preview_env_index, 0), self.num_envs - 1))
            depth_img = depth_meters[env_index].detach().cpu()
        depth_norm = 1.0 - (
            (depth_img - self.cfg.depth_camera_near_clip)
            / (self.cfg.depth_camera_far_clip - self.cfg.depth_camera_near_clip)
        )
        depth_u8 = (depth_norm.clamp(0.0, 1.0) * 255.0).to(torch.uint8).detach().cpu()
        height, width = depth_u8.shape
        rgba = torch.empty((height, width, 4), dtype=torch.uint8)
        rgba[..., 0] = depth_u8
        rgba[..., 1] = depth_u8
        rgba[..., 2] = depth_u8
        rgba[..., 3] = 255
        self._depth_preview_provider.set_bytes_data(rgba.contiguous().numpy().flatten().data, [width, height])

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self._depth_camera_sensor = TiledCamera(self.cfg.depth_camera)
        self.scene.sensors["depth_camera"] = self._depth_camera_sensor

        self._obstacles: list[RigidObject] = []
        obstacle_spawn_cfg = sim_utils.CuboidCfg(
            size=(
                self.cfg.obstacle_radius * 2,
                self.cfg.obstacle_radius * 2,
                self.cfg.obstacle_height,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
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

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _sync_render_obstacles(self, env_ids: torch.Tensor):
        env_origins = self._terrain.env_origins[env_ids]
        obstacle_pos_world = self._obstacle_positions_local[env_ids].clone()
        obstacle_pos_world[:, :, :2] += env_origins[:, :2].unsqueeze(1)

        for obstacle_idx, obstacle_obj in enumerate(self._obstacles):
            root_state = torch.zeros((len(env_ids), 13), device=self.device)
            root_state[:, :3] = obstacle_pos_world[:, obstacle_idx]
            root_state[:, 3] = 1.0
            obstacle_obj.write_root_state_to_sim(root_state, env_ids)

    def _randomize_obstacles(self, env_ids: torch.Tensor):
        num_envs_to_reset = len(env_ids)
        placed_xy = torch.zeros(num_envs_to_reset, self.cfg.num_obstacles, 2, device=self.device)

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
                    valid_mask &= min_dist >= self.cfg.obstacle_min_separation

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
        self._sync_render_obstacles(env_ids)

    def _sample_desired_cmd(self, env_ids: torch.Tensor):
        if len(env_ids) == 0:
            return
        planar_speed = torch.empty(len(env_ids), device=self.device).uniform_(0.0, self.cfg.cmd_body_vel_xy_max)
        planar_heading = torch.empty(len(env_ids), device=self.device).uniform_(-0.5 * torch.pi, 0.5 * torch.pi)
        self._desired_cmd_b[env_ids, 0] = self.cfg.forward_body_axis_sign * planar_speed * torch.cos(planar_heading)
        self._desired_cmd_b[env_ids, 1] = planar_speed * torch.sin(planar_heading)
        self._desired_cmd_b[env_ids, 2] = torch.empty(len(env_ids), device=self.device).uniform_(
            -self.cfg.cmd_vel_z_max, self.cfg.cmd_vel_z_max
        )
        self._cmd_step_age[env_ids] = 0
        self._cmd_resample_interval_steps[env_ids] = torch.randint(
            low=self.cfg.desired_cmd_resample_min_steps,
            high=self.cfg.desired_cmd_resample_max_steps,
            size=(len(env_ids),),
            device=self.device,
        )

    def _get_depth_image_meters(self) -> torch.Tensor:
        self._update_follow_camera_pose()
        depth = self._depth_camera.data.output["depth"].squeeze(-1)
        depth = torch.nan_to_num(
            depth,
            nan=self.cfg.depth_camera_far_clip,
            posinf=self.cfg.depth_camera_far_clip,
            neginf=self.cfg.depth_camera_near_clip,
        )
        depth = depth.clamp(self.cfg.depth_camera_near_clip, self.cfg.depth_camera_far_clip)
        self._update_rgb_preview_ui()
        self._update_depth_preview_ui(depth)
        return depth

    def _project_depth_to_sectors(self) -> torch.Tensor:
        depth = self._get_depth_image_meters()
        batch, height, width = depth.shape
        device = depth.device

        focal_length = self.cfg.depth_camera.spawn.focal_length
        horizontal_aperture = self.cfg.depth_camera.spawn.horizontal_aperture
        fx = (focal_length / horizontal_aperture) * width
        fy = fx
        cx = (width - 1) * 0.5
        cy = (height - 1) * 0.5

        u = torch.arange(width, device=device, dtype=depth.dtype)
        v = torch.arange(height, device=device, dtype=depth.dtype)
        uu, vv = torch.meshgrid(u, v, indexing="xy")
        z = depth
        x = (uu.unsqueeze(0) - cx) / fx * z
        y = (vv.unsqueeze(0) - cy) / fy * z
        points_c = torch.stack([x, y, z], dim=-1).reshape(batch, height * width, 3)

        cam_quat_b = torch.tensor(self.cfg.depth_camera.offset.rot, device=device, dtype=depth.dtype).unsqueeze(0).expand(batch, -1)
        cam_pos_b = torch.tensor(self.cfg.depth_camera.offset.pos, device=device, dtype=depth.dtype).unsqueeze(0).expand(batch, -1)
        points_b = quat_apply(
            cam_quat_b.unsqueeze(1).expand(-1, height * width, -1).reshape(batch * height * width, 4),
            points_c.reshape(batch * height * width, 3),
        ).reshape(batch, height * width, 3)
        points_b = points_b + cam_pos_b.unsqueeze(1)

        horizontal_mask = points_b[:, :, 0] > 0.0
        horizontal_mask &= points_b[:, :, 2].abs() <= self.cfg.depth_slice_half_thickness

        xy_norm = torch.linalg.norm(points_b[:, :, :2], dim=-1)
        angles = torch.atan2(points_b[:, :, 1], points_b[:, :, 0])
        sector_size = (2.0 * torch.pi) / self.cfg.depth_sector_count
        sector_idx = torch.floor((angles + torch.pi) / sector_size).long().clamp(0, self.cfg.depth_sector_count - 1)

        sector_distances = torch.full(
            (batch, self.cfg.depth_sector_count),
            self.cfg.depth_camera_far_clip,
            device=device,
            dtype=depth.dtype,
        )

        batch_idx = torch.arange(batch, device=device).unsqueeze(1).expand(-1, height * width)
        flat_sector_idx = batch_idx * self.cfg.depth_sector_count + sector_idx
        sector_distances_flat = sector_distances.reshape(-1)
        sector_distances_flat.scatter_reduce_(
            0,
            flat_sector_idx[horizontal_mask],
            xy_norm[horizontal_mask],
            reduce="amin",
            include_self=False,
        )
        sector_distances = sector_distances_flat.reshape(batch, self.cfg.depth_sector_count)
        self._latest_sector_distances.copy_(sector_distances)
        return sector_distances

    def _compute_min_obstacle_distance(self) -> torch.Tensor:
        drone_pos_w = self._robot.data.root_pos_w[:, :2]
        env_origins_xy = self._terrain.env_origins[:, :2]
        drone_pos_local = drone_pos_w - env_origins_xy
        distances = torch.linalg.norm(
            drone_pos_local.unsqueeze(1) - self._obstacle_positions_local[:, :, :2],
            dim=2,
        )
        distances = distances - self.cfg.obstacle_radius
        return distances.min(dim=1).values

    def _compute_sector_ttc(self, sector_distances: torch.Tensor, planar_velocity_b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sector_angles = torch.linspace(
            -torch.pi,
            torch.pi,
            steps=self.cfg.depth_sector_count + 1,
            device=self.device,
            dtype=sector_distances.dtype,
        )[:-1]
        sector_dirs = torch.stack([torch.cos(sector_angles), torch.sin(sector_angles)], dim=-1)
        proj_speed = torch.sum(planar_velocity_b.unsqueeze(1) * sector_dirs.unsqueeze(0), dim=-1).clamp(min=0.0)
        ttc = (sector_distances.clamp(max=self.cfg.obstacle_detection_range) / (proj_speed + 1e-6)).clamp(max=3.0)
        return ttc, sector_dirs

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone().clamp(-1.0, 1.0)
        if self.cfg.enforce_forward_commands:
            if self.cfg.forward_body_axis_sign >= 0.0:
                self._actions[:, 0].clamp_(0.0, 1.0)
            else:
                self._actions[:, 0].clamp_(-1.0, 0.0)
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
        self._depth_camera.update(self.step_dt)
        sector_distances = self._project_depth_to_sectors()
        sector_distances_norm = (
            sector_distances / self.cfg.depth_camera_far_clip
        ).clamp(0.0, 1.0)

        obs = torch.cat(
            [
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                self._robot.data.projected_gravity_b,
                self._desired_cmd_b,
                self._cmd_vel_b,
                sector_distances_norm,
                torch.zeros(self.num_envs, 1, device=self.device),
            ],
            dim=-1,
        )

        pad = self.cfg.observation_space - obs.shape[1]
        if pad > 0:
            obs = torch.cat([obs, torch.zeros(self.num_envs, pad, device=self.device)], dim=-1)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        self._cmd_step_age += 1
        resample_ids = torch.nonzero(self._cmd_step_age >= self._cmd_resample_interval_steps, as_tuple=False).squeeze(-1)
        if resample_ids.numel() > 0:
            self._sample_desired_cmd(resample_ids)

        velocity_cmd_frame = torch.stack(
            [
                self._robot.data.root_lin_vel_b[:, 0],
                self._robot.data.root_lin_vel_b[:, 1],
                self._robot.data.root_lin_vel_w[:, 2],
            ],
            dim=1,
        )
        cmd_tracking_sq_error = torch.sum(torch.square(self._desired_cmd_b - velocity_cmd_frame), dim=1)
        cmd_tracking_error = torch.linalg.norm(self._desired_cmd_b - velocity_cmd_frame, dim=1)
        cmd_tracking = torch.exp(-self.cfg.vel_tracking_exp_scale * cmd_tracking_sq_error)

        ang_vel = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)
        sector_distances = self._project_depth_to_sectors()
        min_obstacle_dist = sector_distances.min(dim=1).values
        obstacle_proximity = torch.where(
            min_obstacle_dist < 1.0,
            torch.exp(-min_obstacle_dist * 3.0),
            torch.zeros_like(min_obstacle_dist),
        )
        safety_margin = torch.log(min_obstacle_dist.clamp(min=self.cfg.depth_camera_near_clip, max=2.5))
        cmd_ttc_full, sector_dirs = self._compute_sector_ttc(sector_distances, self._cmd_vel_b[:, :2])
        vel_ttc_full, _ = self._compute_sector_ttc(sector_distances, self._robot.data.root_lin_vel_b[:, :2])

        cmd_speed = torch.linalg.norm(self._cmd_vel_b[:, :2], dim=1)
        vel_speed = torch.linalg.norm(self._robot.data.root_lin_vel_b[:, :2], dim=1)
        desired_speed = torch.linalg.norm(self._desired_cmd_b[:, :2], dim=1)

        cmd_dir = self._cmd_vel_b[:, :2] / cmd_speed.unsqueeze(1).clamp_min(1e-6)
        vel_dir = self._robot.data.root_lin_vel_b[:, :2] / vel_speed.unsqueeze(1).clamp_min(1e-6)

        cmd_scores = torch.sum(cmd_dir.unsqueeze(1) * sector_dirs.unsqueeze(0), dim=-1)
        vel_scores = torch.sum(vel_dir.unsqueeze(1) * sector_dirs.unsqueeze(0), dim=-1)
        cmd_weights = torch.softmax(cmd_scores / 0.1, dim=-1)
        vel_weights = torch.softmax(vel_scores / 0.1, dim=-1)

        cmd_ttc = torch.sum(cmd_weights * cmd_ttc_full, dim=-1)
        vel_ttc = torch.sum(vel_weights * vel_ttc_full, dim=-1)
        cmd_ttc_reward = torch.log(cmd_ttc.clamp(min=0.1, max=3.0))
        vel_ttc_penalty = torch.exp(-2.0 * vel_ttc)
        unsafe_cmd = torch.relu(1.0 - cmd_ttc) * cmd_speed
        stall_penalty = (desired_speed > 0.2).float() * (cmd_ttc > 1.5).float() * (vel_speed < 0.1).float()
        backward_cmd_penalty = torch.relu(-self.cfg.forward_body_axis_sign * self._cmd_vel_b[:, 0])

        action_smoothness = torch.linalg.norm(self._cmd_vel_b - self._prev_cmd_vel_b, dim=1)

        z_pos = self._robot.data.root_pos_w[:, 2]
        height_penalty = torch.where(z_pos < 0.2, (0.2 - z_pos).square(), torch.zeros_like(z_pos))
        height_penalty += torch.where(z_pos > 2.5, (z_pos - 2.5).square(), torch.zeros_like(z_pos))

        rewards = {
            "cmd_tracking": cmd_tracking * self.cfg.cmd_tracking_reward_scale * self.step_dt,
            "ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
            "obstacle_proximity": obstacle_proximity * self.cfg.obstacle_proximity_reward_scale * self.step_dt,
            "safety_margin": safety_margin * self.cfg.safety_margin_reward_scale * self.step_dt,
            "cmd_ttc": cmd_ttc_reward * self.cfg.cmd_ttc_reward_scale * self.step_dt,
            "vel_ttc": vel_ttc_penalty * self.cfg.vel_ttc_penalty_scale * self.step_dt,
            "unsafe_cmd": unsafe_cmd * self.cfg.unsafe_cmd_penalty_scale * self.step_dt,
            "stall": stall_penalty * self.cfg.stall_penalty_scale * self.step_dt,
            "front_risk": backward_cmd_penalty * self.cfg.backward_cmd_penalty_scale * self.step_dt,
            "action_smoothness": action_smoothness * self.cfg.action_smoothness_penalty_scale * self.step_dt,
            "height": height_penalty * self.cfg.height_penalty_scale * self.step_dt,
        }

        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        for key, value in rewards.items():
            self._episode_sums[key] += value
        self._episode_sums["vel_tracking_error"] += cmd_tracking_error * self.step_dt
        self._prev_cmd_vel_b.copy_(self._cmd_vel_b)
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        too_low = self._robot.data.root_pos_w[:, 2] < 0.1
        too_high = self._robot.data.root_pos_w[:, 2] > 2.5
        min_obstacle_dist = self._compute_min_obstacle_distance()
        collision = min_obstacle_dist < self._collision_threshold
        died = too_low | too_high | collision
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        extras = {}
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            if key == "vel_tracking_error":
                extras["Metrics/avg_vel_tracking_error"] = episodic_sum_avg / self.max_episode_length_s
            else:
                extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        self.extras["log"]["Episode_Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        self.extras["log"]["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        if len(env_ids) == self.num_envs:
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0
        self._cmd_vel_b[env_ids] = 0.0
        self._prev_cmd_vel_b[env_ids] = 0.0
        self._randomize_obstacles(env_ids)
        self._sample_desired_cmd(env_ids)

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
        self._depth_camera.reset(env_ids)

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "obstacle_visualizer"):
                obs_marker_cfg = CUBOID_MARKER_CFG.copy()
                pillar_size = self.cfg.obstacle_radius * 2
                obs_marker_cfg.markers["cuboid"].size = (pillar_size, pillar_size, self.cfg.obstacle_height)
                obs_marker_cfg.markers["cuboid"].visual_material = sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.8, 0.2, 0.2)
                )
                obs_marker_cfg.prim_path = "/Visuals/SafetyObstacles"
                self.obstacle_visualizer = VisualizationMarkers(obs_marker_cfg)
            self.obstacle_visualizer.set_visibility(True)
        elif hasattr(self, "obstacle_visualizer"):
            self.obstacle_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if hasattr(self, "obstacle_visualizer"):
            env_origins = self._terrain.env_origins
            env_origins_expanded = env_origins.unsqueeze(1).repeat(1, self.cfg.num_obstacles, 1)
            obstacle_pos_w = self._obstacle_positions_local.clone()
            obstacle_pos_w[:, :, :2] += env_origins_expanded[:, :, :2]
            self.obstacle_visualizer.visualize(obstacle_pos_w.reshape(-1, 3))
