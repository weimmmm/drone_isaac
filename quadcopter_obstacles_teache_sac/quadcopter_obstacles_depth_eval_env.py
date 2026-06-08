from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import omni.kit.commands
import omni.usd
import torch
from isaaclab.sensors.camera import TiledCamera, TiledCameraCfg
from isaaclab.utils.math import quat_apply
from pxr import Sdf, UsdGeom

import isaaclab.sim as sim_utils

from .quadcopter_obstacles_env import QuadcopterObstaclesEnv, QuadcopterObstaclesEnvCfg


def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
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


@dataclass
class QuadcopterObstaclesDepthEvalEnvCfg(QuadcopterObstaclesEnvCfg):
    depth_camera_width: int = 64
    depth_camera_height: int = 48
    depth_camera_near_clip: float = 0.2
    depth_camera_far_clip: float = 8.0
    use_first_person_view: bool = True
    show_depth_preview: bool = True
    depth_preview_env_index: int = 0
    debug_logs: bool = False
    # Display-only correction for the human-facing viewport follow camera.
    # Rotate 180 deg around body-up so the viewport looks out of the UAV nose
    # instead of backwards.
    follow_camera_display_correction: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)


class QuadcopterObstaclesDepthEvalEnv(QuadcopterObstaclesEnv):
    """Evaluation-only variant of the trained teacher env with a forward depth camera per robot."""

    cfg: QuadcopterObstaclesDepthEvalEnvCfg

    def _debug(self, message: str) -> None:
        if getattr(self.cfg, "debug_logs", False):
            print(message, flush=True)

    def _info(self, message: str) -> None:
        print(message, flush=True)

    def __init__(self, cfg: QuadcopterObstaclesDepthEvalEnvCfg | None = None, render_mode: str | None = None, **kwargs):
        super().__init__(cfg=cfg if cfg is not None else QuadcopterObstaclesDepthEvalEnvCfg(), render_mode=render_mode, **kwargs)
        self._depth_camera: TiledCamera | None = None
        self._record_robot_index = 0
        self._depth_preview_window = None
        self._depth_preview_provider = None
        self._depth_preview_widget = None
        self._viewport_follow_cam_path = None
        self._viewport_depth_annotator = None
        self._viewport_render_product = None
        self._viewport_follow_env_index = 0
        self._latest_record_depth_frames: torch.Tensor | None = None

    @property
    def depth_camera(self) -> TiledCamera:
        if self._depth_camera is None:
            raise RuntimeError("Depth camera has not been built yet.")
        return self._depth_camera

    def set_record_robot_index(self, robot_index: int) -> None:
        self._record_robot_index = int(np.clip(robot_index, 0, self.num_envs - 1))

    def _hide_robots_from_depth(self) -> None:
        """Hide robot visual geometry from secondary rays so depth does not see the robot body."""
        stage = omni.usd.get_context().get_stage()
        for idx in range(self.num_envs):
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

    def _build_depth_cameras(self) -> None:
        camera_cfg = TiledCameraCfg(
            prim_path="/World/Robot_.*/OmniNxt/body/front_depth_camera",
            update_period=0.0,
            height=self.cfg.depth_camera_height,
            width=self.cfg.depth_camera_width,
            data_types=["depth"],
            depth_clipping_behavior="max",
            offset=TiledCameraCfg.OffsetCfg(
                pos=(0.10, 0.0, 0.03),
                rot=(1.0, 0.0, 0.0, 0.0),
                convention="world",
            ),
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=12.0,
                focus_distance=4.0,
                horizontal_aperture=20.955,
                clipping_range=(self.cfg.depth_camera_near_clip, self.cfg.depth_camera_far_clip),
            ),
        )
        self._depth_camera = TiledCamera(camera_cfg)

    def _setup_first_person_view(self) -> None:
        if not self.cfg.use_first_person_view:
            return
        try:
            import omni.replicator.core as rep
        except ImportError:
            self._info("[WARN] omni.replicator.core is unavailable; disabling first-person viewport depth capture.")
            self._viewport_follow_cam_path = None
            self._viewport_depth_annotator = None
            self._viewport_render_product = None
            return
        env_index = int(min(max(self.cfg.depth_preview_env_index, 0), self.num_envs - 1))
        viewport_cam_path = "/World/ViewportFollowCamera"
        stage = omni.usd.get_context().get_stage()
        cam_prim = UsdGeom.Camera.Define(stage, viewport_cam_path)
        cam_prim.GetFocalLengthAttr().Set(12.0)
        cam_prim.GetHorizontalApertureAttr().Set(20.955)

        try:
            from omni.kit.viewport.utility import get_active_viewport

            viewport = get_active_viewport()
            if viewport is not None:
                try:
                    viewport.set_active_camera(viewport_cam_path)
                    self._info("[INFO] Viewport using UAV follow camera")
                except Exception as exc:
                    self._info(f"[WARN] Failed to set viewport camera: {exc}")
        except ImportError:
            pass

        self._viewport_follow_cam_path = viewport_cam_path
        self._viewport_follow_env_index = env_index
        self._viewport_render_product = rep.create.render_product(
            viewport_cam_path,
            (self.cfg.depth_camera_width, self.cfg.depth_camera_height),
        )
        self._viewport_depth_annotator = rep.AnnotatorRegistry.get_annotator("distance_to_camera", device="cpu")
        self._viewport_depth_annotator.attach([self._viewport_render_product])

    def _update_follow_camera_pose(self) -> None:
        if not self.cfg.use_first_person_view or self._viewport_follow_cam_path is None:
            return

        env_index = self._viewport_follow_env_index
        root_pos_w = self.robot.data.root_pos_w[env_index : env_index + 1]
        root_quat_w = self.robot.data.root_quat_w[env_index : env_index + 1]
        cam_pos_b = torch.tensor((0.10, 0.0, 0.03), device=self.device).unsqueeze(0)
        cam_pos_w = root_pos_w + quat_apply(root_quat_w, cam_pos_b)
        # OmniNxt body-frame forward is +X. Drive the human-facing viewport with
        # an explicit look-at target so it always points out of the nose.
        forward_b = torch.tensor((1.0, 0.0, 0.0), device=self.device).unsqueeze(0)
        lookahead_distance = 2.0
        look_target_w = cam_pos_w + quat_apply(root_quat_w, forward_b) * lookahead_distance

        self.sim.set_camera_view(
            eye=cam_pos_w[0].detach().cpu().tolist(),
            target=look_target_w[0].detach().cpu().tolist(),
            camera_prim_path=self._viewport_follow_cam_path,
        )

    def _setup_depth_preview_ui(self) -> None:
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
        self._info(f"[INFO] Depth preview window created for UAV front depth camera: env_{env_index}")

    def _update_depth_preview_ui(self, depth_frames_meters: torch.Tensor) -> None:
        if self._depth_preview_provider is None:
            return
        env_index = int(min(max(self.cfg.depth_preview_env_index, 0), self.num_envs - 1))
        depth_img = depth_frames_meters[env_index].detach().cpu()
        if self._viewport_depth_annotator is not None:
            depth_raw = self._viewport_depth_annotator.get_data()
            if depth_raw is not None and getattr(depth_raw, "size", 0) != 0:
                if depth_raw.ndim == 2:
                    depth_img = torch.from_numpy(depth_raw.copy())
                elif depth_raw.ndim == 3:
                    depth_img = torch.from_numpy(depth_raw[:, :, 0].copy())
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

    def _build(self) -> None:
        if self._built:
            return

        self._debug("[DEBUG][depth_env] _build start")
        sim_cfg = sim_utils.SimulationCfg(dt=self.cfg.physics_dt, device=str(self.device))
        self._sim = sim_utils.SimulationContext(sim_cfg)
        self._debug("[DEBUG][depth_env] SimulationContext ready")
        self.sim.set_camera_view(
            eye=list(self.cfg.viewer_eye),
            target=list(self.cfg.viewer_lookat),
            camera_prim_path=self.cfg.viewer_cam_prim_path,
        )

        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_cfg.func("/World/Ground", ground_cfg)
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        self._debug("[DEBUG][depth_env] ground/light ready")

        self._spawn_shared_obstacles()
        self._debug("[DEBUG][depth_env] obstacles spawned")
        self._spawn_boundary_walls()
        self._debug("[DEBUG][depth_env] walls spawned")
        self._build_robot_assets()
        self._debug("[DEBUG][depth_env] robot assets built")
        self._disable_robot_collisions()
        self._debug("[DEBUG][depth_env] robot collisions disabled")
        self._hide_robots_from_depth()
        self._debug("[DEBUG][depth_env] robots hidden from depth")
        self._build_depth_cameras()
        self._debug("[DEBUG][depth_env] depth cameras built")
        self._setup_first_person_view()
        self._debug("[DEBUG][depth_env] first person view setup")
        self._setup_depth_preview_ui()
        self._debug("[DEBUG][depth_env] depth preview UI setup")

        self.sim.reset()
        self._debug("[DEBUG][depth_env] sim.reset done")
        self._body_id = self.robot.find_bodies("body")[0]
        self._robot_mass = self.robot.root_physx_view.get_masses()[0].sum().item()
        self.robot.write_joint_state_to_sim(self.robot.data.default_joint_pos, self.robot.data.default_joint_vel)
        self.robot.update(self.cfg.physics_dt)
        for _ in range(5):
            self.sim.step()
            self.robot.update(self.cfg.physics_dt)
        self._debug("[DEBUG][depth_env] warmup sim steps done")
        self.depth_camera.update(self.cfg.physics_dt)
        self._debug("[DEBUG][depth_env] depth camera update done")

        self._info(
            f"[INFO] Quadcopter Obstacles Teacher Depth Eval - Single World with {self.num_envs} robots and "
            f"{self.cfg.num_obstacles} shared obstacles"
        )
        self._info(f"[INFO] Observation space: {self.cfg.observation_space}")
        self._info(
            f"[INFO] Depth camera shape: {self.cfg.depth_camera_height}x{self.cfg.depth_camera_width}, "
            f"range=({self.cfg.depth_camera_near_clip}, {self.cfg.depth_camera_far_clip})"
        )
        self._info(f"[INFO] OmniNxt mass from asset: {self._robot_mass:.4f} kg")
        self._built = True

    def get_depth_frames(self) -> torch.Tensor:
        """Return normalized depth frames for all robots in [0, 1], where larger means closer obstacle."""
        if not self._built:
            self._build()
        self._update_follow_camera_pose()
        self.sim.render()
        self.depth_camera.update(self.cfg.physics_dt)
        depth = self.depth_camera.data.output["depth"].squeeze(-1).to(self.device)
        depth = torch.nan_to_num(
            depth,
            nan=self.cfg.depth_camera_far_clip,
            posinf=self.cfg.depth_camera_far_clip,
            neginf=self.cfg.depth_camera_near_clip,
        )
        depth = depth.clamp(self.cfg.depth_camera_near_clip, self.cfg.depth_camera_far_clip)
        self._update_depth_preview_ui(depth)
        depth = 1.0 - (depth - self.cfg.depth_camera_near_clip) / (
            self.cfg.depth_camera_far_clip - self.cfg.depth_camera_near_clip
        )
        self._latest_record_depth_frames = depth.clamp(0.0, 1.0)
        return self._latest_record_depth_frames

    def get_record_depth_frames(self) -> torch.Tensor:
        """Return normalized front-camera depth from the tiled camera only.

        This bypasses the human-facing viewport preview path so dataset export
        always uses the real `front_depth_camera` images.
        """
        if not self._built:
            self._build()
        self.sim.render()
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
        self._latest_record_depth_frames = depth.clamp(0.0, 1.0)
        return self._latest_record_depth_frames

    def get_record_depth_frames_u8(self) -> torch.Tensor:
        """Return uint8 front-camera depth frames for dataset writing."""
        if self._latest_record_depth_frames is None:
            self.get_record_depth_frames()
        return (self._latest_record_depth_frames * 255.0).to(torch.uint8)

    def get_follow_camera_depth_u8(self, robot_index: int) -> np.ndarray:
        """Capture a depth image from the dedicated first-person follow camera."""
        if not self._built:
            self._build()
        if self._viewport_follow_cam_path is None or self._viewport_depth_annotator is None:
            self._setup_first_person_view()
        if self._viewport_follow_cam_path is None or self._viewport_depth_annotator is None:
            raise RuntimeError("First-person follow camera is unavailable for depth capture.")

        self._viewport_follow_env_index = int(np.clip(robot_index, 0, self.num_envs - 1))
        self._update_follow_camera_pose()
        self.sim.render()
        depth_raw = self._viewport_depth_annotator.get_data()
        if depth_raw is None or getattr(depth_raw, "size", 0) == 0:
            raise RuntimeError("Follow camera depth annotator returned no data.")

        if depth_raw.ndim == 1:
            expected_size = self.cfg.depth_camera_height * self.cfg.depth_camera_width
            if depth_raw.size == expected_size:
                depth_np = depth_raw.reshape(self.cfg.depth_camera_height, self.cfg.depth_camera_width).copy()
            else:
                raise RuntimeError(
                    f"Unexpected follow camera depth shape {depth_raw.shape}; cannot reshape to "
                    f"{self.cfg.depth_camera_height}x{self.cfg.depth_camera_width}."
                )
        elif depth_raw.ndim == 3:
            depth_np = depth_raw[:, :, 0].copy()
        else:
            depth_np = depth_raw.copy()

        depth = torch.from_numpy(depth_np).float()
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
        return (depth.clamp(0.0, 1.0) * 255.0).to(torch.uint8).cpu().numpy()

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, extras = super().reset(seed=seed, options=options)
        self.get_depth_frames()
        return obs, extras

    def step(self, actions: torch.Tensor):
        obs, rew, terminated, truncated, extras = super().step(actions)
        self.get_depth_frames()
        return obs, rew, terminated, truncated, extras

    def render_depth(self, robot_index: int | None = None) -> np.ndarray:
        """Render a 3-channel visualization of one robot's latest depth image."""
        if robot_index is None:
            robot_index = self._record_robot_index
        depth_frame = self.get_depth_frames()[robot_index].detach().cpu().numpy()
        depth_rgb = np.clip(depth_frame * 255.0, 0.0, 255.0).astype(np.uint8)
        return np.repeat(depth_rgb[:, :, None], 3, axis=2)

    def close(self):
        if self._viewport_depth_annotator is not None and self._viewport_render_product is not None:
            try:
                self._viewport_depth_annotator.detach([self._viewport_render_product])
            except Exception:
                pass
        self._viewport_depth_annotator = None
        self._viewport_render_product = None
        self._depth_camera = None
        super().close()
