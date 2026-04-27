from __future__ import annotations

# 标准库：这里主要用来定义轻量配置类，以及处理本地路径导入。
from dataclasses import dataclass
import os
import sys

# Gym 用来定义观测空间；torch 用来做张量计算和加载网络权重。
import gymnasium as gym
import torch.nn as nn
import torch

# Omniverse / Isaac Lab 里的相机与仿真组件。这里直接内联深度相机逻辑，
# 不再依赖 quadcopter_obstacles_depth_eval_env.py。
import omni.kit.commands
import omni.usd
import isaaclab.sim as sim_utils
from isaaclab.sensors.camera import TiledCamera, TiledCameraCfg
# 把世界坐标系下的向量旋转到无人机机体系下。
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_mul
from pxr import Sdf, UsdGeom

# 这里手动把本地工程路径加入 sys.path，
# 这样即使这个文件被当作脚本入口启动，也能正常导入当前任务包和本地 rsl_rl。
TASK_DIR = os.path.abspath(os.path.dirname(__file__))
ENV_DIR = os.path.abspath(os.path.join(TASK_DIR, ".."))
LOCAL_RSL_RL_DIR = os.path.join(ENV_DIR, "rsl_rl")
for path in (ENV_DIR, LOCAL_RSL_RL_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

# 直接复用 teacher 任务环境的动力学、奖励和基础低维观测。
from quadcopter_obstacles_teacher.quadcopter_obstacles_env import QuadcopterObstaclesEnv, QuadcopterObstaclesEnvCfg


def _make_activation(name: str) -> nn.Module:
    """把 teacher checkpoint 里记录的激活函数名字映射成 torch 模块。"""
    activation_name = name.lower()
    if activation_name == "elu":
        return nn.ELU()
    if activation_name == "relu":
        return nn.ReLU()
    if activation_name == "tanh":
        return nn.Tanh()
    if activation_name == "leaky_relu":
        return nn.LeakyReLU()
    raise ValueError(f"Unsupported teacher activation: {name}")


def _build_actor_mlp(layer_dims: list[int], activation: str) -> nn.Sequential:
    """根据 checkpoint 中记录的层维度，重建 teacher 的 actor MLP。"""
    layers: list[nn.Module] = []
    for index, (in_dim, out_dim) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
        # 每一层都先加一个线性层。
        layers.append(nn.Linear(in_dim, out_dim))
        # 隐藏层后面加激活函数，最后输出层后面不加。
        if index < len(layer_dims) - 2:
            layers.append(_make_activation(activation))
    return nn.Sequential(*layers)


class FrozenTeacherActor(nn.Module):
    """冻结的 teacher actor，只在训练时生成蒸馏目标动作。"""

    def __init__(self, checkpoint_path: str, device: torch.device):
        super().__init__()
        # 先把 checkpoint 加载到 CPU，避免受当前 CUDA 设备状态影响。
        loaded = torch.load(checkpoint_path, weights_only=False, map_location="cpu")

        if loaded.get("export_format") == "frozen_teacher_actor_v1":
            # 快速路径：checkpoint 已经是单独导出的 actor。
            layer_dims = [int(dim) for dim in loaded["layer_dims"]]
            activation = str(loaded.get("activation", "elu"))
            self.actor = _build_actor_mlp(layer_dims, activation)
            self.actor.load_state_dict(loaded["actor_state_dict"])
            # 直接复用 teacher 的观测归一化统计量。
            obs_mean = loaded["obs_mean"].float()
            obs_std = loaded["obs_std"].float().clamp_min(1e-6)
        else:
            # 兼容路径：如果是完整 PPO checkpoint，就从里面把 actor 部分拆出来重建。
            state_dict = loaded["model_state_dict"]
            actor_weight_keys = sorted(
                [key for key in state_dict.keys() if key.startswith("actor.") and key.endswith(".weight")],
                key=lambda key: int(key.split(".")[1]),
            )
            layer_dims = [state_dict[actor_weight_keys[0]].shape[1]]
            for weight_key in actor_weight_keys:
                layer_dims.append(state_dict[weight_key].shape[0])
            # 老版本 teacher actor 默认使用 ELU。
            self.actor = _build_actor_mlp(layer_dims, "elu")

            linear_idx = 0
            for module in self.actor:
                if not isinstance(module, nn.Linear):
                    continue
                # 把 checkpoint 里的每一层线性层参数拷贝到重建后的 MLP 中。
                weight_key = actor_weight_keys[linear_idx]
                bias_key = weight_key.replace(".weight", ".bias")
                module.weight.data.copy_(state_dict[weight_key])
                module.bias.data.copy_(state_dict[bias_key])
                linear_idx += 1

            obs_mean = state_dict["actor_obs_normalizer._mean"].float()
            obs_std = state_dict["actor_obs_normalizer._std"].float().clamp_min(1e-6)

        # buffer 会跟着模型一起搬到设备上，但不会参与训练。
        self.register_buffer("obs_mean", obs_mean)
        self.register_buffer("obs_std", obs_std)
        # 把 teacher 放到目标设备，并切到 eval 模式。
        self.to(device)
        self.eval()
        for param in self.parameters():
            # 显式冻结，避免误参与梯度计算。
            param.requires_grad_(False)

    def act_inference(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """对输入观测做归一化后，前向计算 teacher 动作。"""
        policy_obs = obs["policy"]
        norm_obs = (policy_obs - self.obs_mean.to(policy_obs.device)) / self.obs_std.to(policy_obs.device)
        return self.actor(norm_obs)


@dataclass
class QuadcopterObstaclesRefinerEnvCfg(QuadcopterObstaclesEnvCfg):
    # 默认 teacher checkpoint。训练时用它生成蒸馏目标动作。
    teacher_checkpoint: str = (
        "/home/wei/End_to_end/logs/rsl_rl/quadcopter_obstacles_v5/2026-03-14_02-50-17/model_3000.pt"
    )
    # 下面这些深度相机参数原本来自 depth_eval_env，这里直接内联到当前文件。
    depth_camera_width: int = 90
    depth_camera_height: int = 60
    depth_camera_near_clip: float = 0.2
    depth_camera_far_clip: float = 8.0
    depth_camera_focal_length: float = 12.0
    depth_camera_horizontal_aperture: float = 20.955
    depth_sector_count: int = 90
    depth_slice_half_thickness: float = 0.35
    # 前向深度相机相对无人机 body 的外参。
    depth_camera_offset_pos: tuple[float, float, float] = (0.10, 0.0, 0.03)
    depth_camera_offset_rot: tuple[float, float, float, float] = (0.5, -0.5, 0.5, -0.5)
    depth_camera_offset_convention: str = "ros"
    # 是否加载 teacher 网络。
    enable_teacher: bool = True
    # 是否在 extras / 日志里输出 teacher 相关动作和 teacher-student 差异指标。
    enable_teacher_metrics: bool = True
    # 默认关闭第一人称视角窗口和深度预览窗口，保证训练更快。
    use_first_person_view: bool = False
    show_depth_preview: bool = False
    # 是否打印调试日志。
    debug_logs: bool = False

    def policy_observation_dim(self) -> int:
        # 这里只保留真正可部署的低维状态：
        # 机体系线速度、机体系角速度、重力投影、目标相对机体位置、上一时刻速度命令。
        return 3 + 3 + 3 + 3 + 3

    def critic_base_observation_dim(self) -> int:
        # critic 也需要看到与 actor 一致的基础运动状态和上一时刻速度命令。
        return 3 + 3 + 3 + 3 + 3

    def critic_privileged_observation_dim(self) -> int:
        # critic 训练时可额外看到的特权障碍信息：
        # 最近障碍物方向、最近障碍物距离、前向障碍距离、目标方向障碍距离、是否到达目标。
        return self.num_closest_obstacles * 3 + self.num_closest_obstacles + 1 + 1 + 1


class QuadcopterObstaclesRefinerEnv(QuadcopterObstaclesEnv):
    """基于深度图的 student 环境，teacher 仅用于训练时蒸馏指导。"""

    cfg: QuadcopterObstaclesRefinerEnvCfg

    # ==================== 基础工具与初始化 ====================

    def _debug(self, message: str) -> None:
        """小工具函数：根据配置决定是否打印调试信息。"""
        cfg = getattr(self, "cfg", None)
        if cfg is not None and getattr(cfg, "debug_logs", False):
            print(message, flush=True)

    def __init__(self, cfg: QuadcopterObstaclesRefinerEnvCfg | None = None, render_mode: str | None = None, **kwargs):
        # 如果外部没有传配置，就使用默认配置。
        cfg = cfg if cfg is not None else QuadcopterObstaclesRefinerEnvCfg()
        if getattr(cfg, "debug_logs", False):
            print("[DEBUG][refiner.env] __init__ start", flush=True)
        # 缓存当前仿真步的深度图，避免同一步里重复读取相机。
        self._cached_policy_image: torch.Tensor | None = None
        self._last_image_step: int = -1
        # 深度相机本体与最近一次深度图缓存。
        self._depth_camera: TiledCamera | None = None
        self._latest_record_depth_frames: torch.Tensor | None = None
        # 深度相机外参对应的张量缓存，后续同步相机位姿时会反复用到。
        self._depth_camera_offset_pos_b: torch.Tensor | None = None
        self._depth_camera_offset_rot_b: torch.Tensor | None = None
        # 深度投影缓存，避免每步重复构造像素网格和相机投影系数。
        self._depth_proj_x_scale: torch.Tensor | None = None
        self._depth_proj_y_scale: torch.Tensor | None = None
        self._depth_proj_hw: int = 0
        # 先把 teacher 相关成员准备好，后面父类初始化过程中也能安全访问。
        self._teacher_policy = None
        self._teacher_actions = None
        self._teacher_actions_valid = False
        # 调父类初始化，构建仿真、机器人、深度相机、奖励等基础环境。
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)
        self._debug("[DEBUG][refiner.env] super().__init__ done")
        # student 低维输入维度，只对应可部署的机载状态。
        self._policy_state_dim = self.cfg.policy_observation_dim()
        self._critic_base_state_dim = self.cfg.critic_base_observation_dim()
        self._critic_privileged_dim = self.cfg.critic_privileged_observation_dim()
        self._debug(f"[DEBUG][refiner.env] policy_state_dim={self._policy_state_dim}")
        # 重新定义 student 的观测空间：
        # actor 一路是可部署低维状态向量，一路是深度图；
        # critic 单独看到基础状态和特权障碍信息。
        self.single_observation_space = gym.spaces.Dict(
            {
                "policy": gym.spaces.Dict(
                    {
                        "policy_state": gym.spaces.Box(
                            low=-float("inf"),
                            high=float("inf"),
                            shape=(self._policy_state_dim,),
                            dtype=float,
                        ),
                        # 深度图在父类环境里已经被归一化到 [0, 1]。
                        "policy_scan": gym.spaces.Box(
                            low=0.0,
                            high=1.0,
                            shape=(1, self.cfg.depth_sector_count),
                            dtype=float,
                        ),
                    }
                ),
                "critic": gym.spaces.Dict(
                    {
                        "critic_base_state": gym.spaces.Box(
                            low=-float("inf"),
                            high=float("inf"),
                            shape=(self._critic_base_state_dim,),
                            dtype=float,
                        ),
                        "critic_privileged": gym.spaces.Box(
                            low=-float("inf"),
                            high=float("inf"),
                            shape=(self._critic_privileged_dim,),
                            dtype=float,
                        ),
                    }
                )
            }
        )
        self._debug("[DEBUG][refiner.env] single_observation_space built")
        # 构建 batched 后的向量化环境观测空间。
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space, self.num_envs)
        self.num_states = self._critic_base_state_dim + self._critic_privileged_dim
        self._debug("[DEBUG][refiner.env] observation_space batched")
        # 预先分配 teacher 动作缓存，避免每步重复分配内存。
        self._teacher_policy = None
        self._teacher_actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        # 这些 teacher 相关指标只用于训练时分析和 TensorBoard 记录。
        self._episode_sums["teacher_action_l2"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._episode_sums["teacher_student_l1"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._debug("[DEBUG][refiner.env] __init__ complete")

    @property
    def depth_camera(self) -> TiledCamera:
        """返回已经构建好的 tiled depth camera。"""
        if self._depth_camera is None:
            raise RuntimeError("Depth camera has not been built yet.")
        return self._depth_camera

    # ==================== 深度相机构建 ====================

    def _hide_robots_from_depth(self) -> None:
        """把机器人本体从深度相机的次级光线里隐藏，避免机体自身出现在深度图里。"""
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
        """创建每台无人机前向挂载的 tiled depth camera。"""
        camera_cfg = TiledCameraCfg(
            prim_path="/World/Robot_.*/OmniNxt/body/front_depth_camera",
            update_period=0.0,
            update_latest_camera_pose=True,
            height=self.cfg.depth_camera_height,
            width=self.cfg.depth_camera_width,
            data_types=["depth"],
            depth_clipping_behavior="max",
            offset=TiledCameraCfg.OffsetCfg(
                pos=self.cfg.depth_camera_offset_pos,
                rot=self.cfg.depth_camera_offset_rot,
                convention=self.cfg.depth_camera_offset_convention,
            ),
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=self.cfg.depth_camera_focal_length,
                focus_distance=4.0,
                horizontal_aperture=self.cfg.depth_camera_horizontal_aperture,
                clipping_range=(self.cfg.depth_camera_near_clip, self.cfg.depth_camera_far_clip),
            ),
        )
        self._depth_camera = TiledCamera(camera_cfg)

    def _build_depth_camera_offset_cache(self) -> None:
        """把配置里的深度相机外参缓存成张量，避免每次取图都重新构造。"""
        offset_pos = torch.tensor(self.cfg.depth_camera_offset_pos, device=self.device, dtype=torch.float32)
        offset_rot = torch.tensor(self.cfg.depth_camera_offset_rot, device=self.device, dtype=torch.float32)
        self._depth_camera_offset_pos_b = offset_pos.unsqueeze(0).repeat(self.num_envs, 1)
        self._depth_camera_offset_rot_b = offset_rot.unsqueeze(0).repeat(self.num_envs, 1)

    def _build_depth_projection_cache(self) -> None:
        """缓存像素到相机平面的投影系数，避免每步重复构造 meshgrid。"""
        width = self.cfg.depth_camera_width
        height = self.cfg.depth_camera_height
        fx = (self.cfg.depth_camera_focal_length / self.cfg.depth_camera_horizontal_aperture) * width
        fy = fx
        cx = (width - 1) * 0.5
        cy = (height - 1) * 0.5

        u = torch.arange(width, device=self.device, dtype=torch.float32)
        v = torch.arange(height, device=self.device, dtype=torch.float32)
        uu, vv = torch.meshgrid(u, v, indexing="xy")
        self._depth_proj_x_scale = ((uu - cx) / fx).reshape(1, -1)
        self._depth_proj_y_scale = ((vv - cy) / fy).reshape(1, -1)
        self._depth_proj_hw = int(height * width)

    def _get_body_index(self) -> int:
        """把环境里缓存的 body id 统一解析成可用于张量索引的整数。"""
        body_id = self._body_id
        if isinstance(body_id, torch.Tensor):
            return int(body_id.view(-1)[0].item())
        if isinstance(body_id, (list, tuple)):
            return int(body_id[0])
        return int(body_id)

    def _sync_depth_camera_to_body(self) -> None:
        """在每次取图前，把深度相机世界位姿强制对齐到无人机 body。"""
        body_index = self._get_body_index()
        body_pos_w = self.robot.data.body_pos_w[:, body_index]
        body_quat_w = self.robot.data.body_quat_w[:, body_index]
        if self._depth_camera_offset_pos_b is None or self._depth_camera_offset_rot_b is None:
            self._build_depth_camera_offset_cache()

        cam_pos_w = body_pos_w + quat_apply(body_quat_w, self._depth_camera_offset_pos_b)
        cam_quat_w = quat_mul(body_quat_w, self._depth_camera_offset_rot_b)
        self.depth_camera.set_world_poses(cam_pos_w, cam_quat_w, convention=self.cfg.depth_camera_offset_convention)

    # ==================== 环境构建与 teacher ====================

    def _build(self) -> None:
        self._debug("[DEBUG][refiner.env] _build start")
        if self._built:
            return

        # 这里直接内联 depth_eval_env 需要的 build 流程，
        # 不再依赖外部的 QuadcopterObstaclesDepthEvalEnv。
        sim_cfg = sim_utils.SimulationCfg(dt=self.cfg.physics_dt, device=str(self.device))
        self._sim = sim_utils.SimulationContext(sim_cfg)
        self._debug("[DEBUG][refiner.env] SimulationContext ready")
        self.sim.set_camera_view(
            eye=list(self.cfg.viewer_eye),
            target=list(self.cfg.viewer_lookat),
            camera_prim_path=self.cfg.viewer_cam_prim_path,
        )

        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_cfg.func("/World/Ground", ground_cfg)
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        self._debug("[DEBUG][refiner.env] ground/light ready")

        self._spawn_shared_obstacles()
        self._debug("[DEBUG][refiner.env] obstacles spawned")
        self._spawn_boundary_walls()
        self._debug("[DEBUG][refiner.env] walls spawned")
        self._build_robot_assets()
        self._debug("[DEBUG][refiner.env] robot assets built")
        self._disable_robot_collisions()
        self._debug("[DEBUG][refiner.env] robot collisions disabled")
        self._hide_robots_from_depth()
        self._debug("[DEBUG][refiner.env] robots hidden from depth")
        self._build_depth_cameras()
        self._debug("[DEBUG][refiner.env] depth cameras built")
        self._build_depth_camera_offset_cache()
        self._debug("[DEBUG][refiner.env] depth camera offset cache built")
        self._build_depth_projection_cache()
        self._debug("[DEBUG][refiner.env] depth projection cache built")

        self.sim.reset()
        self._debug("[DEBUG][refiner.env] sim.reset done")
        self._body_id = self.robot.find_bodies("body")[0]
        self._robot_mass = self.robot.root_physx_view.get_masses()[0].sum().item()
        self.robot.write_joint_state_to_sim(self.robot.data.default_joint_pos, self.robot.data.default_joint_vel)
        self.robot.update(self.cfg.physics_dt)
        for _ in range(5):
            self.sim.step()
            self.robot.update(self.cfg.physics_dt)
        self.depth_camera.update(self.cfg.physics_dt)
        self._debug("[DEBUG][refiner.env] depth camera update done")
        self._built = True

        # 只有当前模式真的需要 teacher 时，才加载 teacher 网络。
        if self.cfg.enable_teacher and self._teacher_policy is None:
            self._debug("[DEBUG][refiner.env] building teacher policy")
            self._build_teacher_policy()
            self._debug("[DEBUG][refiner.env] teacher policy ready")

    def _build_teacher_policy(self) -> None:
        # teacher actor 只加载一次，后续所有环境、所有时间步都复用。
        self._debug(f"[DEBUG][refiner.env] loading teacher checkpoint: {self.cfg.teacher_checkpoint}")
        self._teacher_policy = FrozenTeacherActor(self.cfg.teacher_checkpoint, self.device)
        self._debug("[DEBUG][refiner.env] teacher frozen actor ready")

    def _compute_teacher_actions(self, base_obs: dict[str, torch.Tensor]) -> torch.Tensor:
        # 只有显式开启 teacher 且 teacher 已构建好时，才允许查询 teacher 动作。
        if not self.cfg.enable_teacher or self._teacher_policy is None:
            raise RuntimeError("Teacher policy is not initialized.")
        with torch.inference_mode():
            # teacher 始终只做推理，不参与训练。
            return self._teacher_policy.act_inference(base_obs)

    def get_teacher_actions(self) -> torch.Tensor:
        """返回当前时间步缓存的 teacher 动作；如果缓存失效则重新计算。"""
        if self._teacher_actions_valid:
            return self._teacher_actions.clone()
        # teacher 需要的是 teacher 环境那套低维观测，而不是 student 的输入。
        base_obs = QuadcopterObstaclesEnv._get_observations(self)
        teacher_actions = self._compute_teacher_actions(base_obs)
        self._teacher_actions.copy_(teacher_actions)
        self._teacher_actions_valid = True
        return teacher_actions.clone()

    # ==================== student 观测构造 ====================

    def _get_policy_image(self) -> torch.Tensor:
        """读取当前时间步的扇区距离特征，并做一步缓存。"""
        if self._cached_policy_image is not None and self._last_image_step == self.common_step_counter:
            return self._cached_policy_image
        self._debug("[DEBUG][refiner.env] projecting depth to scan sectors")
        policy_scan = self._project_depth_to_scan().unsqueeze(1)
        self._cached_policy_image = policy_scan
        self._last_image_step = self.common_step_counter
        self._debug("[DEBUG][refiner.env] policy scan ready")
        return policy_scan

    def _get_depth_image_meters(self) -> torch.Tensor:
        """读取 tiled 相机的原始深度（单位米）。"""
        if not self._built:
            self._build()
        self._sync_depth_camera_to_body()
        # 与旧版 refiner 环境保持一致：先推进一帧渲染，再拉取最新深度传感器输出。
        self.sim.render()
        self.depth_camera.update(self.cfg.physics_dt)
        depth = self.depth_camera.data.output["depth"].squeeze(-1).to(self.device)
        depth = torch.nan_to_num(
            depth,
            nan=self.cfg.depth_camera_far_clip,
            posinf=self.cfg.depth_camera_far_clip,
            neginf=self.cfg.depth_camera_near_clip,
        )
        return depth.clamp(self.cfg.depth_camera_near_clip, self.cfg.depth_camera_far_clip)

    def _project_depth_to_scan(self) -> torch.Tensor:
        """将深度图投影到水平面，并聚合为扇区最近障碍距离。"""
        depth = self._get_depth_image_meters()
        batch = depth.shape[0]
        device = depth.device

        if self._depth_proj_x_scale is None or self._depth_proj_y_scale is None:
            self._build_depth_projection_cache()

        depth_flat = depth.reshape(batch, self._depth_proj_hw)
        x = self._depth_proj_x_scale.to(device=device, dtype=depth.dtype) * depth_flat
        y = self._depth_proj_y_scale.to(device=device, dtype=depth.dtype) * depth_flat
        points_c = torch.stack([x, y, depth_flat], dim=-1)

        cam_quat_b = self._depth_camera_offset_rot_b.to(device=device, dtype=depth.dtype)
        cam_pos_b = self._depth_camera_offset_pos_b.to(device=device, dtype=depth.dtype)
        points_b = quat_apply(
            cam_quat_b.unsqueeze(1).expand(-1, self._depth_proj_hw, -1).reshape(batch * self._depth_proj_hw, 4),
            points_c.reshape(batch * self._depth_proj_hw, 3),
        ).reshape(batch, self._depth_proj_hw, 3)
        points_b = points_b + cam_pos_b.unsqueeze(1)

        horizontal_mask = points_b[:, :, 0] > 0.0
        horizontal_mask &= points_b[:, :, 2].abs() <= self.cfg.depth_slice_half_thickness

        planar_dist = torch.linalg.norm(points_b[:, :, :2], dim=-1)
        angles = torch.atan2(points_b[:, :, 1], points_b[:, :, 0])
        # The front depth camera only observes a limited horizontal FOV. Map the 180 scan bins
        # to that real visible azimuth span instead of spreading them over a fake 360-degree ring.
        hfov = 2.0 * torch.atan(
            torch.tensor(
                self.cfg.depth_camera_horizontal_aperture / (2.0 * self.cfg.depth_camera_focal_length),
                device=device,
                dtype=depth.dtype,
            )
        )
        half_hfov = 0.5 * hfov
        fov_mask = (angles >= -half_hfov) & (angles <= half_hfov)
        valid_mask = horizontal_mask & fov_mask
        sector_size = hfov / self.cfg.depth_sector_count
        sector_idx = torch.floor((angles + half_hfov) / sector_size).long().clamp(0, self.cfg.depth_sector_count - 1)

        sector_distances = torch.full(
            (batch, self.cfg.depth_sector_count),
            self.cfg.depth_camera_far_clip,
            device=device,
            dtype=depth.dtype,
        )

        batch_idx = torch.arange(batch, device=device).unsqueeze(1).expand(-1, self._depth_proj_hw)
        flat_sector_idx = batch_idx * self.cfg.depth_sector_count + sector_idx
        sector_distances_flat = sector_distances.reshape(-1)
        sector_distances_flat.scatter_reduce_(
            0,
            flat_sector_idx[valid_mask],
            planar_dist[valid_mask],
            reduce="amin",
            include_self=False,
        )
        sector_distances = sector_distances_flat.reshape(batch, self.cfg.depth_sector_count)

        # 归一化到 [0, 1]，与旧版深度图输入保持一致：near -> 1, far -> 0。
        normalized = 1.0 - (sector_distances - self.cfg.depth_camera_near_clip) / (
            self.cfg.depth_camera_far_clip - self.cfg.depth_camera_near_clip
        )
        return normalized.clamp(0.0, 1.0)

    def get_record_depth_frames(self) -> torch.Tensor:
        """读取 tiled 相机的深度图，并归一化到 [0, 1]（用于可视化/调试）。"""
        depth = self._get_depth_image_meters()
        depth = 1.0 - (depth - self.cfg.depth_camera_near_clip) / (
            self.cfg.depth_camera_far_clip - self.cfg.depth_camera_near_clip
        )
        self._latest_record_depth_frames = depth.clamp(0.0, 1.0)
        return self._latest_record_depth_frames

    def get_record_depth_frames_u8(self) -> torch.Tensor:
        """返回 `uint8` 深度图，方便直接写成 PNG 做肉眼对比。"""
        depth_frames = self.get_record_depth_frames()
        return (depth_frames * 255.0).to(torch.uint8)

    def _get_student_policy_state(self) -> torch.Tensor:
        """构造 student 可部署的低维状态。

        这里故意不包含任何障碍物真值特征，
        这样训练后的 student 才能在真实无人机上，仅依赖机载状态估计和目标信息运行。
        """
        # 世界坐标系下的无人机位置和姿态。
        root_pos_w = self.robot.data.root_pos_w
        root_quat_w = self.robot.data.root_quat_w
        # 把线速度和角速度都转到机体系。
        root_lin_vel_b = quat_apply_inverse(root_quat_w, self.robot.data.root_lin_vel_w)
        root_ang_vel_b = quat_apply_inverse(root_quat_w, self.robot.data.root_ang_vel_w)
        # 把重力方向投影到机体系，作为一个紧凑的姿态表征。
        projected_gravity_b = quat_apply_inverse(root_quat_w, self._gravity_vec_w)
        # 目标相对无人机的位置，并表示到机体系下。
        target_vec_w = self._target_positions_w - root_pos_w
        target_pos_b = quat_apply_inverse(root_quat_w, target_vec_w)
        # 拼接成最终的 12 维低维状态向量。
        return torch.cat(
            [
                root_lin_vel_b,
                root_ang_vel_b,
                projected_gravity_b,
                target_pos_b,
            ],
            dim=-1,
        )

    def _get_observations(self) -> dict[str, torch.Tensor]:
        """构造 student policy 真正看到的观测字典。"""
        self._debug("[DEBUG][refiner.env] _get_observations start")
        # 先拿到 teacher 环境的基础低维观测；
        # 这部分只在需要 teacher 指导/teacher 指标时使用。
        base_obs = QuadcopterObstaclesEnv._get_observations(self)
        base_policy_obs = base_obs["policy"]
        self._debug("[DEBUG][refiner.env] base obs ready")
        if self.cfg.enable_teacher_metrics:
            # 先把当前步的 teacher 动作缓存起来，
            # 这样 trainer 记录蒸馏指标时不用重复算一次。
            teacher_actions = self._compute_teacher_actions(base_obs)
            self._debug("[DEBUG][refiner.env] teacher actions ready")
            self._teacher_actions.copy_(teacher_actions)
            self._teacher_actions_valid = True
        else:
            # 如果当前模式不需要 teacher 输出，就把 teacher 缓存标为失效。
            self._teacher_actions_valid = False
        # student 最终只看到“可部署低维状态 + 扇区距离特征”。
        policy_state = base_policy_obs[:, : self._policy_state_dim]
        policy_scan = self._get_policy_image()
        # teacher 低维观测的前 12 维刚好也是 critic 需要的基础运动状态。
        critic_base_state = base_policy_obs[:, : self._critic_base_state_dim]
        # teacher 原始观测布局：
        # [0:12] 基础状态, [12:15] 当前速度命令,
        # [15:15+3N] 障碍方向, [..+N] 障碍距离, [..+1] forward, [..+1] target_ray, [..+1] target_reached
        critic_privileged_start = 15
        critic_privileged_end = critic_privileged_start + self._critic_privileged_dim
        critic_privileged = base_policy_obs[:, critic_privileged_start:critic_privileged_end]
        self._debug("[DEBUG][refiner.env] _get_observations complete")
        return {
            "policy": {
                "policy_state": policy_state,
                "policy_scan": policy_scan,
            },
            "critic": {
                "critic_base_state": critic_base_state,
                "critic_privileged": critic_privileged,
            }
        }

    # ==================== 环境交互 ====================

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        # reset 时清空本时间步相关缓存，保证 reset 后第一帧观测重新计算。
        self._cached_policy_image = None
        self._last_image_step = -1
        self._teacher_actions_valid = False
        # 真正的 reset 逻辑仍然复用 teacher 任务环境。
        return QuadcopterObstaclesEnv.reset(self, seed=seed, options=options)

    def step(self, actions: torch.Tensor):
        """执行 student 动作，并在需要时附加 teacher 指标。"""
        if not self._built:
            raise RuntimeError("Call reset() before step().")

        # 进入新时间步前，把上一时刻的深度图缓存失效掉。
        self._cached_policy_image = None
        self._last_image_step = -1
        # student policy 输出的是归一化动作，范围限制在 [-1, 1]。
        student_actions = actions.to(self.device).clamp(-1.0, 1.0)
        # 只有当前模式需要 teacher 指标/蒸馏目标时，才去查询 teacher 动作。
        teacher_actions = self.get_teacher_actions() if self.cfg.enable_teacher_metrics else None

        # 关键点：真正执行的是 student 自己的动作，不再是 teacher + residual。
        obs, rew, terminated, truncated, extras = QuadcopterObstaclesEnv.step(self, student_actions)

        # 把 student 动作和最终执行动作暴露给外部日志/评估代码。
        extras["student_actions"] = student_actions.clone()
        extras["final_actions"] = student_actions.clone()
        if "log" not in extras:
            extras["log"] = {}
        if teacher_actions is not None:
            # 这些 teacher 指标只是诊断用，不会修改环境 reward。
            teacher_action_l2 = torch.sum(torch.square(teacher_actions), dim=1)
            teacher_student_l1 = torch.sum(torch.abs(teacher_actions - student_actions), dim=1)
            self._episode_sums["teacher_action_l2"] += teacher_action_l2 * self.step_dt
            self._episode_sums["teacher_student_l1"] += teacher_student_l1 * self.step_dt
            # 把 teacher 动作放到 extras 里，trainer 会在环境外部计算蒸馏 loss。
            extras["teacher_actions"] = teacher_actions.clone()
            extras["log"]["Metrics/teacher_action_l2"] = (
                self._episode_sums["teacher_action_l2"].mean().item() / self.max_episode_length_s
            )
            extras["log"]["Metrics/teacher_student_l1"] = (
                self._episode_sums["teacher_student_l1"].mean().item() / self.max_episode_length_s
            )
        return obs, rew, terminated, truncated, extras
