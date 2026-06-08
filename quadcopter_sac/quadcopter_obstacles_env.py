from __future__ import annotations # 启用延迟评估的类型注解（允许在类定义内部使用该类自身的类型提示）

from dataclasses import dataclass, field # 导入数据类装饰器和字段定义工具，用于配置管理

import gymnasium as gym # 导入 Gymnasium，标准的强化学习环境 API 库
import numpy as np # 导入 NumPy，用于 CPU 端的数值计算和数组操作
import torch # 导入 PyTorch，用于 GPU 加速的张量计算

# 导入 IsaacLab 的核心仿真模块和资产模块
import isaaclab.sim as sim_utils # IsaacLab 仿真工具箱，包含创建形状、灯光、物理属性等方法
import isaaclab.utils.math as math_utils # IsaacLab 的数学工具箱
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg # 导入 Articulation 和 RigidObject
from isaaclab.sensors.ray_caster import RayCaster, RayCasterCfg, patterns # 导入 IsaacLab 原生 Warp RayCaster 传感器
from isaaclab.sim import schemas as sim_schemas # 导入仿真模式描述，用于修改物理引擎（PhysX）底层属性
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg # 导入地形生成和导入的配置类
from isaaclab.utils.math import quat_apply, quat_apply_inverse # 用于四元数旋转：将世界系向量转换到机体系，或反之
import isaacsim.core.utils.prims as prim_utils
import terrain as navrl_terrain # 导入自定义的导航强化学习地形模块
from terrain_cfg import HfUniformDiscreteObstaclesTerrainCfg # 导入自定义的高度场离散障碍物地形配置

# 导入 donor 项目的无人机资产 (OmniNxt) 和。
from assets.omninxt.omninxt import OMNINXT_CFG # 导入无人机模型配置
from controller import CrazyflieController, config as controller_config # 导入底层飞行控制器及配置


def _quat_to_euler_deg(quat_w: torch.Tensor) -> torch.Tensor:
    """Convert quaternions in (w, x, y, z) format to Euler XYZ in degrees."""
    # 这里把四元数转换成欧拉角，主要是为了对接 CrazyflieController，
    # 因为控制器内部使用的是 roll / pitch / yaw 的角度表达。
    w, x, y, z = quat_w[:, 0], quat_w[:, 1], quat_w[:, 2], quat_w[:, 3] # 拆解批次化四元数张量的 w, x, y, z 分量

    sinr_cosp = 2.0 * (w * x + y * z) # 计算 Roll（横滚角）相关的三角函数分子
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y) # 计算 Roll 相关的三角函数分母
    roll = torch.atan2(sinr_cosp, cosr_cosp) # 利用 atan2 求出 Roll 弧度值

    sinp = 2.0 * (w * y - z * x) # 计算 Pitch（俯仰角）的 sin 值
    pitch = torch.asin(torch.clamp(sinp, -1.0, 1.0)) # 截断并用 asin 求出 Pitch 弧度值，防止数值溢出导致 NaN

    siny_cosp = 2.0 * (w * z + x * y) # 计算 Yaw（偏航角）相关的三角函数分子
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z) # 计算 Yaw 相关的三角函数分母
    yaw = torch.atan2(siny_cosp, cosy_cosp) # 利用 atan2 求出 Yaw 弧度值

    return torch.rad2deg(torch.stack([roll, pitch, yaw], dim=1)) # 将 roll, pitch, yaw 拼合成张量，并从弧度转换为角度返回


def _quat_to_yaw(quat_w: torch.Tensor) -> torch.Tensor:
    """Extract yaw in radians from quaternions in (w, x, y, z) format."""
    w, x, y, z = quat_w[:, 0], quat_w[:, 1], quat_w[:, 2], quat_w[:, 3] # 拆解四元数分量
    siny_cosp = 2.0 * (w * z + x * y) # 计算 Yaw 相关的三角函数分子
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z) # 计算 Yaw 相关的三角函数分母
    return torch.atan2(siny_cosp, cosy_cosp) # 直接返回弧度制的 Yaw 偏航角张量


def _vec_to_target_frame(vec_w: torch.Tensor, target_dir_w: torch.Tensor) -> torch.Tensor:
    """Project world-frame vectors into NavRL's target-direction frame."""
    target_xy = target_dir_w[..., :2]
    target_xy_norm = torch.linalg.norm(target_xy, dim=-1, keepdim=True).clamp_min(1.0e-6)
    x_axis_xy = target_xy / target_xy_norm
    y_axis_xy = torch.stack([-x_axis_xy[..., 1], x_axis_xy[..., 0]], dim=-1)
    x = torch.sum(vec_w[..., :2] * x_axis_xy, dim=-1, keepdim=True)
    y = torch.sum(vec_w[..., :2] * y_axis_xy, dim=-1, keepdim=True)
    z = vec_w[..., 2:3]
    return torch.cat([x, y, z], dim=-1)


@dataclass
class QuadcopterObstaclesEnvCfg:
    # 定义环境配置数据类，用于统一管理环境超参数
    # ==================== 基础仿真与并行环境配置 ====================
    @dataclass
    class SimCfg:
        device: str = "cuda:0" # 物理仿真默认运行在第 0 张 GPU 上

    @dataclass
    class SceneCfg:
        num_envs: int = 8 # 默认并行运行 8 个无人机环境

    episode_length_s: float = 30.0 # 单个回合 (episode) 的最大时长为 30 秒
    decimation: int = 1 # 动作降采样步数（RL每输出1次动作，物理引擎步进1次）
    physics_dt: float = 1.0 / 100.0 # 物理引擎步长，100Hz
    device: str = "cuda:0" # RL 张量计算默认设备
    is_finite_horizon: bool = False # 是否为有限时间视界 MDP
    seed: int | None = None # 随机种子

    # ==================== 地图配置 ====================
    map_half_extent: float = 20.0 # 地图边长的一半

    # ==================== 障碍物配置 ====================
    num_obstacles: int = 100 # 场景中总共生成多少个静态障碍物。
    lidar_hres: float = 10.0  # NavRL 3D LiDAR 水平角分辨率，单位度。
    lidar_vbeams: int = 4  # NavRL 3D LiDAR 垂直束数。
    lidar_vfov: tuple[float, float] = (0.0, 0.0)  # 始终只扫水平面，避免把地板当作障碍物。
    num_obstacle_rays: int = 144  # 36 个水平束 x 4 个垂直束。
    obstacle_radius: float = 0.3 # height-field 方柱障碍物的半宽，单位米。
    obstacle_height: float = 4.0  # 每个圆柱障碍物的高度，单位米。
    obstacle_spawn_range: float = 20.0  # 障碍物中心在 xy 平面上的生成范围半径，实际地图约为 [-20, 20] x [-20, 20]。
    obstacle_safe_zone: float = 1.0  # 场地中心保留的无障碍安全区半径，避免中间被障碍完全堵死。
    obstacle_min_separation: float = 3  # 不同障碍物中心之间的最小间距，防止圆柱互相贴得太近。
    obstacle_detection_range: float = 4.0  # NavRL LiDAR 最大探测距离；超过这个距离就按“足够远”处理。
    obstacle_proximity_trigger_distance: float = 2.0  # 开始触发障碍物接近惩罚的距离阈值。
    obstacle_collision_margin: float = 0.2  # 认为“已经撞上障碍物”的额外安全边界（防穿模裕度）。
    target_obstacle_clearance: float = 2.0  # 目标点采样时，与障碍物至少保持的净空距离。
    use_raycast_lidar: bool = True  # 是否启用 IsaacLab RayCaster；默认开启，优先使用和 NavRL 一致的静态检测方式。
    render_gradient_obstacles: bool = True  # 是否叠加彩色渐变障碍物外观；只影响渲染，不参与 RayCaster/碰撞。
    obstacle_visual_gradient_segments: int = 5  # 每根障碍物用多少段颜色叠出近似渐变效果。
    obstacle_visual_size_margin: float = 0.0  # 彩色可视化方柱相对真实方柱单边额外放大的尺寸，默认严格对齐真实检测尺寸。
    dyn_obs_num_obstacles: int = 80  # NavRL 动态障碍物数量；会按 4 个宽度档 x 2 个高度档取整。
    dyn_obs_num_observed: int = 5  # 策略观测最近的动态障碍物数量。
    dyn_obs_vel_range: tuple[float, float] = (0.5, 1.5)  # NavRL 动态障碍物速度范围。
    dyn_obs_local_range: tuple[float, float, float] = (5.0, 5.0, 4.5)  # 动态障碍物局部目标采样范围。
    dyn_obs_max_width: float = 1.0  # NavRL 动态障碍物最大宽度。
    dyn_obs_max_3d_height: float = 1.0  # 3D 浮动障碍物高度。
    dyn_obs_max_2d_height: float = 5.0  # 2D 高柱障碍物高度。
    dyn_obs_goal_threshold: float = 0.5  # 距离动态目标小于该值时重新采样目标。
    dyn_obs_velocity_resample_s: float = 2.0  # 约每 2 秒重采样一次速度。
    norm_max_dist: float = 50.0  # NavRL 状态/动态障碍物距离归一化尺度。
    norm_max_vel: float = 2.0  # NavRL 速度归一化尺度。
    
    # ==================== 起点 / 终点采样配置 ====================
    spawn_edge_distance: float = 24.0 # 出生点距离中心的边缘距离，对齐 NavRL
    target_spawn_range: float = 24.0 # 目标点生成的坐标范围，对齐 NavRL
    spawn_min_height: float = 0.5 # 出生最低高度 0.5 米，对齐 NavRL
    spawn_max_height: float = 2.5 # 出生最高高度 2.5 米
    target_min_height: float = 0.5 # 目标最低高度 0.5 米，对齐 NavRL
    target_max_height: float = 2.5 # 目标最高高度 2.5 米
    min_flight_height: float = 0.2 # 低空死亡阈值，对齐 NavRL
    max_flight_height: float = 4.0 # 高空死亡阈值，对齐 NavRL
    out_of_bounds_margin: float = 2.0 # 水平边界额外余量；允许从边缘出生/到达目标，但禁止继续飞远
    target_reach_threshold: float = 0.5 # 判定为“到达目标”的距离阈值，对齐 NavRL

    # ==================== 动作解释为速度指令的范围 ====================
    cmd_body_vel_xy_max: float = 2 # XY 平面最大水平速度指令
    cmd_vel_z_max: float = 0.5 # Z 轴最大垂直速度指令
    yaw_rate_scale: float = 3.141592653589793 # 偏航角速度观测缩放系数，默认按 pi rad/s 归一化

    # ==================== 奖励项权重 ====================
    # 当前任务里外圈安全区域太大，NavRL 原始量级会让“活着且远离障碍”压过“朝目标推进”。
    safety_static_reward_scale: float = 0.1 # 静态障碍安全奖励只做弱 shaping
    dynamic_safety_reward_scale: float = 0.1  # 动态障碍物安全奖励只做弱 shaping
    velocity_to_goal_reward_scale: float = 20.0 # 让朝目标运动成为主奖励
    progress_reward_scale: float = 30.0 # 每步距离目标变近的稠密奖励，解决“还没到目标就没有学习信号”的问题
    constant_reward: float = 0.0 # 去掉每步生存分，避免外围悬停/慢飞白拿奖励
    target_reach_reward: float = 100.0 # 到达目标的一次性成功奖励
    smoothness_penalty_scale: float = 0.1 # 速度突变惩罚权重
    height_penalty_scale: float = 2.0 # 只做高度正则，避免压过前进奖励
    height_penalty_margin: float = 0.2 # 起终点高度带的额外安全余量，对齐 NavRL

    # ==================== Gym / IsaacLab 通用配置 ====================
    observation_space: int = 0 # 观测空间维度（将在 __post_init__ 中自动计算）
    action_space: int = 3 # 动作空间维度： vx, vy, vz
    state_space: int = 0 # 特权状态空间维度（此处未使用）
    auto_reset_done: bool = True # 环境终止后是否自动重置
    debug_vis: bool = True # 是否开启调试可视化
    debug_lidar_rays: bool = True # debug_vis 打开时，是否手动画出 LiDAR/RayCaster 射线。
    debug_lidar_env_count: int = 16 # 默认绘制前 16 个无人机的雷达射线。
    debug_lidar_ray_size: float = 1.0 # 雷达调试线宽。
    viewer_eye: tuple[float, float, float] = (-60.0, 0.0, 30.0) # 渲染器相机位置
    viewer_lookat: tuple[float, float, float] = (0.0, 0.0, 0.0) # 渲染器相机朝向
    viewer_cam_prim_path: str = "/OmniverseKit_Persp" # Omniverse 默认透视相机路径
    viewer_resolution: tuple[int, int] = (1920, 1080) # RGB 阵列渲染分辨率
    reward_debug_interval: int = 200 # 每隔多少步打印一次奖励调试信息
    sim: SimCfg = field(default_factory=SimCfg) # 仿真配置实例
    scene: SceneCfg = field(default_factory=SceneCfg) # 场景配置实例

    def policy_observation_dim(self) -> int:
        # 动态计算策略网络的输入维度
        # NavRL 风格：state(8) + 最近动态障碍物(dyn_obs_num_observed * 10) + 3D LiDAR。
        return 8 + self.dyn_obs_num_observed * 10 + self.num_obstacle_rays

    def navrl_state_dim(self) -> int:
        return 8 + self.dyn_obs_num_observed * 10

    def lidar_hbeams(self) -> int:
        return int(round(360.0 / max(float(self.lidar_hres), 1.0e-6)))

    def lidar_shape(self) -> tuple[int, int]:
        return self.lidar_hbeams(), int(self.lidar_vbeams)

    def target_distance_observation_scale(self) -> float:
        # 使用边缘到边缘的最大可能平面距离作为目标距离缩放尺度，避免 target_dist 以几十米的原始量级进入网络。
        edge_extent = max(
            float(self.spawn_edge_distance),
            float(self.target_spawn_range),
            float(self.map_half_extent),
            float(self.obstacle_spawn_range),
            1.0,
        )
        return 2.0 * edge_extent * (2.0 ** 0.5)

    def altitude_observation_scale(self) -> float:
        # 高度按允许飞行上界缩放，保留超过 1 的数值以便 adapter 仍能识别 too_high。
        return max(float(self.max_flight_height), 1e-6)

    def yaw_rate_observation_scale(self) -> float:
        # 偏航角速度使用可配置缩放系数归一化；超过该范围时允许观测值超过 1，
        # 这样既统一量纲，又不丢失高速旋转信息。
        return max(float(self.yaw_rate_scale), 1e-6)

    def target_velocity_reward_scale_value(self) -> float:
        # 目标方向速度奖励用水平最大指令速度作为主尺度，避免直接使用 m/s 原值。
        return max(float(self.cmd_body_vel_xy_max), 1e-6)

    def __post_init__(self):
        # dataclass 初始化完成后自动调用的钩子函数
        # 构造完成后，统一把环境数和 device 对齐，并自动写回 observation 维度。
        self.scene.num_envs = int(self.scene.num_envs) # 确保环境数为整数
        self.sim.device = self.device # 确保 Sim 的设备与外部设定一致
        self.num_obstacle_rays = self.lidar_hbeams() * int(self.lidar_vbeams)
        self.observation_space = self.policy_observation_dim() # 调用函数计算并赋值真实的观测维度


class QuadcopterObstaclesEnv(gym.Env):
    """Single-world multi-drone teacher environment with privileged obstacle truth."""
    # 这是一个自定义的 Gymnasium 环境，用于多无人机在同一物理世界中并行训练（Teacher 环境提供特权信息）

    metadata = {"render_modes": ["rgb_array"], "render_fps": 100} # 定义环境支持的渲染模式和帧率 (100Hz / decimation 1 = 100 FPS)

    def __init__(self, cfg: QuadcopterObstaclesEnvCfg | None = None, render_mode: str | None = None, **kwargs):
        del kwargs # 忽略多余参数
        # 没有传配置时使用默认配置。
        self.cfg = cfg if cfg is not None else QuadcopterObstaclesEnvCfg()
        self.render_mode = render_mode # 保存渲染模式设定
        
        # 再次确保配置类型安全
        if hasattr(self.cfg, "scene"):
            self.cfg.scene.num_envs = int(self.cfg.scene.num_envs)
        if hasattr(self.cfg, "sim"):
            self.cfg.device = self.cfg.sim.device

        self.device = torch.device(self.cfg.device) # 创建 PyTorch Device 对象
        self.num_envs = self.cfg.scene.num_envs # 提取并行环境数量
        
        # 一个 RL step 等于若干个 physics step，因此真正用于奖励和时长计算的是 step_dt。
        self.step_dt = self.cfg.physics_dt * self.cfg.decimation # 强化学习策略计算的时间间隔（例如 0.01 * 1 = 0.01秒）
        self.max_episode_length = int(round(self.cfg.episode_length_s / self.step_dt)) # 计算单回合最大步数（例如 60.0 / 0.01 = 6000步）
        self.max_episode_length_s = self.max_episode_length * self.step_dt # 真实的最大回合秒数（确保无浮点误差）
        self.num_states = 0 # 状态空间数（特权状态）
        self.common_step_counter = 0 # 全局步数计数器

        # 定义单智能体的动作空间为连续空间 [-1.0, 1.0] 的 Box
        self.single_action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.cfg.action_space,), dtype=float)
        # 根据 num_envs 将单动作空间扩展为批次化的动作空间（用于并行训练框架如 SB3 或 rl_games）
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)
        
        # teacher 环境只提供一组低维特权观测，因此这里的 observation_space 是一维向量字典。
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
        # 将单智能体观测空间扩展为批次化
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space, self.num_envs)

        # 内部状态变量初始化
        self._built = False # 标识物理环境是否已在 Omniverse 中构建
        self._sim: sim_utils.SimulationContext | None = None # IsaacLab 仿真上下文句柄
        self._terrain = None # 地形生成器实例句柄
        self._robot: Articulation | None = None # 无人机 Articulation 对象（管理多刚体关节组）
        self._lidar: RayCaster | None = None # IsaacLab 原生 Warp RayCaster，用于静态障碍物 360 度测距
        self._dynamic_obstacles: list[RigidObject] = []
        self._body_id: torch.Tensor | None = None # 无人机 body 刚体的物理 ID
        self._rgb_annotator = None # Isaac Sim 图像渲染标注器
        self._render_product = None # Isaac Sim 渲染产物
        self._debug_draw = None # Isaac debug_draw 接口，用于手动画 LiDAR 射线
        self._debug_draw_failed = False # debug_draw 获取失败后不再反复尝试，避免刷屏

        self._robot_mass = 0.0 # 无人机物理质量

        # 张量缓冲区初始化（提前在 GPU 上开辟内存，避免 step 时重复分配）
        # 动作会被解释成机体系速度命令，再交给原始 CrazyflieController 变成力 / 力矩。
        self._cmd_vel_b = torch.zeros((self.num_envs, 3), device=self.device) # 转化后的世界系目标速度 (vx, vy, vz)
        self._thrust = torch.zeros((self.num_envs, 1, 3), device=self.device) # 施加到重心的推力 (Num_envs, 1个刚体, XYZ)
        self._moment = torch.zeros((self.num_envs, 1, 3), device=self.device) # 施加到重心的力矩 (Num_envs, 1个刚体, Roll Pitch Yaw)
        
        # 初始化外部的串级 PID 飞行控制器，在 GPU 端并行运算
        self._controller = CrazyflieController(
            num_envs=self.num_envs,
            device=self.device,
            attitude_dt=self.step_dt,
            position_dt=self.step_dt,
        )

        self._target_positions_w = torch.zeros((self.num_envs, 3), device=self.device) # 各自的世界系目标点坐标
        # 上一时刻到目标的距离，用于构造 progress reward (势能差奖励)。
        self._prev_dist_to_target = torch.zeros(self.num_envs, device=self.device) 
        self._prev_drone_vel_w = torch.zeros((self.num_envs, 3), device=self.device) # 上一时刻世界系线速度，用于平滑惩罚
        self._height_range = torch.zeros((self.num_envs, 2), device=self.device) # 当前 episode 的目标高度带 [min_z, max_z]
        
        # 缓存归一化 lidar 距离，避免 reset 时调用 _lidar.update() 导致 BVH 重建卡死。
        # reset 时设为 1.0（最大探测范围 = 无障碍），下一次 step() 中 lidar.update() 后刷新为真实值。
        self._cached_lidar_distances_norm = torch.ones(
            (self.num_envs, self.cfg.num_obstacle_rays), device=self.device
        )

        self._dyn_obs_category_num = 8
        self._dyn_obs_num_total = (
            int(self.cfg.dyn_obs_num_obstacles) // self._dyn_obs_category_num * self._dyn_obs_category_num
        )
        self._dyn_obs_num_each_category = (
            self._dyn_obs_num_total // self._dyn_obs_category_num if self._dyn_obs_num_total > 0 else 0
        )
        self._dyn_obs_width_res = float(self.cfg.dyn_obs_max_width) / 4.0
        self._dyn_obs_state = torch.zeros((self._dyn_obs_num_total, 13), dtype=torch.float, device=self.device)
        if self._dyn_obs_num_total > 0:
            self._dyn_obs_state[:, 3] = 1.0
        self._dyn_obs_goal = torch.zeros((self._dyn_obs_num_total, 3), dtype=torch.float, device=self.device)
        self._dyn_obs_origin = torch.zeros((self._dyn_obs_num_total, 3), dtype=torch.float, device=self.device)
        self._dyn_obs_vel = torch.zeros((self._dyn_obs_num_total, 3), dtype=torch.float, device=self.device)
        self._dyn_obs_size = torch.zeros((self._dyn_obs_num_total, 3), dtype=torch.float, device=self.device)
        self._dyn_obs_step_count = 0

        self._obstacle_positions_w = torch.zeros((self.cfg.num_obstacles, 3), device=self.device) # 障碍物世界系位置
        default_obstacle_height = float(self.cfg.obstacle_height)
        default_obstacle_radius = float(self.cfg.obstacle_radius)
        self._obstacle_heights = torch.full((self.cfg.num_obstacles,), default_obstacle_height, device=self.device) # 每个障碍物高度
        self._obstacle_radii = torch.full((self.cfg.num_obstacles,), default_obstacle_radius, device=self.device) # 每个障碍物半径
        default_half_extent = torch.full((self.cfg.num_obstacles,), default_obstacle_radius, device=self.device)
        self._obstacle_half_widths = default_half_extent.clone() # 宽度的一半（对圆柱即半径）
        self._obstacle_half_lengths = default_half_extent.clone() # 长度的一半（对圆柱即半径）
        self._obstacle_positions_xy = self._obstacle_positions_w[:, :2] # 障碍物位置的 XY 视图引用
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device) # 记录每个环境当前步数
        self._rew_buf = torch.zeros(self.num_envs, device=self.device) # 每步奖励缓冲区
        self._done_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device) # 每步结束/截断标志位
        self._episode_success_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device) # 记录当前 episode 是否曾经到达目标
        
        self._episode_sums = {
            # 这些值用于 TensorBoard 里按 episode 统计不同奖励项和诊断指标。
            # 所有这些缓存都在每步进行累加，当一个 episode 结束时，计算平均值输出到日志，然后清零。
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "safety_static",
                "safety_dynamic",
                "velocity_to_goal",
                "progress",
                "constant",
                "target_reach",
                "smoothness",
                "height",
            ]
        }
        self._shared_obstacle_layout: list[dict] | None = None # 保存布局生成器生成的障碍物布局字典

    @property
    def unwrapped(self) -> QuadcopterObstaclesEnv:
        return self # 标准 Gym API，返回解包后的原始环境对象

    @property
    def sim(self) -> sim_utils.SimulationContext:
        if self._sim is None:
            raise RuntimeError("Environment not built yet.") # 如果仿真未初始化则报错
        return self._sim # 返回仿真上下文

    @property
    def robot(self) -> Articulation:
        if self._robot is None:
            raise RuntimeError("Environment not built yet.")
        return self._robot # 返回无人机多刚体集合的句柄

    def seed(self, seed: int = -1) -> int:
        # 尽量把 torch 和 cuda 的随机种子一起固定，便于复现实验。
        if seed >= 0:
            torch.manual_seed(seed) # 设置 PyTorch CPU 种子
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed) # 设置所有 GPU 种子
            self.cfg.seed = seed
        return seed

    def _build(self) -> None:
        # 懒构建：只有第一次 reset / step 前才真正创建仿真世界，避免导包时直接唤起高开销渲染引擎。
        if self._built:
            return
        print("[BUILD] starting SimulationContext...", flush=True)

        # 创建 IsaacLab 仿真上下文。
        sim_cfg = sim_utils.SimulationCfg(dt=self.cfg.physics_dt, device=str(self.device))
        self._sim = sim_utils.SimulationContext(sim_cfg) # 实例化物理仿真器
        # 设置默认视角的渲染相机位置
        self.sim.set_camera_view(
            eye=list(self.cfg.viewer_eye),
            target=list(self.cfg.viewer_lookat),
            camera_prim_path=self.cfg.viewer_cam_prim_path,
        )

        # TerrainImporter 直接把静态障碍物烘焙进单个地形 mesh，和 NavRL 的处理方式一致。
        terrain_seed = 0 if self.cfg.seed is None else int(self.cfg.seed)
        # 调用外部启发式算法，均匀撒点生成互不重叠的障碍物位置
        self._shared_obstacle_layout = navrl_terrain.sample_uniform_obstacle_layout(
            seed=terrain_seed,
            num_obstacles=self.cfg.num_obstacles,
            map_size=(self.cfg.obstacle_spawn_range * 2.0, self.cfg.obstacle_spawn_range * 2.0),
            obstacle_width_range=(2.0 * self.cfg.obstacle_radius, 2.0 * self.cfg.obstacle_radius),
            obstacle_height_range=(self.cfg.obstacle_height, self.cfg.obstacle_height),
            obstacles_distance=self.cfg.obstacle_min_separation + self.cfg.obstacle_safe_zone,
            avoid_positions=[[0.0, 0.0]], # 确保原点安全（原点可能用于生成什么）
        )

        # 配置由地形生成器导入的大地形底座
        terrain_cfg = TerrainImporterCfg(
            prim_path="/World/obstacleTerrain", # 地形的 USD 层级路径；和 isaac-go2 示例保持一致
            num_envs=1, # 全局地形，所以这里设为 1 即可
            terrain_type="generator",
            terrain_generator=TerrainGeneratorCfg(
                seed=terrain_seed,
                curriculum=False, # 不使用课程学习改变地形
                size=(self.cfg.obstacle_spawn_range * 2.0, self.cfg.obstacle_spawn_range * 2.0),
                border_width=5.0,
                num_rows=1,
                num_cols=1,
                horizontal_scale=0.1, # 地形横向网格精度
                vertical_scale=0.1, # 地形高度精度
                slope_threshold=0.75,
                use_cache=False,
                color_scheme="height", # 按高度赋色
                sub_terrains={
                    "obstacles": HfUniformDiscreteObstaclesTerrainCfg(
                        proportion=1.0,
                        seed=terrain_seed,
                        obstacle_width_range=(2.0 * self.cfg.obstacle_radius, 2.0 * self.cfg.obstacle_radius),
                        obstacle_height_range=(self.cfg.obstacle_height, self.cfg.obstacle_height),
                        num_obstacles=len(self._shared_obstacle_layout), # 将共享障碍布局烘焙进静态地形 mesh，供 RayCaster 做 NavRL 风格检测
                        obstacles_distance=self.cfg.obstacle_min_separation + self.cfg.obstacle_safe_zone,
                        platform_width=0.0,
                        avoid_positions=[[0.0, 0.0]],
                        obstacle_layout=self._shared_obstacle_layout,
                    ),
                },
            ),
            visual_material=None,
            collision_group=-1, # 碰撞组标记
            debug_vis=True,
        )
        print("[BUILD] creating terrain importer...", flush=True)
        self._terrain = terrain_cfg.class_type(terrain_cfg) # 实例化地形导入器
        self._apply_grid_terrain_visual() # 给 terrain mesh 绑定网格材质，只改外观，不改变 RayCaster 检测目标

        # 创建灯光。整体使用偏冷的深色场景光，接近 NavRL 示例里的暗色网格地图。
        key_light_cfg = sim_utils.DistantLightCfg(intensity=2900.0, color=(0.78, 0.82, 0.90)) # 主方向光
        key_light_cfg.func("/World/Light", key_light_cfg)
        sky_light_cfg = sim_utils.DomeLightCfg(intensity=950.0, color=(0.08, 0.10, 0.16)) # 穹顶天光，补齐暗部环境光
        sky_light_cfg.func("/World/SkyLight", sky_light_cfg)

        self._initialize_obstacle_layout() # 初始化张量层面的障碍物位置
        self._spawn_visual_obstacle_overlays() # 叠加无碰撞的彩色渐变障碍物，只改变渲染效果
        self._build_dynamic_obstacles()
        print("[BUILD] spawning robot assets...", flush=True)
        self._build_robot_assets() # 实例化所有无人机对象
        if self.cfg.use_raycast_lidar:
            print("[BUILD] building raycast lidar...", flush=True)
            self._build_lidar_sensor() # 可选：构建和 NavRL 一样的基于 RayCaster 的静态障碍检测传感器
        
        # 禁用无人机之间的碰撞。强化学习让无人机避障是通过“障碍物惩罚”来学习的，
        # 如果不禁用无人机互撞，在早期探索(exploration)阶段会因为频繁互撞导致样本效率极低。
        self._disable_robot_collisions()

        # reset 之后再读取 body id / mass 等信息，此时物理对象已经真正进入仿真。
        print("[BUILD] calling sim.reset()...", flush=True)
        self.sim.reset() # 初始化物理引擎的核心状态
        if self._lidar is not None and (not self._lidar.is_initialized):
            self._lidar._initialize_impl()
        if self._lidar is not None:
            # 关闭 RayCaster 官方的命中点 marker，可避免 viewport 里出现白色拖影。
            # 调试时统一只保留下面 debug_draw 画出的完整射线。
            self._lidar.set_debug_vis(False)
        try:
            self._body_id = self.robot.find_bodies("body")[0] # 查找 OmniNxt 主体刚体 body 的标识 ID 列表，取第一个
        except (IndexError, RuntimeError, ValueError):
            self._body_id = torch.tensor([0], device=self.device) # 后备方案：取索引 0
            
        self._robot_mass = self.robot.root_physx_view.get_masses()[0].sum().item()  # 从底层 physx API 汇总整架无人机质量。
        self.robot.write_joint_state_to_sim(self.robot.data.default_joint_pos, self.robot.data.default_joint_vel)  # 把默认关节状态（旋翼角度）写入仿真。
        self.robot.update(self.cfg.physics_dt)  # 刷新一次本地张量缓存，同步物理引擎最新数据。
        print("[BUILD] warmup stepping physics...", flush=True)
        for _ in range(5):
            # 先空跑几步，让 articulation 缓冲和物理状态稳定下来（例如让悬挂件自然下垂等）。
            self.sim.step()
            self.robot.update(self.cfg.physics_dt)
        if self._lidar is not None and self._lidar.is_initialized:
            self._lidar.reset()
            self._lidar.update(dt=self.step_dt, force_recompute=True)
            self._draw_lidar_debug_rays()

        # 打印初始化确认日志
        print(
            f"[INFO] Quadcopter Obstacles Teacher - Single World with {self.num_envs} robots and "
            f"{self.cfg.num_obstacles} terrain-baked static obstacles and {self._dyn_obs_num_total} dynamic obstacles"
        )
        print(f"[INFO] Observation space: {self.cfg.observation_space}")
        print(f"[INFO] OmniNxt mass from asset: {self._robot_mass:.4f} kg")
        print(f"[INFO] Controller configured mass: {controller_config.CF_MASS:.4f} kg")
        # 交叉验证配置的无人机质量和 USD 文件读取的质量是否匹配，影响 PID 参数
        if abs(self._robot_mass - controller_config.CF_MASS) > 0.05:
            print("[WARN] OmniNxt mass and controller CF_MASS differ significantly. Retune controller/config.py.")

        self._built = True  # 标记环境已经构建完成，避免二次 _build()。

    def _sample_edge_positions(
        self, num_samples: int, lateral_range: float, min_height: float, max_height: float
    ) -> torch.Tensor:
        # 在地图四条边上采样出生点/目标点：
        # 两条水平边 y = ±spawn_edge_distance，另外两条垂直边 x = ±spawn_edge_distance。
        side_indices = torch.randint(0, 4, (num_samples,), device=self.device)  # 随机决定每个点来自 4 条边中的哪一条。
        side_signs = torch.where(
            side_indices % 2 == 0,
            torch.ones(num_samples, device=self.device),  # 0 和 2 对应正方向 (+edge)。
            -torch.ones(num_samples, device=self.device),  # 1 和 3 对应负方向 (-edge)。
        )
        lateral = torch.empty(num_samples, device=self.device).uniform_(-lateral_range, lateral_range)  # 沿边方向的横向偏移均匀采样。
        heights = torch.empty(num_samples, device=self.device).uniform_(min_height, max_height)  # 高度在 [min, max] 内均匀采样。

        positions = torch.zeros((num_samples, 3), device=self.device)  # 分配内存存储输出的 xyz 坐标。
        x_side_mask = side_indices < 2  # 前两条边 (0, 1) 表示上下边界：y 轴固定，x 轴为 lateral 偏移。
        y_side_mask = ~x_side_mask  # 后两条边 (2, 3) 表示左右边界：x 轴固定，y 轴为 lateral 偏移。
        
        # 填充上下边界的点
        positions[x_side_mask, 0] = lateral[x_side_mask]  # 上下边时写入 x。
        positions[x_side_mask, 1] = side_signs[x_side_mask] * self.cfg.spawn_edge_distance  # 上下边时 y 固定为 ±edge。
        
        # 填充左右边界的点
        positions[y_side_mask, 0] = side_signs[y_side_mask] * self.cfg.spawn_edge_distance  # 左右边时 x 固定为 ±edge。
        positions[y_side_mask, 1] = lateral[y_side_mask]  # 左右边时写入 y。
        
        positions[:, 2] = heights  # 全部 z 维写入随机采样的局度。
        return positions  # 返回形状为 (num_samples, 3) 的边缘位置张量。

    def _edge_positions_from_sides(
        self,
        side_indices: torch.Tensor,
        edge_distance: float,
        lateral_range: float,
        min_height: float,
        max_height: float,
    ) -> torch.Tensor:
        num_samples = int(side_indices.numel())
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
        positions[x_side_mask, 1] = side_signs[x_side_mask] * edge_distance
        positions[y_side_mask, 0] = side_signs[y_side_mask] * edge_distance
        positions[y_side_mask, 1] = lateral[y_side_mask]
        positions[:, 2] = heights
        return positions

    def _sample_navigation_position_pairs(self, num_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
        # 起点仍从四条边界采样；目标先放在地图中心附近，强制任务方向指向障碍区域内部。
        start_sides = torch.randint(0, 4, (num_samples,), device=self.device)
        start_pos = self._edge_positions_from_sides(
            start_sides,
            self.cfg.spawn_edge_distance,
            self.cfg.spawn_edge_distance,
            self.cfg.spawn_min_height,
            self.cfg.spawn_max_height,
        )
        target_pos = torch.zeros((num_samples, 3), device=self.device)
        target_pos[:, :2] = torch.empty((num_samples, 2), device=self.device).uniform_(-1.0, 1.0)
        target_pos[:, 2] = torch.empty(num_samples, device=self.device).uniform_(
            self.cfg.target_min_height,
            self.cfg.target_max_height,
        )
        return start_pos, target_pos

    def _sample_obstacle_positions(self, obstacle_indices: torch.Tensor) -> torch.Tensor:
        # 从通过 terrain 生成器预生成的布局中，读取各个障碍物的位置和物理尺寸，装填进 GPU 张量中。
        num_obstacles = int(obstacle_indices.numel())
        positions = torch.zeros((self.cfg.num_obstacles, 3), device=self.device)
        
        # 异常情况：如果传入索引为空，直接把所有障碍物移到很远的地方 (spawn_range + 100) 隐藏起来。
        if num_obstacles == 0:
            positions[:, 0] = self.cfg.obstacle_spawn_range + 100.0 + torch.arange(self.cfg.num_obstacles, device=self.device)
            positions[:, 2] = self._obstacle_heights * 0.5 # 柱子原点位于中心，所以 Z 轴为其高度一半。
            return positions

        layout = self._shared_obstacle_layout # 获取刚才由 navrl_terrain 模块生成的字典列表。

        # 默认先把所有柱塞到远端隐藏。
        positions[:, 0] = self.cfg.obstacle_spawn_range + 100.0 + torch.arange(self.cfg.num_obstacles, device=self.device)
        positions[:, 2] = self._obstacle_heights * 0.5
        
        # 遍历生成的布局，将位置真正写入张量
        for local_idx, obstacle_idx in enumerate(obstacle_indices.tolist()):
            if local_idx >= len(layout): # 如果超过了预生成的障碍物数量则跳出
                break
            obstacle = layout[local_idx] # 获取字典
            width = float(obstacle["width"])
            length = float(obstacle["length"])
            height = float(obstacle["height"])
            
            # 使用和地形生成（terrain.py）完全相同的离散化网格逻辑计算真实的物理中心和尺寸。
            scale = 0.1 # TerrainGeneratorCfg.horizontal_scale
            vertical_scale = 0.1 # TerrainGeneratorCfg.vertical_scale
            map_half = self.cfg.obstacle_spawn_range # cfg.size[0] / 2
            width_pixels_map = int((2.0 * self.cfg.obstacle_spawn_range) / scale)
            length_pixels_map = int((2.0 * self.cfg.obstacle_spawn_range) / scale)
            
            width_pixels = int(width / scale)
            length_pixels = int(length / scale)
            height_pixels = int(height / vertical_scale)
            
            x_center = float(obstacle["x"]) + map_half
            y_center = float(obstacle["y"]) + map_half
            
            x_start = int((x_center - 0.5 * width) / scale)
            y_start = int((y_center - 0.5 * length) / scale)
            x_start = max(0, min(x_start, width_pixels_map - width_pixels))
            y_start = max(0, min(y_start, length_pixels_map - length_pixels))
            
            phys_x_center = (x_start + width_pixels / 2.0) * scale - map_half
            phys_y_center = (y_start + length_pixels / 2.0) * scale - map_half
            phys_height = height_pixels * vertical_scale
            
            # 更新该障碍物的真实物理半径、半宽、半长、高度缓冲变量
            self._obstacle_half_widths[obstacle_idx] = 0.5 * (width_pixels * scale)
            self._obstacle_half_lengths[obstacle_idx] = 0.5 * (length_pixels * scale)
            self._obstacle_radii[obstacle_idx] = max(self._obstacle_half_widths[obstacle_idx], self._obstacle_half_lengths[obstacle_idx])
            self._obstacle_heights[obstacle_idx] = phys_height
            
            # 填入 XYZ 坐标，采用与物理网格完全对齐的几何中心
            positions[obstacle_idx, 0] = phys_x_center
            positions[obstacle_idx, 1] = phys_y_center
            positions[obstacle_idx, 2] = 0.5 * phys_height
        return positions

    def _initialize_obstacle_layout(self) -> None:
        # 封装的布局初始化函数
        obstacle_indices = torch.arange(self.cfg.num_obstacles, device=self.device, dtype=torch.long)
        # 更新障碍物张量坐标
        self._obstacle_positions_w[:] = self._sample_obstacle_positions(obstacle_indices)

    def _apply_grid_terrain_visual(self) -> None:
        # RayCaster 使用的 baked terrain 是高度场方柱；为了避免方柱黑边露出来，这里把它视觉透明化。
        # 额外生成一张纯视觉网格地面用于观看/录制，RayCaster 仍然只打 /World/obstacleTerrain。
        visual_ground_path = "/World/VisualGridFloor"
        ground_cfg = sim_utils.GroundPlaneCfg(
            size=(
                self.cfg.obstacle_spawn_range * 2.0 + 10.0,
                self.cfg.obstacle_spawn_range * 2.0 + 10.0,
            ),
            color=(0.055, 0.075, 0.105),
            physics_material=None,
        )
        ground_cfg.func(visual_ground_path, ground_cfg, translation=(0.0, 0.0, 0.0))

        invisible_material_path = "/World/InvisibleObstacleTerrainMaterial"
        invisible_material_cfg = sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.0, 0.0, 0.0),
            opacity=0.0,
        )
        invisible_material_cfg.func(invisible_material_path, invisible_material_cfg)
        sim_utils.bind_visual_material("/World/obstacleTerrain/terrain", invisible_material_path)

    def _lerp_rgb(self, start: tuple[float, float, float], end: tuple[float, float, float], t: float) -> tuple[float, float, float]:
        # 线性插值 RGB，用多段柱体模拟第二张图里从暗部到亮部的渐变柱。
        return tuple(float(start[i] + (end[i] - start[i]) * t) for i in range(3))

    def _visual_obstacle_color(self, obstacle_idx: int, segment_idx: int, num_segments: int) -> tuple[float, float, float]:
        del obstacle_idx
        # 统一成参考图里的紫红竖向渐变：底部深紫，上部逐渐变红。
        start = (0.16, 0.10, 0.42)
        end = (0.92, 0.14, 0.16)
        denom = max(num_segments - 1, 1)
        t = float(segment_idx) / float(denom)
        return self._lerp_rgb(start, end, t)

    def _spawn_visual_obstacle_overlays(self) -> None:
        # 物理/雷达仍然使用 terrain baked mesh；这里额外放一层无碰撞方柱外观，和 RayCaster 检测到的方柱形状保持一致。
        if not self.cfg.render_gradient_obstacles:
            return

        num_segments = max(int(self.cfg.obstacle_visual_gradient_segments), 1)
        for idx in range(self.cfg.num_obstacles):
            x = float(self._obstacle_positions_w[idx, 0].item())
            y = float(self._obstacle_positions_w[idx, 1].item())
            height = float(self._obstacle_heights[idx].item())
            if height <= 0.0:
                continue

            size_x = float((2.0 * self._obstacle_half_widths[idx]).item()) + 2.0 * self.cfg.obstacle_visual_size_margin
            size_y = float((2.0 * self._obstacle_half_lengths[idx]).item()) + 2.0 * self.cfg.obstacle_visual_size_margin
            segment_height = height / float(num_segments)

            for seg_idx in range(num_segments):
                z = segment_height * (seg_idx + 0.5)
                color = self._visual_obstacle_color(idx, seg_idx, num_segments)
                material = sim_utils.PreviewSurfaceCfg(diffuse_color=color, metallic=0.05)
                prim_path = f"/World/VisualObstacles/Obstacle_{idx:03d}/Segment_{seg_idx:02d}"

                obstacle_cfg = sim_utils.CuboidCfg(
                    size=(size_x, size_y, segment_height),
                    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                    visual_material=material,
                )
                obstacle_cfg.func(prim_path, obstacle_cfg, translation=(x, y, z))

    def _build_dynamic_obstacles(self) -> None:
        # 对齐 NavRL：动态障碍物按 4 个宽度档 x 2 个高度档生成。
        if self._dyn_obs_num_total <= 0:
            return

        def construct_origin_regex(start: int, end: int) -> str:
            return "(" + "|".join(f"{idx:03d}" for idx in range(start, end)) + ")"

        num_width_bins = 4
        cuboid_category_num = num_width_bins
        cylinder_category_num = num_width_bins
        map_range = torch.tensor(
            [self.cfg.map_half_extent, self.cfg.map_half_extent, self.cfg.dyn_obs_local_range[2]],
            device=self.device,
            dtype=torch.float,
        )

        obs_dist = 2.0 * np.sqrt(
            float(self.cfg.map_half_extent) * float(self.cfg.map_half_extent) / float(self._dyn_obs_num_total)
        )
        curr_obs_dist = obs_dist
        prev_pos_list: list[np.ndarray] = []

        def check_pos_validity(prev_pos: list[np.ndarray], curr_pos: np.ndarray, adjusted_dist: float) -> bool:
            for pos in prev_pos:
                if np.linalg.norm(curr_pos - pos) <= adjusted_dist:
                    return False
            return True

        for category_idx in range(cuboid_category_num + cylinder_category_num):
            for origin_idx in range(self._dyn_obs_num_each_category):
                dyn_idx = origin_idx + category_idx * self._dyn_obs_num_each_category
                attempts = 0
                while True:
                    ox = np.random.uniform(low=-self.cfg.map_half_extent, high=self.cfg.map_half_extent)
                    oy = np.random.uniform(low=-self.cfg.map_half_extent, high=self.cfg.map_half_extent)
                    if category_idx < cuboid_category_num:
                        oz = np.random.uniform(low=0.0, high=float(map_range[2].item()))
                    else:
                        oz = 0.5 * float(self.cfg.dyn_obs_max_2d_height)
                    curr_pos = np.array([ox, oy])
                    attempts += 1
                    if check_pos_validity(prev_pos_list, curr_pos, curr_obs_dist) or attempts > 100:
                        prev_pos_list.append(curr_pos)
                        break
                    if attempts % 25 == 0:
                        curr_obs_dist *= 0.8
                curr_obs_dist = obs_dist
                origin = torch.tensor([ox, oy, oz], dtype=torch.float, device=self.device)
                self._dyn_obs_origin[dyn_idx] = origin
                self._dyn_obs_state[dyn_idx, :3] = origin
                prim_utils.create_prim(f"/World/DynamicObstacleOrigin_{dyn_idx:03d}", "Xform", translation=tuple(origin.tolist()))

            if category_idx < cuboid_category_num:
                obs_width = float(category_idx + 1) * float(self.cfg.dyn_obs_max_width) / float(num_width_bins)
                obs_height = float(self.cfg.dyn_obs_max_3d_height)
                spawn_cfg = sim_utils.CuboidCfg(
                    size=(obs_width, obs_width, obs_height),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
                )
                prim_name = "Cuboid"
            else:
                obs_width = float(category_idx - cuboid_category_num + 1) * float(self.cfg.dyn_obs_max_width) / float(num_width_bins)
                obs_height = float(self.cfg.dyn_obs_max_2d_height)
                spawn_cfg = sim_utils.CylinderCfg(
                    radius=0.5 * obs_width,
                    height=obs_height,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
                )
                prim_name = "Cylinder"

            start = category_idx * self._dyn_obs_num_each_category
            end = (category_idx + 1) * self._dyn_obs_num_each_category
            self._dyn_obs_size[start:end] = torch.tensor([obs_width, obs_width, obs_height], dtype=torch.float, device=self.device)
            origin_regex = construct_origin_regex(start, end)
            obstacle_cfg = RigidObjectCfg(
                prim_path=f"/World/DynamicObstacleOrigin_{origin_regex}/{prim_name}",
                spawn=spawn_cfg,
                init_state=RigidObjectCfg.InitialStateCfg(),
            )
            self._dynamic_obstacles.append(RigidObject(cfg=obstacle_cfg))

    def _move_dynamic_obstacles(self) -> None:
        if self._dyn_obs_num_total <= 0:
            return
        if self._dyn_obs_step_count == 0:
            dyn_obs_goal_dist = torch.zeros(self._dyn_obs_num_total, device=self.device)
        else:
            dyn_obs_goal_dist = torch.linalg.norm(self._dyn_obs_state[:, :3] - self._dyn_obs_goal, dim=1)
        new_goal_mask = dyn_obs_goal_dist < float(self.cfg.dyn_obs_goal_threshold)
        num_new_goal = int(torch.count_nonzero(new_goal_mask).item())
        if num_new_goal > 0:
            local_range = torch.tensor(self.cfg.dyn_obs_local_range, dtype=torch.float, device=self.device)
            sample_goal_local = -local_range + 2.0 * local_range * torch.rand((num_new_goal, 3), dtype=torch.float, device=self.device)
            self._dyn_obs_goal[new_goal_mask] = self._dyn_obs_origin[new_goal_mask] + sample_goal_local
            self._dyn_obs_goal[:, 0].clamp_(-self.cfg.map_half_extent, self.cfg.map_half_extent)
            self._dyn_obs_goal[:, 1].clamp_(-self.cfg.map_half_extent, self.cfg.map_half_extent)
            self._dyn_obs_goal[:, 2].clamp_(0.0, self.cfg.dyn_obs_local_range[2])
            self._dyn_obs_goal[self._dyn_obs_num_total // 2 :, 2] = 0.5 * float(self.cfg.dyn_obs_max_2d_height)

        resample_steps = max(int(round(float(self.cfg.dyn_obs_velocity_resample_s) / self.step_dt)), 1)
        if self._dyn_obs_step_count % resample_steps == 0:
            vel_min, vel_max = self.cfg.dyn_obs_vel_range
            vel_norm = float(vel_min) + (float(vel_max) - float(vel_min)) * torch.rand(
                (self._dyn_obs_num_total, 1), dtype=torch.float, device=self.device
            )
            direction = self._dyn_obs_goal - self._dyn_obs_state[:, :3]
            self._dyn_obs_vel[:] = vel_norm * direction / torch.linalg.norm(direction, dim=1, keepdim=True).clamp_min(1.0e-6)

        self._dyn_obs_state[:, :3] += self._dyn_obs_vel * self.step_dt
        for category_idx, obstacle in enumerate(self._dynamic_obstacles):
            start = category_idx * self._dyn_obs_num_each_category
            end = (category_idx + 1) * self._dyn_obs_num_each_category
            obstacle.write_root_state_to_sim(self._dyn_obs_state[start:end])
            obstacle.write_data_to_sim()
            obstacle.update(self.step_dt)
        self._dyn_obs_step_count += 1

    def _build_robot_assets(self) -> None:
        # 按边缘采样的初始位置生成每一架无人机的实例（基于预设 OMNINXT_CFG）。
        robot_spawn_cfg = OMNINXT_CFG.replace(prim_path="/World/Robot_.*/OmniNxt") # 使用正则表达式匹配批量生成时的路径
        initial_spawn = self._sample_edge_positions(
            self.num_envs,
            self.cfg.spawn_edge_distance,
            self.cfg.spawn_min_height,
            self.cfg.spawn_max_height,
        )
        for idx in range(self.num_envs):
            prim_path = f"/World/Robot_{idx:02d}/OmniNxt" # 每个环境对应一台无人机
            robot_spawn_cfg.spawn.func(prim_path, robot_spawn_cfg.spawn, translation=tuple(initial_spawn[idx].tolist()))  # 生成该无人机实例。

        # 这里再创建一个 Articulation 视图对象，用于后续统一高并发地读取和写入所有无人机状态（张量化批处理核心）。
        robot_view_cfg = robot_spawn_cfg.copy()  # 复制一份配置给 Articulation 视图使用。
        robot_view_cfg.spawn = None  # 视图对象不再负责 spawn（孵化），避免重复创建渲染对象。
        self._robot = Articulation(robot_view_cfg)  # 构建统一管理所有无人机的批处理视图控制器。

    def _build_lidar_sensor(self) -> None:
        # 使用 IsaacLab 原生 RayCaster 构造 NavRL 风格 3D Bpearl LiDAR：
        # 36 个水平束 x 4 个垂直束，输入网络前展平成一维，再在 CNN 内恢复为 (36, 4)。
        vfov = (
            max(-89.0, float(self.cfg.lidar_vfov[0])),
            min(89.0, float(self.cfg.lidar_vfov[1])),
        )
        ray_caster_cfg = RayCasterCfg(
            prim_path="/World/Robot_.*/OmniNxt",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
            mesh_prim_paths=["/World/obstacleTerrain"],
            pattern_cfg=patterns.BpearlPatternCfg(
                horizontal_res=float(self.cfg.lidar_hres),
                vertical_ray_angles=torch.linspace(
                    vfov[0],
                    vfov[1],
                    int(self.cfg.lidar_vbeams),
                ),
            ),
            ray_alignment="yaw",
            max_distance=self.cfg.obstacle_detection_range,
            debug_vis=self.cfg.debug_vis,
        )
        self._lidar = ray_caster_cfg.class_type(ray_caster_cfg)

    def _compute_ray_cylinder_distance(
        self,
        ray_origin_w: torch.Tensor,
        ray_dir_w: torch.Tensor,
    ) -> torch.Tensor:
        # 解析几何回退版本：
        # 对每条水平射线和所有圆柱障碍做相交测试，返回最近交点距离。
        rel_origin = ray_origin_w.unsqueeze(1) - self._obstacle_positions_w.unsqueeze(0)
        dx = ray_dir_w[:, 0].unsqueeze(1)
        dy = ray_dir_w[:, 1].unsqueeze(1)
        ox = rel_origin[..., 0]
        oy = rel_origin[..., 1]
        radius = self._obstacle_radii.unsqueeze(0)

        a = dx * dx + dy * dy
        b = 2.0 * (ox * dx + oy * dy)
        c = ox * ox + oy * oy - radius * radius
        disc = b * b - 4.0 * a * c
        valid = disc >= 0.0
        sqrt_disc = torch.sqrt(torch.clamp(disc, min=0.0))
        denom = 2.0 * a.clamp_min(1e-8)
        t0 = (-b - sqrt_disc) / denom
        t1 = (-b + sqrt_disc) / denom

        t_candidates = torch.where(
            t0 > 0.0,
            t0,
            torch.where(t1 > 0.0, t1, torch.full_like(t0, float("inf"))),
        )

        hit_z = ray_origin_w[:, 2].unsqueeze(1) + t_candidates * ray_dir_w[:, 2].unsqueeze(1)
        half_h = 0.5 * self._obstacle_heights.unsqueeze(0)
        center_z = self._obstacle_positions_w[:, 2].unsqueeze(0)
        z_min = center_z - half_h
        z_max = center_z + half_h
        z_valid = (hit_z >= z_min) & (hit_z <= z_max)

        t_candidates = torch.where(valid & z_valid, t_candidates, torch.full_like(t_candidates, float("inf")))
        min_dist = t_candidates.min(dim=1).values
        return torch.where(
            torch.isfinite(min_dist),
            min_dist,
            torch.full_like(min_dist, self.cfg.obstacle_detection_range),
        )

    def _disable_robot_collisions(self) -> None:
        # 关闭不同无人机之间以及无人机与环境之间的碰撞体，
        # 让任务把“障碍物风险”更多地交给 reward/termination 逻辑来定义，且避免死锁和探索低效。
        collision_cfg = sim_utils.CollisionPropertiesCfg(collision_enabled=False)  # 定义“禁用碰撞”的配置选项。
        for idx in range(self.num_envs):
            sim_schemas.modify_collision_properties(f"/World/Robot_{idx:02d}/OmniNxt", collision_cfg) # 覆写刚体碰撞属性

    def _refresh_lidar_cache(self) -> None:
        """Read fresh ray distances from the RayCaster and write into the cache."""
        if self._lidar is None or not self._lidar.is_initialized:
            return
        ray_origins_w = self._lidar.data.pos_w.unsqueeze(1)
        ray_hits_w = self._lidar.data.ray_hits_w
        ray_distances = torch.linalg.norm(ray_hits_w - ray_origins_w, dim=-1)
        ray_distances = torch.where(
            torch.isfinite(ray_distances),
            ray_distances,
            torch.full_like(ray_distances, self.cfg.obstacle_detection_range),
        )
        if ray_distances.shape[1] < self.cfg.num_obstacle_rays:
            pad_width = self.cfg.num_obstacle_rays - ray_distances.shape[1]
            pad = torch.full(
                (ray_distances.shape[0], pad_width),
                self.cfg.obstacle_detection_range,
                device=ray_distances.device,
                dtype=ray_distances.dtype,
            )
            ray_distances = torch.cat([ray_distances, pad], dim=1)
        elif ray_distances.shape[1] > self.cfg.num_obstacle_rays:
            ray_distances = ray_distances[:, : self.cfg.num_obstacle_rays]
        self._cached_lidar_distances_norm[:] = (
            ray_distances.clamp_max(self.cfg.obstacle_detection_range) / self.cfg.obstacle_detection_range
        ).clamp(0.0, 1.0)

    def _compute_front_ray_distances(self, drone_pos_w: torch.Tensor, drone_quat_w: torch.Tensor) -> torch.Tensor:
        # 优先使用缓存的 RayCaster 数据（在 step() 中通过 _refresh_lidar_cache 更新）。
        if self._lidar is not None and self._lidar.is_initialized:
            return self._cached_lidar_distances_norm.clone()

        drone_yaw = _quat_to_yaw(drone_quat_w)
        hbeams, vbeams = self.cfg.lidar_shape()
        horizontal_angles = (
            torch.arange(hbeams, device=self.device, dtype=torch.float32)
            * (2.0 * torch.pi / float(hbeams))
            - torch.pi
        )
        vfov = (
            max(-89.0, float(self.cfg.lidar_vfov[0])),
            min(89.0, float(self.cfg.lidar_vfov[1])),
        )
        vertical_angles = torch.deg2rad(
            torch.linspace(vfov[0], vfov[1], vbeams, device=self.device, dtype=torch.float32)
        )
        world_angles = drone_yaw.unsqueeze(1).unsqueeze(2) + horizontal_angles.view(1, hbeams, 1)
        vertical_angles = vertical_angles.view(1, 1, vbeams)
        ray_dirs_w = torch.stack(
            [
                torch.cos(vertical_angles) * torch.cos(world_angles),
                torch.cos(vertical_angles) * torch.sin(world_angles),
                torch.sin(vertical_angles).expand(world_angles.shape[0], world_angles.shape[1], vbeams),
            ],
            dim=-1,
        ).reshape(self.num_envs, self.cfg.num_obstacle_rays, 3)
        ray_origins = drone_pos_w.unsqueeze(1).expand(-1, self.cfg.num_obstacle_rays, -1)
        ray_distances = self._compute_ray_cylinder_distance(
            ray_origins.reshape(-1, 3),
            ray_dirs_w.reshape(-1, 3),
        ).view(self.num_envs, self.cfg.num_obstacle_rays)
        return (
            ray_distances.clamp_max(self.cfg.obstacle_detection_range) / self.cfg.obstacle_detection_range
        ).clamp(0.0, 1.0)

    def _get_debug_draw_interface(self):
        # RayCaster 自带 debug_vis 在部分 IsaacLab/viewport 组合里不画线，所以这里用 debug_draw 手动画。
        if self._debug_draw is not None or self._debug_draw_failed:
            return self._debug_draw
        try:
            from isaacsim.util.debug_draw import _debug_draw

            self._debug_draw = _debug_draw.acquire_debug_draw_interface()
        except Exception:
            self._debug_draw_failed = True
            self._debug_draw = None
        return self._debug_draw

    def _get_lidar_world_rays(self, env_count: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        # 复现 IsaacLab RayCaster 的世界系起点/方向，保证调试线和真实传感器完全一致。
        if self._lidar is None or not self._lidar.is_initialized:
            return None

        sensor_pos_w = self._lidar.data.pos_w[:env_count]
        sensor_quat_w = self._lidar.data.quat_w[:env_count]
        ray_starts = self._lidar.ray_starts[:env_count]
        ray_directions = self._lidar.ray_directions[:env_count]
        num_rays = ray_starts.shape[1]

        if self._lidar.cfg.ray_alignment == "world":
            ray_starts_w = ray_starts + sensor_pos_w.unsqueeze(1)
            ray_directions_w = ray_directions
        elif self._lidar.cfg.ray_alignment == "yaw":
            ray_starts_w = math_utils.quat_apply_yaw(
                sensor_quat_w.unsqueeze(1).expand(-1, num_rays, -1).reshape(-1, 4),
                ray_starts.reshape(-1, 3),
            ).view(env_count, num_rays, 3)
            ray_starts_w = ray_starts_w + sensor_pos_w.unsqueeze(1)
            ray_directions_w = math_utils.quat_apply_yaw(
                sensor_quat_w.unsqueeze(1).expand(-1, num_rays, -1).reshape(-1, 4),
                ray_directions.reshape(-1, 3),
            ).view(env_count, num_rays, 3)
        elif self._lidar.cfg.ray_alignment == "base":
            ray_starts_w = quat_apply(
                sensor_quat_w.unsqueeze(1).expand(-1, num_rays, -1).reshape(-1, 4),
                ray_starts.reshape(-1, 3),
            ).view(env_count, num_rays, 3)
            ray_starts_w = ray_starts_w + sensor_pos_w.unsqueeze(1)
            ray_directions_w = quat_apply(
                sensor_quat_w.unsqueeze(1).expand(-1, num_rays, -1).reshape(-1, 4),
                ray_directions.reshape(-1, 3),
            ).view(env_count, num_rays, 3)
        else:
            return None
        return ray_starts_w, ray_directions_w

    def _draw_lidar_debug_rays(self) -> None:
        # 只在非 headless 调试时绘制，避免正式训练额外开销。
        if not (self.cfg.debug_vis and self.cfg.debug_lidar_rays):
            return
        draw_interface = self._get_debug_draw_interface()
        if draw_interface is None:
            return

        env_count = min(max(int(self.cfg.debug_lidar_env_count), 1), self.num_envs)
        if self._lidar is not None and self._lidar.is_initialized:
            # 仿照 NavRL，只基于 RayCaster 的真实命中点画线，不补“最大量程空射线”。
            # 这样调试图和真实检测严格一致，也能避免空射线与命中射线叠在一起造成“穿模/补一根”的错觉。
            sensor_pos_w = self._lidar.data.pos_w[:env_count]
            ray_hits_w = self._lidar.data.ray_hits_w[:env_count]
            hit_is_finite = torch.isfinite(ray_hits_w).all(dim=-1)

            if torch.any(hit_is_finite):
                ray_origins_w = sensor_pos_w.unsqueeze(1).expand_as(ray_hits_w)[hit_is_finite]
                ray_endpoints_w = ray_hits_w[hit_is_finite]
            else:
                ray_origins_w = torch.empty((0, 3), device=self.device)
                ray_endpoints_w = torch.empty((0, 3), device=self.device)
        else:
            # 没有 RayCaster 时才退回解析几何重建，避免调试线和真实传感器不一致。
            root_pos_w = self.robot.data.root_pos_w[:env_count]
            root_quat_w = self.robot.data.root_quat_w[:env_count]
            ray_distances = self._compute_front_ray_distances(
                self.robot.data.root_pos_w,
                self.robot.data.root_quat_w,
            )[:env_count]
            ray_distances = ray_distances * self.cfg.obstacle_detection_range
            drone_yaw = _quat_to_yaw(root_quat_w)
            hbeams, vbeams = self.cfg.lidar_shape()
            horizontal_angles = (
                torch.arange(hbeams, device=self.device, dtype=torch.float32)
                * (2.0 * torch.pi / float(hbeams))
                - torch.pi
            )
            vfov = (
                max(-89.0, float(self.cfg.lidar_vfov[0])),
                min(89.0, float(self.cfg.lidar_vfov[1])),
            )
            vertical_angles = torch.deg2rad(
                torch.linspace(vfov[0], vfov[1], vbeams, device=self.device, dtype=torch.float32)
            )
            world_angles = drone_yaw.unsqueeze(1).unsqueeze(2) + horizontal_angles.view(1, hbeams, 1)
            vertical_angles = vertical_angles.view(1, 1, vbeams)
            ray_dirs_w = torch.stack(
                [
                    torch.cos(vertical_angles) * torch.cos(world_angles),
                    torch.cos(vertical_angles) * torch.sin(world_angles),
                    torch.sin(vertical_angles).expand(world_angles.shape[0], world_angles.shape[1], vbeams),
                ],
                dim=-1,
            ).reshape(env_count, self.cfg.num_obstacle_rays, 3)
            ray_origins_w = root_pos_w.unsqueeze(1).expand(-1, self.cfg.num_obstacle_rays, -1)
            ray_endpoints_w = ray_origins_w + ray_dirs_w * ray_distances.unsqueeze(-1)

        origins = ray_origins_w.reshape(-1, 3).detach().cpu().tolist()
        endpoints = ray_endpoints_w.reshape(-1, 3).detach().cpu().tolist()
        colors = [(0.2, 0.9, 1.0, 1.0)] * len(origins)
        sizes = [float(self.cfg.debug_lidar_ray_size)] * len(origins)
        try:
            draw_interface.clear_lines()
            draw_interface.draw_lines(origins, endpoints, colors, sizes)
        except Exception:
            pass

    def _update_prev_target_distance(self, env_ids: torch.Tensor) -> None:
        # 每次 reset 场景或 step 结束后，都维护更新一次“上一帧到目标的直线距离”。
        # 强化学习 progress reward (势能奖励) 依赖该差值。
        distances = torch.linalg.norm(self._target_positions_w[env_ids] - self.robot.data.root_pos_w[env_ids], dim=1)
        self._prev_dist_to_target[env_ids] = distances

    def _compute_dynamic_obstacle_features(
        self, root_pos_w: torch.Tensor, target_dir_w: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 对齐 NavRL：选最近的 K 个动态障碍物，转换到目标方向坐标系并做归一化。
        dyn_obs_num = int(self.cfg.dyn_obs_num_observed)
        if self._dyn_obs_num_total <= 0 or dyn_obs_num <= 0:
            features = torch.zeros((self.num_envs, dyn_obs_num * 10), device=self.device)
            collision = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            reward_dist = torch.full((self.num_envs, dyn_obs_num), self.cfg.obstacle_detection_range, device=self.device)
            return features, collision, reward_dist

        dyn_pos = self._dyn_obs_state[:, :3].unsqueeze(0).expand(self.num_envs, -1, -1)
        dyn_rpos = dyn_pos - root_pos_w.unsqueeze(1)
        dyn_rpos_for_distance = dyn_rpos.clone()
        dyn_rpos_for_distance[:, self._dyn_obs_num_total // 2 :, 2] = 0.0

        dyn_distance_2d = torch.linalg.norm(dyn_rpos_for_distance[..., :2], dim=2)
        k = min(dyn_obs_num, self._dyn_obs_num_total)
        _, closest_idx = torch.topk(dyn_distance_2d, k, dim=1, largest=False)
        range_mask = dyn_distance_2d.gather(1, closest_idx) > self.cfg.obstacle_detection_range

        gather_idx = closest_idx.unsqueeze(-1).expand(-1, -1, 3)
        closest_rpos = torch.gather(dyn_rpos_for_distance, 1, gather_idx)
        closest_rpos_g = _vec_to_target_frame(closest_rpos, target_dir_w.unsqueeze(1).expand(-1, k, -1))
        closest_rpos_g[range_mask] = 0.0

        closest_distance = torch.linalg.norm(closest_rpos, dim=-1, keepdim=True)
        closest_distance_2d = torch.linalg.norm(closest_rpos_g[..., :2], dim=-1, keepdim=True) / self.cfg.norm_max_dist
        closest_distance_z = closest_rpos_g[..., 2:3] / self.cfg.norm_max_dist
        closest_rpos_gn = closest_rpos_g / closest_distance.clamp_min(1.0e-6)

        closest_vel = self._dyn_obs_vel[closest_idx]
        closest_vel[range_mask] = 0.0
        closest_vel_g = _vec_to_target_frame(closest_vel, target_dir_w.unsqueeze(1).expand(-1, k, -1)) / self.cfg.norm_max_vel

        closest_size = self._dyn_obs_size[closest_idx]
        closest_width = closest_size[..., 0:1]
        closest_width_category = closest_width / self._dyn_obs_width_res - 1.0
        closest_width_category[range_mask] = 0.0
        closest_height = closest_size[..., 2:3]
        closest_height_category = torch.where(
            closest_height > self.cfg.dyn_obs_max_3d_height,
            torch.zeros_like(closest_height),
            closest_height,
        )
        closest_height_category[range_mask] = 0.0

        dyn_features = torch.cat(
            [
                closest_rpos_gn,
                closest_distance_2d,
                closest_distance_z,
                closest_vel_g,
                closest_width_category,
                closest_height_category,
            ],
            dim=-1,
        )

        closest_distance_2d_collision = torch.linalg.norm(closest_rpos[..., :2], dim=-1, keepdim=True)
        closest_distance_2d_collision[range_mask.unsqueeze(-1)] = float("inf")
        closest_distance_z_collision = torch.linalg.norm(closest_rpos[..., 2:3], dim=-1, keepdim=True)
        closest_distance_z_collision[range_mask.unsqueeze(-1)] = float("inf")
        dynamic_collision_2d = closest_distance_2d_collision <= (closest_width * 0.5 + 0.3)
        dynamic_collision_z = closest_distance_z_collision <= (closest_height * 0.5 + 0.3)
        dynamic_collision = torch.any(dynamic_collision_2d & dynamic_collision_z, dim=1).squeeze(-1)

        closest_reward_dist = torch.linalg.norm(closest_rpos, dim=-1) - closest_size[..., 0] * 0.5
        closest_reward_dist[range_mask] = self.cfg.obstacle_detection_range

        if k < dyn_obs_num:
            pad_features = torch.zeros((self.num_envs, dyn_obs_num - k, 10), device=self.device)
            dyn_features = torch.cat([dyn_features, pad_features], dim=1)
            pad_reward = torch.full(
                (self.num_envs, dyn_obs_num - k), self.cfg.obstacle_detection_range, device=self.device
            )
            closest_reward_dist = torch.cat([closest_reward_dist, pad_reward], dim=1)

        return dyn_features.reshape(self.num_envs, dyn_obs_num * 10), dynamic_collision, closest_reward_dist

    def _get_observations(self) -> dict[str, torch.Tensor]:
        # 核心观测数据组装：收集当前的所有状态，构造成特征张量返回给策略网络 (Actor-Critic)。
        # 对齐 NavRL：目标方向坐标系 state + 最近动态障碍物 + 占据式 3D LiDAR。
        root_pos_w = self.robot.data.root_pos_w # 取出世界系 xyz
        root_quat_w = self.robot.data.root_quat_w # 取出世界系四元数
        target_vec_w = self._target_positions_w - root_pos_w
        target_dist = torch.linalg.norm(target_vec_w, dim=1, keepdim=True)
        target_dir_2d = target_vec_w.clone()
        target_dir_2d[:, 2] = 0.0
        rpos_clipped = target_vec_w / target_dist.clamp_min(1.0e-6)
        rpos_clipped_g = _vec_to_target_frame(rpos_clipped, target_dir_2d)
        distance_2d = torch.linalg.norm(target_vec_w[:, :2], dim=1, keepdim=True) / self.cfg.norm_max_dist
        distance_z = target_vec_w[:, 2:3] / self.cfg.norm_max_dist
        vel_g = _vec_to_target_frame(self.robot.data.root_lin_vel_w, target_dir_2d) / self.cfg.norm_max_vel
        drone_state = torch.cat([rpos_clipped_g, distance_2d, distance_z, vel_g], dim=-1)

        dynamic_features, _, _ = self._compute_dynamic_obstacle_features(root_pos_w, target_dir_2d)
        lidar_scan = 1.0 - self._compute_front_ray_distances(root_pos_w, root_quat_w)
        obs = torch.cat(
            [
                drone_state,
                dynamic_features,
                lidar_scan,
            ],
            dim=-1,
        )
        return {"policy": obs} # 包装成字典，适配 rl_games 和 gym 标准规范。

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        # gymnasium 标准的全局 reset 接口。首次 reset 会触发整个仿真世界的懒构建 (lazy build)。
        del options
        if seed is not None:
            self.seed(seed)
        if not self._built:
            self._build()
        env_ids = torch.arange(self.num_envs, device=self.device) # 生成包含所有环境索引的张量
        obs = self._reset_idx(env_ids) # 调用内部局部 reset 方法全量重置
        return obs, {} # 返回观察值和空的 info 字典

    def _reset_idx(self, env_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        # 张量化的局部重置方法。根据传入的 env_ids (张量)，仅重置这些发生碰撞或超时的特定环境。
        # 非常关键：这保证了不需要等待所有飞机全部死掉才统一重置，能够让不同智能体错峰重置，大幅提升采样吞吐量。
        if env_ids.numel() == 0:
            return self._get_observations() # 如果数组为空直接返回当前观测

        # 起点和目标按 NavRL 的边界区域逻辑独立采样。
        start_pos, target_pos = self._sample_navigation_position_pairs(len(env_ids))

        # default_root_state 是底层资产默认 root state 的模板拷贝，形状为 (num_envs, 13) 
        # 包含了 [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]
        # 这里在它的基础上把需要重置的环境状态重写为新的初值。
        root_state = self.robot.data.default_root_state.clone()
        diff = target_pos - start_pos
        # 让出生时机头大致朝向目标方向，以减轻由于控制器剧烈调整朝向带来的初始不稳定状态。
        yaw = torch.atan2(diff[:, 1], diff[:, 0])
        cy = torch.cos(yaw * 0.5)
        sy = torch.sin(yaw * 0.5)

        root_state[env_ids, :3] = start_pos # 覆写 XYZ
        root_state[env_ids, 3] = cy # qw
        root_state[env_ids, 4] = 0.0 # qx
        root_state[env_ids, 5] = 0.0 # qy
        root_state[env_ids, 6] = sy # qz (由四元数绕 Z 轴旋转半角的公式推导)
        # 线速度和角速度一律清零，从静止平稳开始新 episode。
        root_state[env_ids, 7:] = 0.0

        # 把修改好的新的根状态写入底层仿真引擎中。
        self.robot.write_root_state_to_sim(root_state[env_ids], env_ids=env_ids)
        # 重置电机/关节速度为其默认值
        self.robot.write_joint_state_to_sim(
            self.robot.data.default_joint_pos[env_ids],
            self.robot.data.default_joint_vel[env_ids],
            env_ids=env_ids,
        )

        # 同步更新与该环境相关的所有的张量缓存与“逻辑状态量”。
        self._target_positions_w[env_ids] = target_pos
        self._done_buf[env_ids] = False
        self._episode_success_buf[env_ids] = False
        self._rew_buf[env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0
        self._cmd_vel_b[env_ids] = 0.0
        self._prev_drone_vel_w[env_ids] = 0.0
        self._height_range[env_ids, 0] = torch.minimum(start_pos[:, 2], target_pos[:, 2])
        self._height_range[env_ids, 1] = torch.maximum(start_pos[:, 2], target_pos[:, 2])

        # 将被重置飞机的内部控制器状态同步归零和重置。
        controller_state = {
            "position": root_state[:, :3].clone(),
            "attitude": _quat_to_euler_deg(root_state[:, 3:7]),
        }
        self._controller.reset(controller_state, env_ids=env_ids)
        
        # 将速度设定点强行设为 0 悬停，避免继承上回合最后的狂暴动作冲刺指令。
        self._controller.set_velocity_setpoint(
            vx=torch.zeros(len(env_ids), device=self.device),
            vy=torch.zeros(len(env_ids), device=self.device),
            vz=torch.zeros(len(env_ids), device=self.device),
            velocity_body=True,
            env_ids=env_ids,
        )

        # 重置完成后但不推进物理时间的前提下，拉取更新一次张量缓存。
        self.robot.update(0.0)
        # 不在 reset 中调用 _lidar.update()，因为 BVH 重建在批量 reset 时极其耗时（会卡死）。
        # 改为将缓存设置为 1.0（最大探测距离 = 无障碍），下一次 step() 中 lidar.update() 后自动刷新。
        self._cached_lidar_distances_norm[env_ids] = 1.0
        # progress reward 是算两帧间势能差的，所以得重置一下新起点的初始距离基准。
        self._update_prev_target_distance(env_ids)
        # 将张量记录里的日志累加器清零，开始新的统计回合。
        for key in self._episode_sums:
            self._episode_sums[key][env_ids] = 0.0
        return self._get_observations() # 返回最新观测

    def step(self, actions: torch.Tensor):
        # RL 每次循环的核心推理步骤，接受模型推理输出的 actions 张量。
        # 这里的 actions 不是底层的四个电机推力大小，而是高阶速度指令：
        # [target_forward_vxy, target_left_vxy, world_vz_cmd]
        if not self._built:
            raise RuntimeError("Call reset() before step().")

        actions = actions.to(self.device).clamp(-1.0, 1.0) # 把网络输出严格限制在 [-1, 1] 以内，防发散
        
        # 策略观测已经被旋转到“目标方向坐标系”，所以动作的 XY 也按目标方向坐标系解释：
        # action[0] 是朝目标方向速度，action[1] 是相对目标方向的横向绕障速度。
        root_pos_w = self.robot.data.root_pos_w
        target_vec_xy = self._target_positions_w[:, :2] - root_pos_w[:, :2]
        target_dist_xy = torch.linalg.norm(target_vec_xy, dim=1, keepdim=True).clamp_min(1.0e-6)
        target_forward_xy = target_vec_xy / target_dist_xy
        target_left_xy = torch.stack((-target_forward_xy[:, 1], target_forward_xy[:, 0]), dim=1)
        cmd_forward = actions[:, 0:1] * self.cfg.cmd_body_vel_xy_max
        cmd_left = actions[:, 1:2] * self.cfg.cmd_body_vel_xy_max
        self._cmd_vel_b[:, :2] = cmd_forward * target_forward_xy + cmd_left * target_left_xy
        # 第三个维度解释为竖直(Z)速度命令。
        self._cmd_vel_b[:, 2] = actions[:, 2] * self.cfg.cmd_vel_z_max

        # 取出无人机当前的真实姿态和角速度，供控制系统使用。
        root_quat_w = self.robot.data.root_quat_w
        root_ang_vel_b = quat_apply_inverse(root_quat_w, self.robot.data.root_ang_vel_w)
        
        # 把高阶的速度指令送进 CrazyflieController（底层使用 PID 或自适应律）
        self._controller.set_velocity_setpoint(
            vx=self._cmd_vel_b[:, 0],
            vy=self._cmd_vel_b[:, 1],
            vz=self._cmd_vel_b[:, 2],
            velocity_body=False, # XY 已经是世界系速度，控制器内部再转到机体系
        )
        
        # 计算反馈控制误差，并得出最终施加在无人机重心的整体力(Force)与力矩(Torque)。
        force, torque = self._controller.compute(
            {
                "position": self.robot.data.root_pos_w,
                "velocity": self.robot.data.root_lin_vel_w,
                "attitude": _quat_to_euler_deg(root_quat_w),
                "angular_velocity": torch.rad2deg(root_ang_vel_b),
            }
        )
        # 写进缓冲张量的指定格式 (num_envs, num_bodies, xyz)
        self._thrust[:, 0, :] = force
        self._moment[:, 0, :] = torque

        # decimation (降采样/重复动作)：在一个 RL step 步长内，保持当前力矩，推动底层物理引擎若干次
        for _ in range(self.cfg.decimation):
            # 将算出的力矩直接作为外部力施加在 Base Link（核心躯干）上。
            self.robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)
            self.robot.write_data_to_sim() # 刷新到底层 PhysX
            self.sim.step() # PhysX 在后台推进一个 dt (例如 0.01 秒)
            self.robot.update(self.cfg.physics_dt) # 拉取物理引擎解算出的新位置
        self._move_dynamic_obstacles()
        if self._lidar is not None and self._lidar.is_initialized:
            self._lidar.update(dt=self.step_dt, force_recompute=True)
            self._refresh_lidar_cache()  # 从 RayCaster 拉取最新距离写入缓存
            self._draw_lidar_debug_rays()

        # 更新统计步数
        self.common_step_counter += 1
        self.episode_length_buf += 1

        # ==================== reward 计算 (奖励塑形 Reward Shaping) ====================
        # 对齐 NavRL：朝目标速度 + 静态障碍安全奖励 - 平滑惩罚 - 高度惩罚 + 常数项。
        root_quat_w = self.robot.data.root_quat_w
        root_pos_w = self.robot.data.root_pos_w
        target_vec_w = self._target_positions_w - self.robot.data.root_pos_w
        target_dir_2d = target_vec_w.clone()
        target_dir_2d[:, 2] = 0.0
        
        # 当前到目标的欧氏绝对距离。
        distance_to_target = torch.linalg.norm(target_vec_w, dim=1)
        target_direction = target_vec_w / distance_to_target.unsqueeze(1).clamp_min(1e-6) # 目标方向单位向量

        # 到达目标的当前步标记，并锁存当前 episode 是否曾经到达过目标。
        target_reached = distance_to_target < self.cfg.target_reach_threshold
        self._episode_success_buf |= target_reached

        # NavRL 的静态障碍安全奖励：
        # lidar_scan = (R - dist) / R, reward = log(R - lidar_scan)
        lidar_distances_norm = self._compute_front_ray_distances(self.robot.data.root_pos_w, root_quat_w)
        R = self.cfg.obstacle_detection_range
        lidar_scan = 1.0 - lidar_distances_norm
        reward_safety_static = torch.log(
            (R - lidar_scan).clamp(min=1.0e-6, max=R)
        ).mean(dim=1)
        lidar_distances = lidar_distances_norm * R

        # 判断是否发生了严重硬碰撞（任意 LiDAR 束命中距离小于安全余量）
        closest_hazard_distance = lidar_distances.min(dim=1).values
        obstacle_collision = closest_hazard_distance < self.cfg.obstacle_collision_margin
        _, dynamic_collision, dynamic_reward_dist = self._compute_dynamic_obstacle_features(root_pos_w, target_dir_2d)
        reward_safety_dynamic = torch.log(
            dynamic_reward_dist.clamp(min=1.0e-6, max=self.cfg.obstacle_detection_range)
        ).mean(dim=1)

        # NavRL 的朝目标速度奖励
        reward_vel = torch.sum(self.robot.data.root_lin_vel_w * target_direction, dim=1)
        reward_progress = self._prev_dist_to_target - distance_to_target

        # NavRL 的平滑惩罚：惩罚相邻两步的速度跳变
        penalty_smooth = torch.linalg.norm(self.robot.data.root_lin_vel_w - self._prev_drone_vel_w, dim=1)

        # NavRL 的高度惩罚：超过起终点高度带上下 0.2m 才开始惩罚
        lower_height = self._height_range[:, 0] - self.cfg.height_penalty_margin
        upper_height = self._height_range[:, 1] + self.cfg.height_penalty_margin
        penalty_height = torch.zeros(self.num_envs, device=self.device)
        above_band = root_pos_w[:, 2] > upper_height
        below_band = root_pos_w[:, 2] < lower_height
        penalty_height[above_band] = torch.square(root_pos_w[above_band, 2] - upper_height[above_band])
        penalty_height[below_band] = torch.square(lower_height[below_band] - root_pos_w[below_band, 2])

        self._prev_drone_vel_w[:] = self.robot.data.root_lin_vel_w
        self._prev_dist_to_target[:] = distance_to_target.detach()

        # 判断高度是否坠毁或飞出安全高度区间
        too_low = root_pos_w[:, 2] < self.cfg.min_flight_height
        too_high = root_pos_w[:, 2] > self.cfg.max_flight_height
        horizontal_bound = (
            max(
                float(self.cfg.spawn_edge_distance),
                float(self.cfg.target_spawn_range),
                float(self.cfg.obstacle_spawn_range),
            )
            + float(self.cfg.out_of_bounds_margin)
        )
        out_of_bounds = torch.any(root_pos_w[:, :2].abs() > horizontal_bound, dim=1)
        
        # NavRL 终止条件：低/高空越界或碰撞。到达目标只统计，不终止。
        died = too_low | too_high | obstacle_collision | dynamic_collision

        rewards = {
            "safety_static": reward_safety_static * self.cfg.safety_static_reward_scale,
            "safety_dynamic": reward_safety_dynamic * self.cfg.dynamic_safety_reward_scale,
            "velocity_to_goal": reward_vel * self.cfg.velocity_to_goal_reward_scale,
            "progress": reward_progress * self.cfg.progress_reward_scale,
            "constant": torch.full((self.num_envs,), float(self.cfg.constant_reward), device=self.device),
            "target_reach": target_reached.float() * self.cfg.target_reach_reward,
            "smoothness": -penalty_smooth * self.cfg.smoothness_penalty_scale,
            "height": -penalty_height * self.cfg.height_penalty_scale,
        }
        
        # 沿字典纵轴向把各项张量堆叠(stack)后求和，得出每个环境的单一总奖励值向量
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        
        # 累加统计日志用的记录本
        for key, value in rewards.items():
            self._episode_sums[key] += value
        self._rew_buf[:] = reward # 写入总奖励缓存

        # 如果开启了打印 Debug，则每隔定量的全局步数，在前台打印出平均奖励分布。
        if self.cfg.reward_debug_interval > 0 and (self.common_step_counter % self.cfg.reward_debug_interval) == 0:
            print(
                "[REWARD DEBUG] "
                f"step={self.common_step_counter} "
                f"vel_goal={rewards['velocity_to_goal'].mean().item():+.4f} "
                f"progress={rewards['progress'].mean().item():+.4f} "
                f"safety={rewards['safety_static'].mean().item():+.4f} "
                f"dyn_safety={rewards['safety_dynamic'].mean().item():+.4f} "
                f"constant={rewards['constant'].mean().item():+.4f} "
                f"target={rewards['target_reach'].mean().item():+.4f} "
                f"smooth={rewards['smoothness'].mean().item():+.4f} "
                f"height={rewards['height'].mean().item():+.4f}"
            )

        # ==================== 终止条件判定 (Termination / Truncation) ====================
        # Truncation (时间耗尽，被截断)
        timeout = self.episode_length_buf >= self.max_episode_length 
        # 到达目标立即成功终止，否则策略只会学习“朝目标有速度”，不一定学习真正到点。
        terminated = died | target_reached
        self._done_buf[:] = terminated | timeout # 任一结束条件满足，这回合即结束。

        # 整理一个含有详细事件触发标记的诊断字典，这个字典在 PPO 框架中通常被称为 infos 字典。
        extras = {
            "time_outs": timeout.clone(),
            "target_reached": target_reached.clone(),
            "episode_success": self._episode_success_buf.clone(),
            "terminated": terminated.clone(),
            "violation": obstacle_collision.clone(),
            "too_low": too_low.clone(),
            "too_high": too_high.clone(),
            "out_of_bounds": out_of_bounds.clone(),
            "obstacle_collision": obstacle_collision.clone(),
            "dynamic_collision": dynamic_collision.clone(),
            "distance_to_target": distance_to_target.clone(),
        }

        # 获取所有本步骤刚刚完结的那些特定环境的索引（例如第 3、7 架飞机死了或到了）。
        done_ids = torch.nonzero(self._done_buf, as_tuple=False).squeeze(-1)
        if done_ids.numel() > 0: # 如果存在完结的环境
            # 统计这一波结束的环境里的成功率：只要 episode 内曾经到达目标，就记为成功。
            episode_success = self._episode_success_buf[done_ids]
            batch_success_rate = episode_success.float().mean().item()

            log = {}
            for key in self._episode_sums:
                # 遍历累加器里的内容，把整个 episode 积累的总奖励除以该环境生命期的总秒数，得出归一化的平均项强度。
                episodic_sum_avg = torch.mean(self._episode_sums[key][done_ids])
                log[f"Episode_Reward/{key}"] = episodic_sum_avg / self.max_episode_length_s
            
            # 追加到 tensorboard 的各个诊断数据项的计数（在这个批次里，死于各类死法的人数）。
            log["Episode_Termination/died"] = torch.count_nonzero(died[done_ids]).item()
            log["Episode_Termination/time_out"] = torch.count_nonzero(timeout[done_ids]).item()
            log["Episode_Termination/too_low"] = torch.count_nonzero(too_low[done_ids]).item()
            log["Episode_Termination/too_high"] = torch.count_nonzero(too_high[done_ids]).item()
            log["Episode_Termination/out_of_bounds"] = torch.count_nonzero(out_of_bounds[done_ids]).item()
            log["Episode_Termination/obstacle_collision"] = torch.count_nonzero(obstacle_collision[done_ids]).item()
            log["Episode_Termination/dynamic_collision"] = torch.count_nonzero(dynamic_collision[done_ids]).item()
            log["Episode_Termination/target_reached"] = torch.count_nonzero(target_reached[done_ids]).item()
            log["Episode_Termination/episode_success"] = torch.count_nonzero(episode_success).item()
            log["Metrics/success_rate"] = batch_success_rate
            log["Metrics/success_rate_all_envs"] = self._episode_success_buf.float().mean().item()
            log["Metrics/avg_closest_hazard_distance"] = closest_hazard_distance.mean().item()
            log["Metrics/avg_distance_to_target"] = distance_to_target.mean().item()
            entered_obstacle_region = (
                (root_pos_w[:, 0].abs() <= float(self.cfg.obstacle_spawn_range))
                & (root_pos_w[:, 1].abs() <= float(self.cfg.obstacle_spawn_range))
            )
            log["Metrics/entered_obstacle_region_rate"] = entered_obstacle_region.float().mean().item()
            extras["log"] = log # 将日志丢进 extras 传出

        rew = self._rew_buf.clone()
        terminated_out = terminated.clone()
        truncated_out = (timeout & ~terminated).clone()  # 真正的截断 = 超时但没死

        # ===== 关键：保存截断环境的终端观测 =====
        # auto_reset 会覆盖截断环境的状态为新 episode 初始状态，
        # 但 SAC 需要用截断前的真实最后状态做 Q-value bootstrap，
        # 否则 Q(s_timeout) 被错误地估计为 Q(s_new_episode_at_edge)。
        if truncated_out.any():
            pre_reset_obs = self._get_observations()
            trunc_ids = torch.nonzero(truncated_out, as_tuple=False).flatten()
            extras["terminal_obs"] = {k: v[trunc_ids].clone() for k, v in pre_reset_obs.items()}
            extras["terminal_obs_mask"] = truncated_out.clone()

        # Gymnasium 特有逻辑：如果开启了 auto_reset，则帮策略层直接调用 reset。
        if self.cfg.auto_reset_done:
            if done_ids.numel() > 0:
                # 局部重置刚刚死掉/成功的那些环境，其余的无缝继续跑（异步批处理的关键所在）。
                self._reset_idx(done_ids)

        # 自动 reset 之后，再重新去底层系统拉取一次全新的 observation。
        # 这样返回给 RL 的 obs，对应的是重置后出生点的"下一状态"，符合 MDP 推理流。
        obs = self._get_observations()
        return obs, rew, terminated_out, truncated_out, extras

    def render(self) -> np.ndarray | None:
        if not self._built:
            return None
        self.sim.render() # 唤起 Isaac Sim 渲染管线。

        if self.render_mode != "rgb_array":
            return None # 如果只是 human 模式则后台播放画面，不返回内存数组。

        # 安全性拦截：确保当前跑的仿真实例并非纯 Headless 无 UI 模式。
        if self.sim.render_mode.value < self.sim.RenderMode.PARTIAL_RENDERING.value:
            raise RuntimeError(
                "Cannot render 'rgb_array' when the simulation render mode does not support rendering."
            )
            
        if self._rgb_annotator is None:
            # 延迟初始化 (Lazy Initialization)：
            # Omniverse Replicator (rep) 是图像生成的底层库。首次 render 时再延迟创建渲染附着节点，避免训练时徒增显存开销。
            import omni.replicator.core as rep

            # 绑定我们在配置里指定的相机路径和分辨率，创建一张图像纹理画布。
            self._render_product = rep.create.render_product(
                self.cfg.viewer_cam_prim_path,
                self.cfg.viewer_resolution,
            )
            # 申请使用 RGB 通道的标注器，并将数据拉回 CPU 内存端。
            self._rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
            self._rgb_annotator.attach([self._render_product]) # 挂载挂接
            
        # 提取当前帧底层内存数据，用 NumPy 直接重组为 (H, W, 4) 格式图片。
        rgb_data = self._rgb_annotator.get_data()
        rgb_data = np.frombuffer(rgb_data, dtype=np.uint8).reshape(*rgb_data.shape)
        
        if rgb_data.size == 0:
            # 在流水线刚启动，第一帧数据还没流出来时，直接造一张符合尺寸的纯黑遮罩顶上去，防止越界报错。
            width, height = self.cfg.viewer_resolution
            return np.zeros((height, width, 3), dtype=np.uint8)
            
        return rgb_data[:, :, :3] # 截去 Alpha 通道，只返回 RGB 3个通道。

    def close(self):
        # 析构环境函数：非常重要！必须正确切断并销毁 Omniverse 里的各种渲染、回调链接。
        # 否则连续启停脚本极易导致显存碎片堆积，引起 OOM (Out Of Memory) 报错和进程死锁。
        if self._sim is None:
            return
            
        # 安全断开和卸载图像渲染层提取管道
        if self._rgb_annotator is not None and self._render_product is not None:
            try:
                self._rgb_annotator.detach([self._render_product])
            except Exception:
                pass
        self._rgb_annotator = None
        self._render_product = None
        
        # 依次关停物理引擎的时间轴、清空所有的帧回调函数绑定、直接清除仿真上下文实例以释放底层显存。
        self._sim._timeline.stop()
        self._sim.clear_all_callbacks()
        self._sim.clear_instance()
        
        self._sim = None
        self._robot = None
        self._built = False # 状态退回未构建
