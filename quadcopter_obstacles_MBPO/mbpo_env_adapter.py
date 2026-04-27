# 兼容Python 3.7+的注解语法，允许使用类自身类型注解
from __future__ import annotations

# 导入PyTorch，用于张量计算、GPU加速
import torch


class BatchedMBPOAdapter:
    """
    类功能注释：
    适配器类：将 Isaac Gym 批量并行的教师环境，适配为 MBPO 算法需要的批量接口
    MBPO：一种强化学习算法，需要统一的环境交互接口
    核心作用：统一环境调用方式，隔离底层环境差异
    """

    # 初始化函数：接收原始环境，配置参数
    def __init__(self, env):
        self.env = env                  # 保存传入的原始 Isaac 环境对象
        self.device = env.device        # 设备：CPU / GPU（和底层环境保持一致）
        self.num_envs = env.num_envs    # 批量并行的环境数量（同时跑多少个机器人/无人机）
        self.action_space = env.single_action_space  # 单个环境的动作空间
        self.action_dim = int(env.cfg.action_space)  # 动作维度（输出动作的长度）
        self.state_dim = int(env.cfg.observation_space)  # 状态/观测维度
        # 观测状态中：目标距离 对应的索引位置
        self._idx_target_dist = 4
        # 观测状态中：归一化高度/海拔 对应的索引位置
        self._idx_altitude = 10
        # 观测状态中：前向雷达束对应的索引位置
        self._idx_front_rays_start = 11
        self._idx_front_rays_end = self._idx_front_rays_start + int(env.cfg.num_obstacle_rays)

    # 内部函数：构建并返回当前的策略观测状态
    def _build_state(self):
        # 调用底层环境获取观测，提取 policy 分支的观测，发送到指定设备
        return self.env._get_observations()["policy"].to(self.device)

    # 重置所有环境（整体重置）
    def reset(self):
        self.env.reset()          # 重置底层所有并行环境
        return self._build_state()  # 返回重置后的观测状态

    # 仅重置【结束的环境】（高效批量重置，只重置done=True的环境）
    # 参数：done_mask 布尔张量，标记哪些环境需要重置
    def reset_done(self, done_mask: torch.Tensor):
        # 找到所有 done=True 的环境索引
        done_ids = torch.nonzero(done_mask, as_tuple=False).squeeze(-1)
        # 如果没有需要重置的环境，直接返回当前状态
        if done_ids.numel() == 0:
            return self._build_state()
        # 只重置指定索引的环境（批量高效操作）
        self.env._reset_idx(done_ids)
        # 返回重置后的新状态
        return self._build_state()

    # 环境步进函数：执行动作，返回 新状态、奖励、是否终止、信息字典
    # 核心交互接口，MBPO算法会高频调用
    def step(self, action: torch.Tensor):
        # 如果输入动作是一维（单个环境），扩展为二维适配批量格式 [1, action_dim]
        if action.ndim == 1:
            action = action.unsqueeze(0)
        # 执行底层环境步进，获取原始返回值
        _, rewards, terminated, truncated, extras = self.env.step(action)
        # 构建新的观测状态
        states = self._build_state()
        # 是否到达目标（布尔张量）
        target_reached = extras["target_reached"].bool()
        # SSAC 安全违规只使用障碍碰撞；高度越界仍终止真实 episode，但不进入大 C 安全惩罚。
        violations = extras["violation"].bool()
        # 封装信息字典，返回给强化学习算法
        info = {
            "violation": violations,                     # 是否发生安全违规（障碍碰撞）
            "target_reached": target_reached,           # 是否到达目标点
            "terminated": terminated.bool(),            # 是否任务终止
            "truncated": truncated.bool(),              # 是否步数截断（超时）
            "obstacle_collision": extras["obstacle_collision"].bool(),  # 是否撞到障碍物
            "too_low": extras["too_low"].bool(),        # 是否低空死亡
            "too_high": extras["too_high"].bool(),      # 是否高空死亡
            "out_of_bounds": extras["out_of_bounds"].bool(),  # 是否水平飞出训练区域
        }
        # 返回：新状态、奖励（转float）、是否终止、信息字典
        return states, rewards.float(), terminated.bool(), info

    # 检查【是否完成任务】：根据状态判断是否到达目标
    def check_done(self, states: torch.Tensor):
        # 到达目标只作为成功统计；高度越界是普通 episode 终止，不进入 SSAC 大 C 安全惩罚。
        altitude = states[:, self._idx_altitude]
        cfg = self.env.cfg
        altitude_scale = cfg.altitude_observation_scale()
        too_low = altitude < (cfg.min_flight_height / altitude_scale)
        too_high = altitude > (cfg.max_flight_height / altitude_scale)
        return too_low | too_high

    # 检查【是否违规/失败】：根据状态判断是否发生异常
    def check_violation(self, states: torch.Tensor):
        cfg = self.env.cfg  # 环境配置参数
        # 基于前向雷达相关观测估计最近障碍距离，不再依赖特权摘要量。
        front_rays = states[:, self._idx_front_rays_start:self._idx_front_rays_end]
        min_normalized_obstacle_distance = front_rays.min(dim=1).values
        collision_margin_normalized = cfg.obstacle_collision_margin / max(cfg.obstacle_detection_range, 1e-6)
        # 判定1：任意雷达束距离小于安全边距 → 碰撞违规
        obstacle_collision = min_normalized_obstacle_distance < collision_margin_normalized
        return obstacle_collision
