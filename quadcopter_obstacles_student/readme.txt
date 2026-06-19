4.2

日志：6-18——17：17
在 camera 和 teacher 环境里都加了同样逻辑：
原来：
action -> 直接变成 _cmd_vel_b -> 送控制器
现在：
action -> _target_cmd_vel_b -> 低通平滑 + 速度变化率限制 -> _cmd_vel_b -> 送控制器
涉及文件：
quadcopter_obstacles_camera/quadcopter_obstacles_env.py
quadcopter_obstacles_teacher/quadcopter_obstacles_env.py
quadcopter_obstacles_camera/cfg/drone.yaml
新增参数：
cmd_vel_smoothing_alpha: 0.35
cmd_body_vel_xy_rate_limit: 4.0
cmd_vel_z_rate_limit: 4.0
作用是减少 teacher/student 输出速度指令突然跳变导致的大俯仰角切换。


当前版本（相对 4.1 的改进）

目标：
在 4.1 的速度指令平滑基础上，继续解决 student 飞行过程中速度忽快忽慢、机体姿态和相机视野抖动、评估信息不够细的问题，并加入 yaw 控制能力。

1. student 动作从 3 维扩展到 4 维

4.1：
action = [vx_body, vy_body, vz]

当前：
action = [vx_body, vy_body, vz, yaw]

新增第 4 维 yaw 输出，环境里会把 action[:, 3] 映射到 yaw 角命令，并经过：
_target_cmd_yaw_deg -> yaw 低通平滑 + yaw 角速度限制 -> _cmd_yaw_deg -> controller.yaw_setpoint

新增参数：
cmd_yaw_angle_max: 180.0
cmd_yaw_smoothing_alpha: 0.35
cmd_yaw_rate_limit: 90.0

注意：
因为动作维度从 3 维变成 4 维，4.1 之前训练出来的旧 checkpoint 不能直接继续用，需要重新训练。


2. student 最大水平速度改成 2m/s

4.1：
cmd_body_vel_xy_max: 1.0

当前：
cmd_body_vel_xy_max: 2.0

同时保留速度命令平滑和限速：
cmd_vel_smoothing_alpha: 0.35
cmd_body_vel_xy_rate_limit: 2.0
cmd_vel_z_rate_limit: 4.0

目的：
让策略和控制器的最大水平速度统一到 2m/s，同时避免策略输出突然跳变导致一会儿加速、一会儿减速。


3. 增加平滑相关奖励，抑制抖动和频繁加减速

新增 reward 参数：
action_rate_penalty_scale: -1.0
cmd_rate_penalty_scale: -1.5
velocity_accel_penalty_scale: -0.8
overspeed_penalty_scale: -0.20

对应含义：
action_rate：惩罚策略 action 前后变化太大
cmd_rate：惩罚平滑后的速度命令变化太大
velocity_accel：惩罚实际速度变化太大
overspeed：惩罚水平速度超过 cmd_body_vel_xy_max

目的：
减少飞一段突然减速、再突然加速的现象，让策略更倾向于连续、平稳的速度输出。


4. 增加 yaw 朝向目标点的奖励

新增 reward 参数：
yaw_alignment_reward_scale: 1.0
yaw_error_penalty_scale: -0.2

环境会计算当前 yaw 和“无人机指向目标点方向”的夹角：
yaw_error_deg = target_yaw_deg - current_yaw_deg

奖励鼓励机头朝向目标点，减少相机视野乱晃和长期侧飞。

注意：
当前 yaw 动作是绝对世界系 yaw 角。这个设计能跑，但从可学习性上看，后面可能更适合改成“相对目标方向的 yaw offset”，例如：
cmd_yaw = target_yaw + action[:, 3] * yaw_offset_max
这样策略只需要学相对偏转角，不需要自己推世界系绝对 yaw。


5. 控制器增加 roll/pitch 姿态 setpoint 平滑

4.1 主要平滑的是策略输出的速度命令。

当前进一步在 controller 里平滑外环输出给姿态环的 roll/pitch setpoint：
ATTITUDE_RP_SETPOINT_SMOOTHING_ALPHA = 0.35
ATTITUDE_RP_SETPOINT_RATE_LIMIT_DPS = 30.0

同时把速度外环允许的最大姿态角收紧：
PID_VEL_ROLL_MAX = 15.0
PID_VEL_PITCH_MAX = 15.0

目的：
即使速度命令还有变化，控制器也不要马上给出很大的俯仰/横滚角，减少相机画面快速看天、看地的问题。


6. eval.py 增加更详细的诊断日志

新增可配置参数：
--seed
--diagnostic_interval
--diagnostic_env_id
--metrics_out

评估日志现在会记录更多控制链路信息：
action
target_cmd
cmd
target_cmd_yaw
cmd_yaw
yaw_error
vel_cmd_frame
rpy_deg
outer_rp
smoothed_rp
thrust_pwm

汇总指标里新增：
action_delta_abs_mean
action_jump_rate_abs_delta_gt_0p5
action_saturation_rate_abs_gt_0p95
cmd_delta_abs_mean
cmd_yaw_delta_abs_mean_deg
yaw_error_abs_mean_deg
cmd_tracking_error_mean
vel_tracking_error_mean
overspeed_rate_xy_gt_cmd_max
rpy_delta_abs_mean_deg
outer_rp_abs_mean_deg
smoothed_rp_abs_mean_deg

作用：
可以更容易判断飞行不平滑到底来自：
策略 action 抖动；
速度命令平滑后仍然跳变；
控制器 roll/pitch 输出过大；
实际速度跟踪误差过大；
yaw 指令或 yaw 跟踪不稳定。


7. 训练时的视频逻辑改为只录评估视频

4.1：
--video 会用 Gym RecordVideo 录训练过程。

当前：
--video 不再录训练 rollout，而是在 evaluation 触发时录评估视频。

视频保存位置：
logs/rsl_rl/quadcopter_obstacles_student/<run>/videos/eval/

如果 logger 是 wandb，会上传到：
Eval/video

作用：
不用看训练时随机探索的画面，而是直接看当前策略在评估环境里的真实表现。


8. 训练内评估保留 NavRL 风格统计，并加入静态/动态碰撞拆分

评估指标包括：
Eval/success_rate
Eval/collision_rate
Eval/static_collision_rate
Eval/dynamic_collision_rate
Eval/too_low_rate
Eval/too_high_rate
Eval/avg_speed
Eval/avg_speed_xy

目的：
区分 SR 低到底是静态障碍物碰撞多，还是动态障碍物碰撞多，方便针对性调参。


当前还需要注意的问题

1. 当前 yaw 输出是绝对世界系 yaw，后续建议改成相对目标方向的 yaw offset，可学习性更好。

2. eval.py 默认模型路径现在指向一个具体 checkpoint。如果动作维度或训练版本变了，最好显式传 --model，避免加载旧模型。

3. train.py 里的 --num_envs 默认值会覆盖 env.yaml 里的 scene.num_envs。如果想让 yaml 完全控制环境数量，后面可以把 --num_envs 默认改成 None。

4. 训练内 evaluation 的 num_envs 目前主要控制统计前多少个环境；底层 step 仍然会跑完整训练环境数量。如果想评估真正只跑 2048 台，需要单独创建 eval env。
