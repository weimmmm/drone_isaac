4.3
06-19_14:54
相对上一次提交 4.2 的改动：

1. 新增 drone_camera/quadcopter_obstacles_student 项目副本，用于 camera/student 方向的独立训练和评估。

2. 保留并整理 yaml 配置加载方式：
   cfg/env.yaml
   cfg/drone.yaml
   cfg/obstacle.yaml
   cfg/dynamic_obstacle.yaml
   cfg/reward.yaml
   cfg/train.yaml

3. student 动作维度为 4 维：
   [vx_body, vy_body, vz, yaw]
   其中速度命令和 yaw 命令都带低通平滑与限速。

4. 速度与平滑参数：
   cmd_body_vel_xy_max: 2.0
   cmd_vel_smoothing_alpha: 0.35
   cmd_body_vel_xy_rate_limit: 2.0
   cmd_vel_z_rate_limit: 4.0
   cmd_yaw_rate_limit: 90.0

5. reward 中增加平滑和稳定性约束：
   action_rate
   cmd_rate
   velocity_accel
   overspeed
   yaw_alignment
   yaw_error

6. 环境包含静态障碍物和动态障碍物，并在日志中区分静态/动态碰撞率，方便判断 SR 低的原因。

7. eval.py 增加诊断日志，记录 action、cmd、yaw、rpy、outer_rp、smoothed_rp、速度跟踪误差、超速率等信息，方便判断抖动来自策略还是控制器。

8. train.py 支持从 train.yaml 读取 wandb 和 evaluation 配置；--video 改成录评估视频，不录训练探索过程。

注意：
当前目录在 git 状态里是新增未跟踪目录；如果需要提交，需要把 drone_camera/quadcopter_obstacles_student 加入版本管理。
