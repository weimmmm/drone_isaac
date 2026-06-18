4.1

日志：6-18——12:15
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