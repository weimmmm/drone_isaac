from __future__ import annotations

import argparse
import os
import random
import sys


DEFAULT_REPRO_SEED = 3407
TASK_DIR = os.path.abspath(os.path.dirname(__file__))
ENV_DIR = os.path.abspath(os.path.join(TASK_DIR, ".."))
ROOT_DIR = os.path.abspath(os.path.join(TASK_DIR, "..", ".."))
RSL_RL_DIR = os.path.join(ROOT_DIR, "rsl_rl")
ISAACLAB_ROOT = os.environ.get("ISAACLAB_PATH", "/home/wei/IsaacLab")
ISAACLAB_SOURCE_DIRS = [
    os.path.join(ISAACLAB_ROOT, "source", name)
    for name in ("isaaclab", "isaaclab_assets", "isaaclab_rl", "isaaclab_tasks")
]
DEFAULT_MODEL_PATH = "/home/wei/End_to_end/logs/rsl_rl/quadcopter_obstacles_student/2026-06-18_17-17-12_5090_quadcopter_obstacles_student_safe_multihead/model_5300.pt"
DEFAULT_METRICS_OUT = "/home/wei/End_to_end/logs/rsl_rl/quadcopter_obstacles_student/eval_student_metrics.txt"

for path in (ROOT_DIR, ENV_DIR, RSL_RL_DIR, TASK_DIR, *ISAACLAB_SOURCE_DIRS):
    if path not in sys.path:
        sys.path.insert(0, path)


def _bootstrap_repro_env(seed: int) -> None:
    required_env = {
        "PYTHONHASHSEED": str(seed),
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    }
    desired_env = os.environ.copy()
    python_paths = [
        path
        for path in (ROOT_DIR, ENV_DIR, RSL_RL_DIR, TASK_DIR, *ISAACLAB_SOURCE_DIRS)
        if os.path.isdir(path)
    ]
    existing_pythonpath = desired_env.get("PYTHONPATH", "")
    desired_env["PYTHONPATH"] = os.pathsep.join(
        [*python_paths, *[path for path in existing_pythonpath.split(os.pathsep) if path]]
    )
    changed = []
    for key, value in required_env.items():
        if desired_env.get(key) != value:
            desired_env[key] = value
            changed.append(f"{key}={value}")

    desired_env.setdefault("OMNI_KIT_RENDERER", "Vulkan")
    desired_env.setdefault("OMNI_KIT_NO_OPENGL_RENDERING", "1")
    if desired_env.get("OMNI_KIT_RENDERER") != os.environ.get("OMNI_KIT_RENDERER"):
        changed.append(f"OMNI_KIT_RENDERER={desired_env['OMNI_KIT_RENDERER']}")
    if desired_env.get("OMNI_KIT_NO_OPENGL_RENDERING") != os.environ.get("OMNI_KIT_NO_OPENGL_RENDERING"):
        changed.append(f"OMNI_KIT_NO_OPENGL_RENDERING={desired_env['OMNI_KIT_NO_OPENGL_RENDERING']}")

    if changed:
        if os.environ.get("QUADCOPTER_REPRO_BOOTSTRAPPED") == "1":
            raise RuntimeError("Reproducibility bootstrap failed: " + ", ".join(changed))
        desired_env["QUADCOPTER_REPRO_BOOTSTRAPPED"] = "1"
        print("[INFO] Re-exec with reproducibility env: " + ", ".join(changed), flush=True)
        os.execvpe(sys.executable, [sys.executable, *sys.argv], desired_env)


def _preparse_seed(default_seed: int) -> int:
    for idx, arg in enumerate(sys.argv[1:], start=1):
        if arg == "--seed" and idx + 1 < len(sys.argv):
            try:
                return int(sys.argv[idx + 1])
            except ValueError:
                return default_seed
        if arg.startswith("--seed="):
            try:
                return int(arg.split("=", 1)[1])
            except ValueError:
                return default_seed
    return default_seed


_bootstrap_repro_env(_preparse_seed(DEFAULT_REPRO_SEED))


def _configure_determinism(seed: int, torch_module) -> None:
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed(seed)
        torch_module.cuda.manual_seed_all(seed)
    torch_module.backends.cuda.matmul.allow_tf32 = False
    torch_module.backends.cudnn.allow_tf32 = False
    torch_module.backends.cudnn.deterministic = True
    torch_module.backends.cudnn.benchmark = False
    try:
        torch_module.set_float32_matmul_precision("highest")
    except Exception:
        pass
    torch_module.use_deterministic_algorithms(False)


def main():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Evaluate a trained student quadcopter obstacles policy.")
    parser.add_argument("--task", type=str, default="Isaac-Quadcopter-Obstacles-Student-v0")
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--metrics_out", type=str, default=DEFAULT_METRICS_OUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_REPRO_SEED)
    parser.add_argument("--diagnostic_interval", type=int, default=250)
    parser.add_argument("--diagnostic_env_id", type=int, default=0)
    parser.add_argument("--env_cfg_dir", type=str, default=os.path.join(TASK_DIR, "cfg"))
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    _bootstrap_repro_env(args.seed)
    args.enable_cameras = True

    os.makedirs(os.path.dirname(args.metrics_out), exist_ok=True)
    log_file = open(args.metrics_out, "w", encoding="utf-8")

    def log(message: str):
        print(message)
        log_file.write(message + "\n")
        log_file.flush()

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
     
    try:
        log("simulation_app_started")

        import gymnasium as gym
        import torch
        _configure_determinism(args.seed, torch)

        from isaaclab.utils.dict import class_to_dict

        from local_rsl_rl import RslRlVecEnvWrapper
        from rsl_rl.runners import OnPolicyRunner

        import quadcopter_obstacles_student  # noqa: F401
        from quadcopter_obstacles_student.config_utils import apply_env_cfg_dir
        from quadcopter_obstacles_student.quadcopter_obstacles_env import _quat_to_euler_deg
        from isaaclab.utils.math import quat_apply_inverse
        log("imports_after_app_ok")

        def _load_cfg_from_registry(task_name: str, entry_point_key: str):
            cfg_entry_point = gym.spec(task_name).kwargs.get(entry_point_key)
            if cfg_entry_point is None:
                raise ValueError(f"Missing '{entry_point_key}' for task '{task_name}'.")
            if callable(cfg_entry_point):
                return cfg_entry_point()
            if isinstance(cfg_entry_point, str):
                module_name, attr_name = cfg_entry_point.split(":")
                module = __import__(module_name, fromlist=[attr_name])
                cfg_or_cls = getattr(module, attr_name)
                return cfg_or_cls() if callable(cfg_or_cls) else cfg_or_cls
            return cfg_entry_point

        env_cfg = _load_cfg_from_registry(args.task, "env_cfg_entry_point")
        agent_cfg = _load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
        env_cfg_paths = apply_env_cfg_dir(env_cfg, args.env_cfg_dir)
        log("configs_loaded")
        if env_cfg_paths:
            for path in env_cfg_paths:
                log(f"env_config={path}")

        env_cfg.scene.num_envs = args.num_envs
        env_cfg.debug_vis = False
        env_cfg.sim.device = args.device
        env_cfg.seed = args.seed
        agent_cfg.seed = args.seed
        agent_cfg.device = args.device

        env = gym.make(args.task, cfg=env_cfg, render_mode=None)
        env.unwrapped.seed(args.seed)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        log("env_created")
        env.unwrapped.eval()
        env.seed(args.seed)
        obs = env.reset()
        env.unwrapped.episode_length_buf.zero_()
        log("env_reset_navrl_eval")

        runner = OnPolicyRunner(env, class_to_dict(agent_cfg), log_dir=None, device=agent_cfg.device)
        runner.load(args.model, load_optimizer=False)
        policy = runner.get_inference_policy(device=agent_cfg.device)
        log("policy_loaded")
        eval_steps = args.steps if args.steps is not None else env.max_episode_length

        reward_sum = 0.0
        reward_sq_sum = 0.0
        done_seen = torch.zeros(args.num_envs, dtype=torch.bool, device=args.device)
        episode_return = torch.zeros(args.num_envs, dtype=torch.float, device=args.device)
        episode_length = torch.zeros(args.num_envs, dtype=torch.float, device=args.device)
        first_return = torch.zeros(args.num_envs, dtype=torch.float, device=args.device)
        first_episode_length = torch.zeros(args.num_envs, dtype=torch.float, device=args.device)
        first_success = torch.zeros(args.num_envs, dtype=torch.bool, device=args.device)
        first_timeout = torch.zeros(args.num_envs, dtype=torch.bool, device=args.device)
        first_died = torch.zeros(args.num_envs, dtype=torch.bool, device=args.device)
        first_collision = torch.zeros(args.num_envs, dtype=torch.bool, device=args.device)
        first_static_collision = torch.zeros(args.num_envs, dtype=torch.bool, device=args.device)
        first_dynamic_collision = torch.zeros(args.num_envs, dtype=torch.bool, device=args.device)
        speed_sum = 0.0
        speed_xy_sum = 0.0
        action_abs_sum = torch.zeros(env.num_actions, device=args.device)
        action_sq_sum = torch.zeros(env.num_actions, device=args.device)
        prev_actions = None
        prev_target_cmd = None
        prev_cmd = None
        prev_cmd_yaw = None
        prev_vel_cmd_frame = None
        prev_rpy = None
        prev_outer_rp = None
        prev_smoothed_rp = None
        diag_sample_count = 0
        interval_diag_samples = 0
        action_delta_abs_sum = torch.zeros(env.num_actions, device=args.device)
        action_delta_sq_sum = torch.zeros(env.num_actions, device=args.device)
        action_jump_count = torch.zeros(env.num_actions, device=args.device)
        action_saturation_count = torch.zeros(env.num_actions, device=args.device)
        target_cmd_delta_abs_sum = torch.zeros(3, device=args.device)
        cmd_delta_abs_sum = torch.zeros(3, device=args.device)
        cmd_yaw_abs_sum = 0.0
        cmd_yaw_delta_abs_sum = 0.0
        yaw_error_abs_sum = 0.0
        vel_delta_abs_sum = torch.zeros(3, device=args.device)
        rpy_abs_sum = torch.zeros(3, device=args.device)
        rpy_delta_abs_sum = torch.zeros(3, device=args.device)
        outer_rp_abs_sum = torch.zeros(2, device=args.device)
        outer_rp_delta_abs_sum = torch.zeros(2, device=args.device)
        smoothed_rp_abs_sum = torch.zeros(2, device=args.device)
        smoothed_rp_delta_abs_sum = torch.zeros(2, device=args.device)
        cmd_tracking_error_sum = 0.0
        vel_tracking_error_sum = 0.0
        speed_delta_sum = 0.0
        rate_des_abs_sum = torch.zeros(3, device=args.device)
        rate_actual_abs_sum = torch.zeros(3, device=args.device)
        thrust_pwm_sum = 0.0
        thrust_pwm_delta_sum = 0.0
        motor_pwm_range_sum = 0.0
        overspeed_count = 0.0
        prev_thrust_pwm = None
        interval_action_delta_abs_sum = torch.zeros(env.num_actions, device=args.device)
        interval_action_jump_count = torch.zeros(env.num_actions, device=args.device)
        interval_action_saturation_count = torch.zeros(env.num_actions, device=args.device)
        interval_cmd_delta_abs_sum = torch.zeros(3, device=args.device)
        interval_cmd_yaw_delta_abs_sum = 0.0
        interval_yaw_error_abs_sum = 0.0
        interval_vel_delta_abs_sum = torch.zeros(3, device=args.device)
        interval_rpy_delta_abs_sum = torch.zeros(3, device=args.device)
        interval_outer_rp_abs_sum = torch.zeros(2, device=args.device)
        interval_outer_rp_delta_abs_sum = torch.zeros(2, device=args.device)
        interval_smoothed_rp_abs_sum = torch.zeros(2, device=args.device)
        interval_smoothed_rp_delta_abs_sum = torch.zeros(2, device=args.device)
        interval_vel_tracking_error_sum = 0.0
        interval_overspeed_count = 0.0
        final_target_distance = None
        final_obstacle_distance = None
        final_dynamic_obstacle_distance = None
        final_wall_distance = None

        with torch.inference_mode():
            for step in range(eval_steps):
                actions = policy(obs)
                obs, rew, dones, extras = env.step(actions)
                unwrapped = env.unwrapped

                rew_flat = rew.reshape(-1)
                reward_sum += rew_flat.sum().item()
                reward_sq_sum += torch.square(rew_flat).sum().item()
                episode_return += rew_flat
                episode_length += (~done_seen).float()

                dones_bool = dones.bool().reshape(-1)
                new_done = dones_bool & (~done_seen)
                if new_done.any():
                    first_return[new_done] = episode_return[new_done]
                    first_episode_length[new_done] = episode_length[new_done]
                    success = extras.get("target_reached", torch.zeros_like(dones_bool)).bool()
                    timeout = extras.get("time_outs", torch.zeros_like(dones_bool)).bool()
                    obstacle_collision = extras.get("obstacle_collision", torch.zeros_like(dones_bool)).bool()
                    wall_collision = extras.get("wall_collision", torch.zeros_like(dones_bool)).bool()
                    too_low = extras.get("too_low", torch.zeros_like(dones_bool)).bool()
                    too_high = extras.get("too_high", torch.zeros_like(dones_bool)).bool()
                    first_success[new_done] = success[new_done]
                    first_timeout[new_done] = timeout[new_done]
                    dynamic_obstacle_collision = extras.get(
                        "dynamic_obstacle_collision", torch.zeros_like(dones_bool)
                    ).bool()
                    first_static_collision[new_done] = obstacle_collision[new_done]
                    first_dynamic_collision[new_done] = dynamic_obstacle_collision[new_done]
                    first_collision[new_done] = (
                        obstacle_collision | dynamic_obstacle_collision | wall_collision
                    )[new_done]
                    first_died[new_done] = (
                        obstacle_collision | dynamic_obstacle_collision | wall_collision | too_low | too_high
                    )[new_done]
                    done_seen |= new_done

                lin_vel_w = unwrapped.robot.data.root_lin_vel_w
                speed_sum += torch.linalg.norm(lin_vel_w, dim=1).sum().item()
                speed_xy_sum += torch.linalg.norm(lin_vel_w[:, :2], dim=1).sum().item()

                action_abs_sum += actions.abs().sum(dim=0)
                action_sq_sum += torch.square(actions).sum(dim=0)
                action_saturation_count += (actions.abs() > 0.95).float().sum(dim=0)

                root_quat_w = unwrapped.robot.data.root_quat_w
                rpy = _quat_to_euler_deg(root_quat_w)
                lin_vel_b = quat_apply_inverse(root_quat_w, lin_vel_w)
                vel_cmd_frame = torch.stack([lin_vel_b[:, 0], lin_vel_b[:, 1], lin_vel_w[:, 2]], dim=1)
                target_cmd = getattr(unwrapped, "_target_cmd_vel_b", unwrapped._cmd_vel_b)
                cmd = unwrapped._cmd_vel_b
                cmd_yaw = getattr(unwrapped, "_cmd_yaw_deg", torch.zeros(args.num_envs, device=args.device))
                target_cmd_yaw = getattr(unwrapped, "_target_cmd_yaw_deg", cmd_yaw)
                yaw_error_deg = extras.get("yaw_error_deg", torch.zeros(args.num_envs, device=args.device))
                controller = unwrapped._controller
                outer_rp = torch.stack([controller._outer_roll_cmd, controller._outer_pitch_cmd], dim=1)
                smoothed_rp = torch.stack([controller._smoothed_roll_des, controller._smoothed_pitch_des], dim=1)
                thrust_pwm = controller._outer_thrust_cmd
                motor_pwm = controller.power_distribution.motor_thrust

                diag_sample_count += args.num_envs
                interval_diag_samples += args.num_envs
                speed_xy_now = torch.linalg.norm(lin_vel_w[:, :2], dim=1)
                overspeed_count += (speed_xy_now > float(unwrapped.cfg.cmd_body_vel_xy_max)).float().sum().item()
                interval_overspeed_count += (speed_xy_now > float(unwrapped.cfg.cmd_body_vel_xy_max)).float().sum().item()
                rpy_abs_sum += rpy.abs().sum(dim=0)
                outer_rp_abs_sum += outer_rp.abs().sum(dim=0)
                smoothed_rp_abs_sum += smoothed_rp.abs().sum(dim=0)
                interval_outer_rp_abs_sum += outer_rp.abs().sum(dim=0)
                interval_smoothed_rp_abs_sum += smoothed_rp.abs().sum(dim=0)
                cmd_tracking_error_sum += torch.linalg.norm(target_cmd - cmd, dim=1).sum().item()
                cmd_yaw_abs_sum += cmd_yaw.abs().sum().item()
                yaw_error_abs_sum += yaw_error_deg.abs().sum().item()
                interval_yaw_error_abs_sum += yaw_error_deg.abs().sum().item()
                vel_tracking_error_sum += torch.linalg.norm(cmd - vel_cmd_frame, dim=1).sum().item()
                interval_vel_tracking_error_sum += torch.linalg.norm(cmd - vel_cmd_frame, dim=1).sum().item()
                thrust_pwm_sum += thrust_pwm.abs().sum().item()
                motor_pwm_range_sum += (motor_pwm.max(dim=1).values - motor_pwm.min(dim=1).values).sum().item()
                rate_desired = getattr(controller.attitude_controller, "last_rate_desired", None)
                rate_actual = getattr(controller.attitude_controller, "last_rate_actual", None)
                if rate_desired is not None:
                    rate_des_abs_sum += rate_desired.abs().sum(dim=0)
                if rate_actual is not None:
                    rate_actual_abs_sum += rate_actual.abs().sum(dim=0)

                if prev_actions is not None:
                    action_delta = actions - prev_actions
                    target_cmd_delta = target_cmd - prev_target_cmd
                    cmd_delta = cmd - prev_cmd
                    cmd_yaw_delta = ((cmd_yaw - prev_cmd_yaw + 180.0) % 360.0) - 180.0
                    vel_delta = vel_cmd_frame - prev_vel_cmd_frame
                    rpy_delta = rpy - prev_rpy
                    outer_rp_delta = outer_rp - prev_outer_rp
                    smoothed_rp_delta = smoothed_rp - prev_smoothed_rp
                    action_delta_abs_sum += action_delta.abs().sum(dim=0)
                    action_delta_sq_sum += torch.square(action_delta).sum(dim=0)
                    action_jump_count += (action_delta.abs() > 0.5).float().sum(dim=0)
                    interval_action_delta_abs_sum += action_delta.abs().sum(dim=0)
                    interval_action_jump_count += (action_delta.abs() > 0.5).float().sum(dim=0)
                    interval_action_saturation_count += (actions.abs() > 0.95).float().sum(dim=0)
                    target_cmd_delta_abs_sum += target_cmd_delta.abs().sum(dim=0)
                    cmd_delta_abs_sum += cmd_delta.abs().sum(dim=0)
                    cmd_yaw_delta_abs_sum += cmd_yaw_delta.abs().sum().item()
                    vel_delta_abs_sum += vel_delta.abs().sum(dim=0)
                    rpy_delta_abs_sum += rpy_delta.abs().sum(dim=0)
                    outer_rp_delta_abs_sum += outer_rp_delta.abs().sum(dim=0)
                    smoothed_rp_delta_abs_sum += smoothed_rp_delta.abs().sum(dim=0)
                    interval_cmd_delta_abs_sum += cmd_delta.abs().sum(dim=0)
                    interval_cmd_yaw_delta_abs_sum += cmd_yaw_delta.abs().sum().item()
                    interval_vel_delta_abs_sum += vel_delta.abs().sum(dim=0)
                    interval_rpy_delta_abs_sum += rpy_delta.abs().sum(dim=0)
                    interval_outer_rp_delta_abs_sum += outer_rp_delta.abs().sum(dim=0)
                    interval_smoothed_rp_delta_abs_sum += smoothed_rp_delta.abs().sum(dim=0)
                    speed_delta_sum += torch.linalg.norm(vel_delta, dim=1).sum().item()
                    thrust_pwm_delta_sum += (thrust_pwm - prev_thrust_pwm).abs().sum().item()

                prev_actions = actions.detach().clone()
                prev_target_cmd = target_cmd.detach().clone()
                prev_cmd = cmd.detach().clone()
                prev_cmd_yaw = cmd_yaw.detach().clone()
                prev_vel_cmd_frame = vel_cmd_frame.detach().clone()
                prev_rpy = rpy.detach().clone()
                prev_outer_rp = outer_rp.detach().clone()
                prev_smoothed_rp = smoothed_rp.detach().clone()
                prev_thrust_pwm = thrust_pwm.detach().clone()

                target_world = unwrapped._target_positions_w
                robot_pos_w = unwrapped.robot.data.root_pos_w
                final_target_distance = torch.linalg.norm(target_world - robot_pos_w, dim=1)
                final_obstacle_distance = unwrapped._compute_closest_obstacle_signed_distance(robot_pos_w)
                if "closest_dynamic_obstacle_distance" in extras:
                    final_dynamic_obstacle_distance = extras["closest_dynamic_obstacle_distance"]
                final_wall_distance = unwrapped._compute_wall_signed_distance(robot_pos_w)

                if args.diagnostic_interval > 0 and (step + 1) % args.diagnostic_interval == 0:
                    mean_reward = reward_sum / ((step + 1) * args.num_envs)
                    done_count = int(done_seen.sum().item())
                    env_id = max(0, min(int(args.diagnostic_env_id), args.num_envs - 1))
                    log(
                        f"[eval] step={step + 1} mean_step_reward={mean_reward:.4f} "
                        f"episodes_done={done_count} success={int(first_success.sum().item())} "
                        f"died={int(first_died.sum().item())} timeout={int(first_timeout.sum().item())}"
                    )
                    log(
                        f"[diag/env{env_id}] action={actions[env_id].detach().cpu().tolist()} "
                        f"target_cmd={target_cmd[env_id].detach().cpu().tolist()} "
                        f"cmd={cmd[env_id].detach().cpu().tolist()} "
                        f"target_cmd_yaw={float(target_cmd_yaw[env_id].item()):.2f} "
                        f"cmd_yaw={float(cmd_yaw[env_id].item()):.2f} "
                        f"yaw_error={float(yaw_error_deg[env_id].item()):.2f} "
                        f"vel_cmd_frame={vel_cmd_frame[env_id].detach().cpu().tolist()} "
                        f"rpy_deg={rpy[env_id].detach().cpu().tolist()} "
                        f"outer_rp={outer_rp[env_id].detach().cpu().tolist()} "
                        f"smoothed_rp={smoothed_rp[env_id].detach().cpu().tolist()} "
                        f"thrust_pwm={float(thrust_pwm[env_id].item()):.1f}"
                    )
                    interval_delta_samples = max(interval_diag_samples - args.num_envs, 1)
                    log(
                        "[diag/interval] "
                        f"action_delta_abs_mean={(interval_action_delta_abs_sum / interval_delta_samples).detach().cpu().tolist()} "
                        f"action_jump_rate={(interval_action_jump_count / interval_delta_samples).detach().cpu().tolist()} "
                        f"action_saturation_rate={(interval_action_saturation_count / max(interval_diag_samples, 1)).detach().cpu().tolist()} "
                        f"cmd_delta_abs_mean={(interval_cmd_delta_abs_sum / interval_delta_samples).detach().cpu().tolist()} "
                        f"cmd_yaw_delta_abs_mean={interval_cmd_yaw_delta_abs_sum / interval_delta_samples:.6f} "
                        f"yaw_error_abs_mean_deg={interval_yaw_error_abs_sum / max(interval_diag_samples, 1):.6f} "
                        f"vel_delta_abs_mean={(interval_vel_delta_abs_sum / interval_delta_samples).detach().cpu().tolist()} "
                        f"rpy_delta_abs_mean_deg={(interval_rpy_delta_abs_sum / interval_delta_samples).detach().cpu().tolist()} "
                        f"outer_rp_abs_mean_deg={(interval_outer_rp_abs_sum / max(interval_diag_samples, 1)).detach().cpu().tolist()} "
                        f"outer_rp_delta_abs_mean_deg={(interval_outer_rp_delta_abs_sum / interval_delta_samples).detach().cpu().tolist()} "
                        f"smoothed_rp_abs_mean_deg={(interval_smoothed_rp_abs_sum / max(interval_diag_samples, 1)).detach().cpu().tolist()} "
                        f"smoothed_rp_delta_abs_mean_deg={(interval_smoothed_rp_delta_abs_sum / interval_delta_samples).detach().cpu().tolist()} "
                        f"vel_tracking_error_mean={interval_vel_tracking_error_sum / max(interval_diag_samples, 1):.6f}"
                        f" overspeed_rate={interval_overspeed_count / max(interval_diag_samples, 1):.6f}"
                    )
                    interval_diag_samples = 0
                    interval_action_delta_abs_sum.zero_()
                    interval_action_jump_count.zero_()
                    interval_action_saturation_count.zero_()
                    interval_cmd_delta_abs_sum.zero_()
                    interval_cmd_yaw_delta_abs_sum = 0.0
                    interval_yaw_error_abs_sum = 0.0
                    interval_vel_delta_abs_sum.zero_()
                    interval_rpy_delta_abs_sum.zero_()
                    interval_outer_rp_abs_sum.zero_()
                    interval_outer_rp_delta_abs_sum.zero_()
                    interval_smoothed_rp_abs_sum.zero_()
                    interval_smoothed_rp_delta_abs_sum.zero_()
                    interval_vel_tracking_error_sum = 0.0
                    interval_overspeed_count = 0.0

        total_samples = eval_steps * args.num_envs
        mean_reward = reward_sum / total_samples
        reward_std = max(reward_sq_sum / total_samples - mean_reward**2, 0.0) ** 0.5
        avg_speed = speed_sum / total_samples
        avg_speed_xy = speed_xy_sum / total_samples
        action_abs_mean = (action_abs_sum / total_samples).tolist()
        action_rms = torch.sqrt(action_sq_sum / total_samples).tolist()
        action_saturation_rate = (action_saturation_count / total_samples).tolist()
        delta_samples = max((eval_steps - 1) * args.num_envs, 1)
        action_delta_abs_mean = (action_delta_abs_sum / delta_samples).tolist()
        action_delta_rms = torch.sqrt(action_delta_sq_sum / delta_samples).tolist()
        action_jump_rate = (action_jump_count / delta_samples).tolist()
        target_cmd_delta_abs_mean = (target_cmd_delta_abs_sum / delta_samples).tolist()
        cmd_delta_abs_mean = (cmd_delta_abs_sum / delta_samples).tolist()
        cmd_yaw_abs_mean = cmd_yaw_abs_sum / max(diag_sample_count, 1)
        cmd_yaw_delta_abs_mean = cmd_yaw_delta_abs_sum / delta_samples
        yaw_error_abs_mean = yaw_error_abs_sum / max(diag_sample_count, 1)
        vel_delta_abs_mean = (vel_delta_abs_sum / delta_samples).tolist()
        rpy_abs_mean = (rpy_abs_sum / max(diag_sample_count, 1)).tolist()
        rpy_delta_abs_mean = (rpy_delta_abs_sum / delta_samples).tolist()
        outer_rp_abs_mean = (outer_rp_abs_sum / max(diag_sample_count, 1)).tolist()
        outer_rp_delta_abs_mean = (outer_rp_delta_abs_sum / delta_samples).tolist()
        smoothed_rp_abs_mean = (smoothed_rp_abs_sum / max(diag_sample_count, 1)).tolist()
        smoothed_rp_delta_abs_mean = (smoothed_rp_delta_abs_sum / delta_samples).tolist()
        rate_des_abs_mean = (rate_des_abs_sum / max(diag_sample_count, 1)).tolist()
        rate_actual_abs_mean = (rate_actual_abs_sum / max(diag_sample_count, 1)).tolist()

        log("=== Evaluation Summary ===")
        log(f"model={args.model}")
        log(f"num_envs={args.num_envs}")
        log(f"steps={eval_steps}")
        done_count = int(done_seen.sum().item())
        success_count = int(first_success.sum().item())
        died_count = int(first_died.sum().item())
        timeout_count = int(first_timeout.sum().item())
        collision_count = int(first_collision.sum().item())
        static_collision_count = int(first_static_collision.sum().item())
        dynamic_collision_count = int(first_dynamic_collision.sum().item())
        log(f"episodes_done={done_count}")
        log(f"success_count={success_count}")
        log(f"died_count={died_count}")
        log(f"timeout_count={timeout_count}")
        log(f"collision_count={collision_count}")
        log(f"static_collision_count={static_collision_count}")
        log(f"dynamic_collision_count={dynamic_collision_count}")
        if done_count > 0:
            log(f"success_rate={success_count / done_count:.4f}")
            log(f"collision_rate={collision_count / done_count:.4f}")
            log(f"static_collision_rate={static_collision_count / done_count:.4f}")
            log(f"dynamic_collision_rate={dynamic_collision_count / done_count:.4f}")
            log(f"died_rate={died_count / done_count:.4f}")
            log(f"timeout_rate={timeout_count / done_count:.4f}")
        log(f"first_episode_return_mean={first_return.mean().item():.4f}")
        log(f"first_episode_length_mean={first_episode_length.mean().item():.4f}")
        log(f"mean_step_reward={mean_reward:.6f}")
        log(f"std_step_reward={reward_std:.6f}")
        log(f"avg_speed_mean={avg_speed:.4f}")
        log(f"avg_speed_xy_mean={avg_speed_xy:.4f}")
        if final_target_distance is not None:
            log(f"final_target_distance_mean={final_target_distance.mean().item():.4f}")
            log(f"final_target_distance_median={final_target_distance.median().item():.4f}")
        if final_obstacle_distance is not None:
            log(f"final_obstacle_distance_mean={final_obstacle_distance.mean().item():.4f}")
            log(f"final_obstacle_distance_median={final_obstacle_distance.median().item():.4f}")
        if final_dynamic_obstacle_distance is not None:
            log(f"final_dynamic_obstacle_distance_mean={final_dynamic_obstacle_distance.mean().item():.4f}")
            log(f"final_dynamic_obstacle_distance_median={final_dynamic_obstacle_distance.median().item():.4f}")
        if final_wall_distance is not None:
            log(f"final_wall_distance_mean={final_wall_distance.mean().item():.4f}")
            log(f"final_wall_distance_median={final_wall_distance.median().item():.4f}")
        log(f"action_abs_mean={action_abs_mean}")
        log(f"action_rms={action_rms}")
        log(f"action_saturation_rate_abs_gt_0p95={action_saturation_rate}")
        log("=== Control Diagnostics ===")
        log(f"action_delta_abs_mean={action_delta_abs_mean}")
        log(f"action_delta_rms={action_delta_rms}")
        log(f"action_jump_rate_abs_delta_gt_0p5={action_jump_rate}")
        log(f"target_cmd_delta_abs_mean={target_cmd_delta_abs_mean}")
        log(f"cmd_delta_abs_mean={cmd_delta_abs_mean}")
        log(f"cmd_yaw_abs_mean_deg={cmd_yaw_abs_mean:.6f}")
        log(f"cmd_yaw_delta_abs_mean_deg={cmd_yaw_delta_abs_mean:.6f}")
        log(f"yaw_error_abs_mean_deg={yaw_error_abs_mean:.6f}")
        log(f"vel_delta_abs_mean={vel_delta_abs_mean}")
        log(f"cmd_tracking_error_mean={cmd_tracking_error_sum / max(diag_sample_count, 1):.6f}")
        log(f"vel_tracking_error_mean={vel_tracking_error_sum / max(diag_sample_count, 1):.6f}")
        log(f"overspeed_rate_xy_gt_cmd_max={overspeed_count / max(diag_sample_count, 1):.6f}")
        log(f"speed_delta_mean={speed_delta_sum / delta_samples:.6f}")
        log(f"rpy_abs_mean_deg={rpy_abs_mean}")
        log(f"rpy_delta_abs_mean_deg={rpy_delta_abs_mean}")
        log(f"outer_rp_abs_mean_deg={outer_rp_abs_mean}")
        log(f"outer_rp_delta_abs_mean_deg={outer_rp_delta_abs_mean}")
        log(f"smoothed_rp_abs_mean_deg={smoothed_rp_abs_mean}")
        log(f"smoothed_rp_delta_abs_mean_deg={smoothed_rp_delta_abs_mean}")
        log(f"rate_des_abs_mean_dps={rate_des_abs_mean}")
        log(f"rate_actual_abs_mean_dps={rate_actual_abs_mean}")
        log(f"thrust_pwm_abs_mean={thrust_pwm_sum / max(diag_sample_count, 1):.3f}")
        log(f"thrust_pwm_delta_abs_mean={thrust_pwm_delta_sum / delta_samples:.3f}")
        log(f"motor_pwm_range_mean={motor_pwm_range_sum / max(diag_sample_count, 1):.3f}")

        env.close()
    except Exception as exc:
        log(f"exception={type(exc).__name__}: {exc}")
        raise
    finally:
        try:
            log("simulation_app_closing")
        except Exception:
            pass
        log_file.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
