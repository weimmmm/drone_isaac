from __future__ import annotations

import argparse
import os
import sys


TASK_DIR = "/home/wei/End_to_end/drone_isaac/quadcopter_obstacles_teacher"
ENV_DIR = os.path.abspath(os.path.join(TASK_DIR, ".."))
ROOT_DIR = os.path.abspath(os.path.join(TASK_DIR, "..", ".."))
LOCAL_RSL_RL_DIR = os.path.join(ENV_DIR, "rsl_rl")
DEFAULT_MODEL_PATH = (
    "/home/wei/End_to_end/logs/rsl_rl/quadcopter_obstacles_v5/2026-03-13_16-30-27/model_1400.pt"
)

for path in (ROOT_DIR, ENV_DIR, LOCAL_RSL_RL_DIR, TASK_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)


def main():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Evaluate a trained teacher quadcopter obstacles policy.")
    parser.add_argument("--task", type=str, default="Isaac-Quadcopter-Obstacles-Teacher-v0")
    parser.add_argument("--num_envs", type=int, default=128)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--metrics_out", type=str, default="/home/wei/End_to_end/logs/rsl_rl/quadcopter_obstacles_v5/2026-03-13_16-30-27/eval_teacher_metrics.txt")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True

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

        from isaaclab.utils.dict import class_to_dict

        from local_rsl_rl import RslRlVecEnvWrapper
        from rsl_rl.runners import OnPolicyRunner

        import quadcopter_obstacles_teacher  # noqa: F401
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
        log("configs_loaded")

        env_cfg.scene.num_envs = args.num_envs
        env_cfg.debug_vis = False
        env_cfg.sim.device = args.device
        agent_cfg.device = args.device

        env = gym.make(args.task, cfg=env_cfg, render_mode=None)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        log("env_created")
        env.unwrapped.eval()
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
        speed_sum = 0.0
        speed_xy_sum = 0.0
        action_abs_sum = torch.zeros(env.num_actions, device=args.device)
        action_sq_sum = torch.zeros(env.num_actions, device=args.device)
        final_target_distance = None
        final_obstacle_distance = None
        final_wall_distance = None

        with torch.inference_mode():
            for step in range(eval_steps):
                actions = policy(obs)
                obs, rew, dones, extras = env.step(actions)

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
                    first_died[new_done] = (obstacle_collision | wall_collision | too_low | too_high)[new_done]
                    done_seen |= new_done

                lin_vel_w = env.unwrapped.robot.data.root_lin_vel_w
                speed_sum += torch.linalg.norm(lin_vel_w, dim=1).sum().item()
                speed_xy_sum += torch.linalg.norm(lin_vel_w[:, :2], dim=1).sum().item()

                action_abs_sum += actions.abs().sum(dim=0)
                action_sq_sum += torch.square(actions).sum(dim=0)

                target_world = env.unwrapped._target_positions_w
                robot_pos_w = env.unwrapped.robot.data.root_pos_w
                final_target_distance = torch.linalg.norm(target_world - robot_pos_w, dim=1)
                final_obstacle_distance = env.unwrapped._compute_closest_obstacle_signed_distance(robot_pos_w)
                final_wall_distance = env.unwrapped._compute_wall_signed_distance(robot_pos_w)

                if (step + 1) % 250 == 0:
                    mean_reward = reward_sum / ((step + 1) * args.num_envs)
                    done_count = int(done_seen.sum().item())
                    log(
                        f"[eval] step={step + 1} mean_step_reward={mean_reward:.4f} "
                        f"episodes_done={done_count} success={int(first_success.sum().item())} "
                        f"died={int(first_died.sum().item())} timeout={int(first_timeout.sum().item())}"
                    )

        total_samples = eval_steps * args.num_envs
        mean_reward = reward_sum / total_samples
        reward_std = max(reward_sq_sum / total_samples - mean_reward**2, 0.0) ** 0.5
        avg_speed = speed_sum / total_samples
        avg_speed_xy = speed_xy_sum / total_samples
        action_abs_mean = (action_abs_sum / total_samples).tolist()
        action_rms = torch.sqrt(action_sq_sum / total_samples).tolist()

        log("=== Evaluation Summary ===")
        log(f"model={args.model}")
        log(f"num_envs={args.num_envs}")
        log(f"steps={eval_steps}")
        done_count = int(done_seen.sum().item())
        success_count = int(first_success.sum().item())
        died_count = int(first_died.sum().item())
        timeout_count = int(first_timeout.sum().item())
        log(f"episodes_done={done_count}")
        log(f"success_count={success_count}")
        log(f"died_count={died_count}")
        log(f"timeout_count={timeout_count}")
        if done_count > 0:
            log(f"success_rate={success_count / done_count:.4f}")
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
        if final_wall_distance is not None:
            log(f"final_wall_distance_mean={final_wall_distance.mean().item():.4f}")
            log(f"final_wall_distance_median={final_wall_distance.median().item():.4f}")
        log(f"action_abs_mean={action_abs_mean}")
        log(f"action_rms={action_rms}")

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
