from __future__ import annotations

import argparse
import os
import re
import sys
import traceback


TASK_DIR = os.path.abspath(os.path.dirname(__file__))
ENV_DIR = os.path.abspath(os.path.join(TASK_DIR, ".."))
ROOT_DIR = os.path.abspath(os.path.join(TASK_DIR, "..", ".."))
LOCAL_RSL_RL_DIR = os.path.join(ENV_DIR, "rsl_rl")

for path in (ROOT_DIR, ENV_DIR, LOCAL_RSL_RL_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)


from isaaclab.app import AppLauncher


DEFAULT_CHECKPOINT = "/home/wei/End_to_end/logs/rsl_rl/quadcopter_obstacles_refiner/2026-03-18_11-02-34/model_3850.pt"


parser = argparse.ArgumentParser(description="Evaluate the standalone depth student or teacher baseline.")
parser.add_argument("--task", type=str, default="Isaac-Quadcopter-Obstacles-Refiner-v0")
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--mode", type=str, choices=("student", "teacher"), default="student")
parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
parser.add_argument("--teacher_checkpoint", type=str, default=None)
parser.add_argument(
    "--metrics_out",
    type=str,
    default="/home/wei/End_to_end/logs/rsl_rl/quadcopter_obstacles_refiner/2026-03-18_11-02-34/eval_refiner_metrics.txt",
)
parser.add_argument("--max_steps", type=int, default=4000)
parser.add_argument("--debug_logs", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
from PIL import Image

from isaaclab.utils.dict import class_to_dict

from local_rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

import quadcopter_obstacles_refiner  # noqa: F401


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


def _resolve_checkpoint_path(checkpoint_path: str, experiment_name: str) -> str:
    if os.path.exists(checkpoint_path):
        return checkpoint_path

    log_root = os.path.join(ROOT_DIR, "logs", "rsl_rl", experiment_name)
    latest_dir = os.path.join(log_root, "latest")
    if os.path.isdir(latest_dir):
        checkpoints = [name for name in os.listdir(latest_dir) if re.match(r"model_.*\.pt", name)]
        if checkpoints:
            checkpoints.sort(key=lambda name: f"{name:0>15}")
            return os.path.join(latest_dir, checkpoints[-1])

    if not os.path.isdir(log_root):
        raise FileNotFoundError(f"Checkpoint not found and log root does not exist: {checkpoint_path}")

    run_dirs = [entry.path for entry in os.scandir(log_root) if entry.is_dir() and entry.name != "latest"]
    if not run_dirs:
        raise FileNotFoundError(f"No run directories available under: {log_root}")
    run_dirs.sort()
    for run_dir in reversed(run_dirs):
        checkpoints = [name for name in os.listdir(run_dir) if re.match(r"model_.*\.pt", name)]
        if checkpoints:
            checkpoints.sort(key=lambda name: f"{name:0>15}")
            return os.path.join(run_dir, checkpoints[-1])
    raise FileNotFoundError(f"No checkpoints found under: {log_root}")


def _save_depth_snapshot(env, output_dir: str, step: int, env_index: int = 0) -> str | None:
    """保存一帧深度图，默认导出第 0 个并行环境。"""
    unwrapped = env.unwrapped
    if not hasattr(unwrapped, "get_record_depth_frames_u8"):
        return None

    depth_u8 = unwrapped.get_record_depth_frames_u8()
    if depth_u8.numel() == 0:
        return None

    env_index = max(0, min(int(env_index), depth_u8.shape[0] - 1))
    frame = depth_u8[env_index].detach().cpu().numpy().astype(np.uint8)
    output_path = os.path.join(output_dir, f"depth_step_{step:06d}_env_{env_index:03d}.png")
    Image.fromarray(frame, mode="L").save(output_path)
    return output_path


def main() -> None:
    os.chdir(ROOT_DIR)
    os.makedirs(os.path.dirname(args_cli.metrics_out), exist_ok=True)
    depth_output_dir = os.path.join(TASK_DIR, "data")
    os.makedirs(depth_output_dir, exist_ok=True)
    log_file = open(args_cli.metrics_out, "w", encoding="utf-8")

    def log(message: str) -> None:
        print(message)
        log_file.write(message + "\n")
        log_file.flush()

    try:
        log("simulation_app_started")

        env_cfg = _load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
        agent_cfg = _load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
        log("configs_loaded")

        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.debug_vis = False
        env_cfg.debug_logs = args_cli.debug_logs
        env_cfg.sim.device = args_cli.device
        if args_cli.teacher_checkpoint:
            env_cfg.teacher_checkpoint = args_cli.teacher_checkpoint
        if args_cli.mode == "student":
            env_cfg.enable_teacher = False
            env_cfg.enable_teacher_metrics = False
        else:
            env_cfg.enable_teacher = True
            env_cfg.enable_teacher_metrics = True
        if hasattr(env_cfg, "auto_reset_done"):
            env_cfg.auto_reset_done = False
        agent_cfg.device = args_cli.device
        resolved_checkpoint = _resolve_checkpoint_path(args_cli.checkpoint, agent_cfg.experiment_name)

        env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        log("env_created")

        obs = env.reset()
        env.unwrapped.episode_length_buf.zero_()
        log("env_reset")

        if args_cli.mode == "student":
            runner = OnPolicyRunner(env, class_to_dict(agent_cfg), log_dir=None, device=agent_cfg.device)
            runner.load(resolved_checkpoint, load_optimizer=False)
            policy = runner.get_inference_policy(device=agent_cfg.device)
            log(f"checkpoint_loaded={resolved_checkpoint}")
        else:
            policy = None
            log("teacher_mode_enabled")

        reward_sum = 0.0
        reward_sq_sum = 0.0
        done_count = 0
        success_count = 0.0
        died_count = 0.0
        timeout_count = 0.0
        speed_sum = 0.0
        speed_xy_sum = 0.0
        action_abs_sum = torch.zeros(env.num_actions, device=args_cli.device)
        action_sq_sum = torch.zeros(env.num_actions, device=args_cli.device)
        teacher_action_abs_sum = torch.zeros(env.num_actions, device=args_cli.device)
        teacher_action_sq_sum = torch.zeros(env.num_actions, device=args_cli.device)
        teacher_student_l1_sum = 0.0
        teacher_student_l2_sum = 0.0
        final_target_distance = None
        final_obstacle_distance = None
        final_wall_distance = None

        finished_envs = torch.zeros(args_cli.num_envs, dtype=torch.bool, device=args_cli.device)
        step = 0

        with torch.inference_mode():
            while not bool(finished_envs.all().item()):
                if args_cli.max_steps > 0 and step >= args_cli.max_steps:
                    log(f"max_steps_reached={args_cli.max_steps}")
                    break

                if policy is None:
                    actions = env.unwrapped.get_teacher_actions()
                else:
                    actions = policy(obs).clone()
                actions[finished_envs] = 0.0

                obs, rew, dones, extras = env.step(actions)

                reward_sum += rew.sum().item()
                reward_sq_sum += torch.square(rew).sum().item()
                done_flags = dones.to(torch.bool)
                newly_finished = done_flags & ~finished_envs
                done_count += int(newly_finished.sum().item())

                success_flags = extras["target_reached"].to(torch.bool)
                timeout_flags = extras["time_outs"].to(torch.bool)

                success_count += float((newly_finished & success_flags).float().sum().item())
                timeout_count += float((newly_finished & timeout_flags).float().sum().item())
                died_count += float((newly_finished & ~success_flags & ~timeout_flags).float().sum().item())

                lin_vel_w = env.unwrapped.robot.data.root_lin_vel_w
                speed_sum += torch.linalg.norm(lin_vel_w, dim=1).sum().item()
                speed_xy_sum += torch.linalg.norm(lin_vel_w[:, :2], dim=1).sum().item()
                executed_actions = extras.get("final_actions", actions)
                action_abs_sum += executed_actions.abs().sum(dim=0)
                action_sq_sum += torch.square(executed_actions).sum(dim=0)

                teacher_actions = extras.get("teacher_actions")
                if teacher_actions is not None:
                    teacher_action_abs_sum += teacher_actions.abs().sum(dim=0)
                    teacher_action_sq_sum += torch.square(teacher_actions).sum(dim=0)
                    teacher_student_l1_sum += torch.abs(teacher_actions - executed_actions).sum().item()
                    teacher_student_l2_sum += torch.square(teacher_actions - executed_actions).sum().item()

                target_world = env.unwrapped._target_positions_w
                robot_pos_w = env.unwrapped.robot.data.root_pos_w
                final_target_distance = torch.linalg.norm(target_world - robot_pos_w, dim=1)
                final_obstacle_distance = env.unwrapped._compute_closest_obstacle_signed_distance(robot_pos_w)
                final_wall_distance = env.unwrapped._compute_wall_signed_distance(robot_pos_w)

                finished_envs |= done_flags
                step += 1

                if step % 20 == 0:
                    saved_path = _save_depth_snapshot(env, depth_output_dir, step)
                    if saved_path is not None:
                        log(f"depth_snapshot_saved={saved_path}")

                if step % 250 == 0:
                    mean_reward = reward_sum / (step * args_cli.num_envs)
                    log(
                        f"[student_eval] step={step} mean_step_reward={mean_reward:.4f} "
                        f"episodes_done={done_count} success={success_count:.2f} "
                        f"died={died_count:.2f} timeout={timeout_count:.2f} "
                        f"finished_envs={int(finished_envs.sum().item())}/{args_cli.num_envs}"
                    )

        total_samples = max(step * args_cli.num_envs, 1)
        mean_reward = reward_sum / total_samples
        reward_std = max(reward_sq_sum / total_samples - mean_reward**2, 0.0) ** 0.5
        avg_speed = speed_sum / total_samples
        avg_speed_xy = speed_xy_sum / total_samples
        action_abs_mean = (action_abs_sum / total_samples).tolist()
        action_rms = torch.sqrt(action_sq_sum / total_samples).tolist()
        teacher_action_abs_mean = (teacher_action_abs_sum / total_samples).tolist()
        teacher_action_rms = torch.sqrt(teacher_action_sq_sum / total_samples).tolist()

        log("=== Student Evaluation Summary ===")
        log(f"mode={args_cli.mode}")
        log(f"checkpoint={resolved_checkpoint if args_cli.mode == 'student' else args_cli.checkpoint}")
        log(f"num_envs={args_cli.num_envs}")
        log(f"steps={step}")
        log(f"episodes_done={done_count}")
        log(f"success_count={success_count:.2f}")
        log(f"died_count={died_count:.2f}")
        log(f"timeout_count={timeout_count:.2f}")
        if done_count > 0:
            log(f"success_rate={success_count / done_count:.4f}")
            log(f"died_rate={died_count / done_count:.4f}")
            log(f"timeout_rate={timeout_count / done_count:.4f}")
        log(f"mean_step_reward={mean_reward:.6f}")
        log(f"std_step_reward={reward_std:.6f}")
        log(f"avg_speed_mean={avg_speed:.4f}")
        log(f"avg_speed_xy_mean={avg_speed_xy:.4f}")
        log(f"mean_teacher_student_l1={teacher_student_l1_sum / max(total_samples * env.num_actions, 1):.6f}")
        log(f"mean_teacher_student_l2={teacher_student_l2_sum / max(total_samples * env.num_actions, 1):.6f}")
        if final_target_distance is not None:
            log(f"final_target_distance_mean={final_target_distance.mean().item():.4f}")
            log(f"final_target_distance_median={final_target_distance.median().item():.4f}")
        if final_obstacle_distance is not None:
            log(f"final_obstacle_distance_mean={final_obstacle_distance.mean().item():.4f}")
            log(f"final_obstacle_distance_median={final_obstacle_distance.median().item():.4f}")
        if final_wall_distance is not None:
            log(f"final_wall_distance_mean={final_wall_distance.mean().item():.4f}")
            log(f"final_wall_distance_median={final_wall_distance.median().item():.4f}")
        log(f"executed_action_abs_mean={action_abs_mean}")
        log(f"executed_action_rms={action_rms}")
        log(f"teacher_action_abs_mean={teacher_action_abs_mean}")
        log(f"teacher_action_rms={teacher_action_rms}")

        env.close()
    except Exception as exc:
        log(f"exception={type(exc).__name__}: {exc}")
        log(traceback.format_exc())
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
