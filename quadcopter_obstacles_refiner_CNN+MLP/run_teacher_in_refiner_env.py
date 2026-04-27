from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime


TASK_DIR = os.path.abspath(os.path.dirname(__file__))
ENV_DIR = os.path.abspath(os.path.join(TASK_DIR, ".."))
ROOT_DIR = os.path.abspath(os.path.join(TASK_DIR, "..", ".."))
LOCAL_RSL_RL_DIR = os.path.join(ENV_DIR, "rsl_rl")

for path in (ROOT_DIR, ENV_DIR, LOCAL_RSL_RL_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)


from isaaclab.app import AppLauncher


DEFAULT_TEACHER_CHECKPOINT = (
    "/home/wei/End_to_end/logs/rsl_rl/quadcopter_obstacles_v5/2026-03-14_14-38-38/model_3000.pt"
)


parser = argparse.ArgumentParser(
    description="Run the refiner environment with teacher actions and save front-depth images periodically."
)
parser.add_argument("--task", type=str, default="Isaac-Quadcopter-Obstacles-Refiner-v0")
parser.add_argument("--teacher_checkpoint", type=str, default=DEFAULT_TEACHER_CHECKPOINT)
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--max_steps", type=int, default=1000)
parser.add_argument("--save_depth_every", type=int, default=30)
parser.add_argument("--depth_out", type=str, default=None)
parser.add_argument("--debug_logs", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 这个环境依赖深度相机，因此即使 headless 也要启用 cameras。
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from PIL import Image

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


def main() -> None:
    print("[teacher_run] main start", flush=True)
    env_cfg = _load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.teacher_checkpoint = args_cli.teacher_checkpoint
    env_cfg.enable_teacher = True
    env_cfg.enable_teacher_metrics = True
    env_cfg.debug_logs = args_cli.debug_logs
    env_cfg.sim.device = args_cli.device
    if hasattr(env_cfg, "auto_reset_done"):
        env_cfg.auto_reset_done = True

    depth_out = args_cli.depth_out
    if depth_out is None:
        run_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        depth_out = os.path.join(ROOT_DIR, "logs", "depth_snapshots", f"teacher_refiner_{run_stamp}")
    depth_out = os.path.abspath(depth_out)
    os.makedirs(depth_out, exist_ok=True)
    print(f"[teacher_run] depth_out={depth_out}", flush=True)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    try:
        print("[teacher_run] env created", flush=True)
        obs, _ = env.reset()
        del obs
        print("[teacher_run] env reset", flush=True)

        success_count = 0.0
        died_count = 0.0
        timeout_count = 0.0
        done_count = 0
        prev_depth_frames = None

        with torch.inference_mode():
            for step in range(args_cli.max_steps):
                teacher_actions = env.unwrapped.get_teacher_actions()

                _, _, terminated, truncated, info = env.step(teacher_actions)

                if args_cli.save_depth_every > 0 and step % args_cli.save_depth_every == 0:
                    depth_frames = env.unwrapped.get_record_depth_frames().detach().cpu()
                    depth_frames_u8 = env.unwrapped.get_record_depth_frames_u8().detach().cpu()
                    robot_pos_w = env.unwrapped.robot.data.root_pos_w.detach().cpu()
                    camera_pos_w = env.unwrapped.depth_camera.data.pos_w.detach().cpu()
                    for env_index in range(depth_frames_u8.shape[0]):
                        env_dir = os.path.join(depth_out, f"env_{env_index:02d}")
                        os.makedirs(env_dir, exist_ok=True)
                        image_path = os.path.join(env_dir, f"step_{step:05d}.png")
                        Image.fromarray(depth_frames_u8[env_index].numpy()).save(image_path)
                        if prev_depth_frames is None:
                            depth_delta = 0.0
                        else:
                            depth_delta = float(
                                torch.mean(
                                    torch.abs(
                                        depth_frames[env_index].float() - prev_depth_frames[env_index].float()
                                    )
                                ).item()
                            )
                        pos_xyz = robot_pos_w[env_index].tolist()
                        cam_xyz = camera_pos_w[env_index].tolist()
                        print(
                            f"[teacher_run] saved env={env_index} step={step} "
                            f"pos=({pos_xyz[0]:.3f},{pos_xyz[1]:.3f},{pos_xyz[2]:.3f}) "
                            f"cam=({cam_xyz[0]:.3f},{cam_xyz[1]:.3f},{cam_xyz[2]:.3f}) "
                            f"depth_delta={depth_delta:.6f}",
                            flush=True,
                        )
                    prev_depth_frames = depth_frames.clone()

                done_flags = torch.as_tensor(terminated | truncated, device=args_cli.device, dtype=torch.bool)
                newly_finished = done_flags

                target_reached = torch.as_tensor(info["target_reached"], device=args_cli.device, dtype=torch.bool)
                time_outs = torch.as_tensor(info["time_outs"], device=args_cli.device, dtype=torch.bool)

                success_count += float((newly_finished & target_reached).float().sum().item())
                timeout_count += float((newly_finished & time_outs).float().sum().item())
                died_count += float((newly_finished & ~target_reached & ~time_outs).float().sum().item())
                done_count += int(newly_finished.sum().item())

                if step % 250 == 0:
                    print(
                        f"[teacher_run] step={step} episodes_done={done_count} "
                        f"success={success_count:.0f} died={died_count:.0f} timeout={timeout_count:.0f} "
                        f"done_this_step={int(done_flags.sum().item())}/{args_cli.num_envs}",
                        flush=True,
                    )

        print("=== Teacher Run Summary ===", flush=True)
        print(f"teacher_checkpoint={args_cli.teacher_checkpoint}", flush=True)
        print(f"num_envs={args_cli.num_envs}", flush=True)
        print(f"depth_out={depth_out}", flush=True)
        print(f"episodes_done={done_count}", flush=True)
        if done_count > 0:
            print(f"success_rate={success_count / done_count:.4f}", flush=True)
            print(f"died_rate={died_count / done_count:.4f}", flush=True)
            print(f"timeout_rate={timeout_count / done_count:.4f}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[teacher_run] exception={type(exc).__name__}: {exc}", flush=True)
        raise
    finally:
        simulation_app.close()
