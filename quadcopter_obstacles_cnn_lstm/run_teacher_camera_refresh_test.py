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
    description="Run the CNN-LSTM refiner environment with teacher actions and verify depth-camera refresh in headless mode."
)
parser.add_argument("--task", type=str, default="Isaac-Quadcopter-Obstacles-CNN-LSTM-v0")
parser.add_argument("--teacher_checkpoint", type=str, default=DEFAULT_TEACHER_CHECKPOINT)
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--max_steps", type=int, default=300)
parser.add_argument("--sample_every", type=int, default=10)
parser.add_argument("--save_images", action="store_true", default=False)
parser.add_argument("--depth_out", type=str, default=None)
parser.add_argument("--debug_logs", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# The task depends on the depth camera, so camera rendering must stay enabled even in headless mode.
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from PIL import Image

import quadcopter_obstacles_cnn_lstm  # noqa: F401


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


def _prepare_output_dir() -> str:
    if args_cli.depth_out is not None:
        depth_out = os.path.abspath(args_cli.depth_out)
    else:
        run_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        depth_out = os.path.join(ROOT_DIR, "logs", "depth_snapshots", f"cnn_lstm_camera_refresh_{run_stamp}")
    os.makedirs(depth_out, exist_ok=True)
    return depth_out


def main() -> None:
    print("[camera_test] main start", flush=True)
    env_cfg = _load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.teacher_checkpoint = args_cli.teacher_checkpoint
    env_cfg.enable_teacher = True
    env_cfg.enable_teacher_metrics = True
    env_cfg.debug_logs = args_cli.debug_logs
    env_cfg.sim.device = args_cli.device
    if hasattr(env_cfg, "auto_reset_done"):
        env_cfg.auto_reset_done = True

    depth_out = _prepare_output_dir()
    print(f"[camera_test] depth_out={depth_out}", flush=True)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    try:
        print("[camera_test] env created", flush=True)
        obs, _ = env.reset()
        del obs
        print("[camera_test] env reset", flush=True)

        prev_depth = None
        changed_samples = 0
        sampled_steps = 0

        with torch.inference_mode():
            for step in range(args_cli.max_steps):
                teacher_actions = env.unwrapped.get_teacher_actions()
                env.step(teacher_actions)

                if args_cli.sample_every > 0 and step % args_cli.sample_every == 0:
                    depth_frames = env.unwrapped.get_record_depth_frames().detach().cpu()
                    depth_mean = float(depth_frames.mean().item())
                    depth_min = float(depth_frames.min().item())
                    depth_max = float(depth_frames.max().item())

                    if prev_depth is None:
                        mean_abs_delta = 0.0
                        max_abs_delta = 0.0
                    else:
                        delta = torch.abs(depth_frames.float() - prev_depth.float())
                        mean_abs_delta = float(delta.mean().item())
                        max_abs_delta = float(delta.max().item())
                        if mean_abs_delta > 1e-5 or max_abs_delta > 1e-4:
                            changed_samples += 1

                    sampled_steps += 1
                    print(
                        f"[camera_test] step={step:04d} "
                        f"depth_mean={depth_mean:.6f} depth_min={depth_min:.6f} depth_max={depth_max:.6f} "
                        f"mean_abs_delta={mean_abs_delta:.6f} max_abs_delta={max_abs_delta:.6f}",
                        flush=True,
                    )

                    if args_cli.save_images:
                        depth_u8 = env.unwrapped.get_record_depth_frames_u8().detach().cpu()
                        for env_index in range(depth_u8.shape[0]):
                            env_dir = os.path.join(depth_out, f"env_{env_index:02d}")
                            os.makedirs(env_dir, exist_ok=True)
                            image_path = os.path.join(env_dir, f"step_{step:05d}.png")
                            Image.fromarray(depth_u8[env_index].numpy()).save(image_path)

                    prev_depth = depth_frames.clone()

        print("=== Camera Refresh Summary ===", flush=True)
        print(f"teacher_checkpoint={args_cli.teacher_checkpoint}", flush=True)
        print(f"num_envs={args_cli.num_envs}", flush=True)
        print(f"max_steps={args_cli.max_steps}", flush=True)
        print(f"sample_every={args_cli.sample_every}", flush=True)
        print(f"sampled_steps={sampled_steps}", flush=True)
        print(f"changed_samples={changed_samples}", flush=True)
        if sampled_steps > 1:
            print(f"changed_ratio={changed_samples / (sampled_steps - 1):.4f}", flush=True)
        print("[camera_test] If mean_abs_delta/max_abs_delta stay near zero for all samples, the camera is likely not refreshing.", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[camera_test] exception={type(exc).__name__}: {exc}", flush=True)
        raise
    finally:
        simulation_app.close()
