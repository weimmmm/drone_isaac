from __future__ import annotations

import argparse
import os
import sys


TASK_DIR = os.path.abspath(os.path.dirname(__file__))
ENV_DIR = os.path.abspath(os.path.join(TASK_DIR, ".."))
ROOT_DIR = os.path.abspath(os.path.join(TASK_DIR, "..", ".."))
LOCAL_RSL_RL_DIR = os.path.join(ENV_DIR, "rsl_rl")

for path in (ROOT_DIR, ENV_DIR, LOCAL_RSL_RL_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)


from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Export one normalized front-depth frame to inspect the camera view.")
parser.add_argument("--task", type=str, default="Isaac-Quadcopter-Obstacles-Refiner-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--env_index", type=int, default=0)
parser.add_argument("--output", type=str, required=True)
parser.add_argument("--debug_logs", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 深度图导出需要真实启用相机渲染。
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
from PIL import Image

import quadcopter_obstacles_refiner  # noqa: F401
import quadcopter_obstacles_teacher  # noqa: F401


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
    env_cfg = _load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.debug_logs = args_cli.debug_logs

    # 这里只做视角对比，不需要 teacher 指标干扰。
    if hasattr(env_cfg, "enable_teacher_metrics"):
        env_cfg.enable_teacher_metrics = False

    env = gym.make(args_cli.task, cfg=env_cfg)
    try:
        env.reset()

        unwrapped = env.unwrapped
        if not hasattr(unwrapped, "get_record_depth_frames_u8"):
            raise RuntimeError(f"Task '{args_cli.task}' does not expose get_record_depth_frames_u8().")

        depth_u8 = unwrapped.get_record_depth_frames_u8()
        env_index = max(0, min(int(args_cli.env_index), depth_u8.shape[0] - 1))
        frame = depth_u8[env_index].detach().cpu().numpy().astype(np.uint8)

        output_path = os.path.abspath(args_cli.output)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        Image.fromarray(frame, mode="L").save(output_path)

        print(f"[INFO] Saved depth snapshot to: {output_path}")
        print(f"[INFO] Task={args_cli.task} env_index={env_index} shape={frame.shape}")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
