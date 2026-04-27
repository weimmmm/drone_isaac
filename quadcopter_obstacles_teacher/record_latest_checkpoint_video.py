from __future__ import annotations

import argparse
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


TASK_DIR = os.path.abspath(os.path.dirname(__file__))
ENV_DIR = os.path.abspath(os.path.join(TASK_DIR, ".."))
ROOT_DIR = os.path.abspath(os.path.join(TASK_DIR, "..", ".."))
LOCAL_RSL_RL_DIR = os.path.join(ROOT_DIR, "rsl_rl")

for path in (ROOT_DIR, ENV_DIR, LOCAL_RSL_RL_DIR, TASK_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from isaaclab.app import AppLauncher


def _default_watch_dir() -> str:
    return os.path.join(ROOT_DIR, "logs", "rsl_rl", "quadcopter_obstacles_v5_lstm")


parser = argparse.ArgumentParser(description="Watch latest teacher checkpoint and record rollout videos.")
parser.add_argument("--task", type=str, default="Isaac-Quadcopter-Obstacles-Teacher-v0")
parser.add_argument("--watch_dir", type=str, default=_default_watch_dir())
parser.add_argument("--load_run", type=str, default=None, help="Optional specific run directory name under watch_dir.")
parser.add_argument("--checkpoint_regex", type=str, default=r"model_\d+\.pt")
parser.add_argument("--poll_interval", type=float, default=60.0)
parser.add_argument("--video_dir", type=str, default=os.path.join(TASK_DIR, "videos", "watch"))
parser.add_argument("--steps", type=int, default=1200)
parser.add_argument("--fps", type=int, default=25)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--max_new_checkpoints", type=int, default=0, help="0 means keep watching forever.")
parser.add_argument("--once", action="store_true", default=False, help="Record only the current latest checkpoint once.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab.utils.dict import class_to_dict

from local_rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

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


def _find_latest_checkpoint(watch_dir: str, run_name: str | None, checkpoint_regex: str) -> tuple[str, str]:
    if run_name is not None:
        run_path = os.path.join(watch_dir, run_name)
        if not os.path.isdir(run_path):
            raise FileNotFoundError(f"Run directory does not exist: {run_path}")
    else:
        run_dirs = [entry.path for entry in os.scandir(watch_dir) if entry.is_dir()]
        if not run_dirs:
            raise FileNotFoundError(f"No run directories found under: {watch_dir}")
        run_dirs.sort(key=os.path.getmtime)
        run_path = run_dirs[-1]

    pattern = re.compile(checkpoint_regex)
    checkpoints = [name for name in os.listdir(run_path) if pattern.fullmatch(name)]
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints matching '{checkpoint_regex}' in: {run_path}")
    checkpoints.sort(key=lambda name: int(re.findall(r"\d+", name)[-1]))
    ckpt_name = checkpoints[-1]
    return run_path, os.path.join(run_path, ckpt_name)


def _make_video_writer(video_path: str, frame_shape: tuple[int, int, int], fps: int) -> cv2.VideoWriter:
    height, width = frame_shape[0], frame_shape[1]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {video_path}")
    return writer


def _record_checkpoint_video(model_path: str, run_path: str) -> str:
    env_cfg = _load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    agent_cfg = _load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    if hasattr(env_cfg, "auto_reset_done"):
        env_cfg.auto_reset_done = False
    agent_cfg.device = args_cli.device

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    obs = env.reset()

    runner = OnPolicyRunner(env, class_to_dict(agent_cfg), log_dir=None, device=agent_cfg.device)
    runner.load(model_path, load_optimizer=False)
    policy = runner.get_inference_policy(device=agent_cfg.device)

    rel_run = os.path.basename(run_path)
    ckpt_stem = Path(model_path).stem
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = Path(args_cli.video_dir) / rel_run
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = str(out_dir / f"{ckpt_stem}_{timestamp}.mp4")

    writer = None
    try:
        with torch.inference_mode():
            for _ in range(args_cli.steps):
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
                frame = env.unwrapped.render()
                if frame is None:
                    continue
                frame_bgr = cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR)
                if writer is None:
                    writer = _make_video_writer(video_path, frame_bgr.shape, args_cli.fps)
                writer.write(frame_bgr)
                if bool(torch.all(dones).item()):
                    break
    finally:
        if writer is not None:
            writer.release()
        env.close()

    return video_path


def main() -> None:
    os.makedirs(args_cli.video_dir, exist_ok=True)
    os.chdir(ROOT_DIR)

    seen_checkpoints: set[str] = set()
    recorded = 0

    while simulation_app.is_running():
        try:
            run_path, model_path = _find_latest_checkpoint(args_cli.watch_dir, args_cli.load_run, args_cli.checkpoint_regex)
            if model_path not in seen_checkpoints:
                print(f"[watch] recording checkpoint: {model_path}", flush=True)
                video_path = _record_checkpoint_video(model_path, run_path)
                print(f"[watch] saved video: {video_path}", flush=True)
                seen_checkpoints.add(model_path)
                recorded += 1
                if args_cli.once or (args_cli.max_new_checkpoints > 0 and recorded >= args_cli.max_new_checkpoints):
                    break
            else:
                print(f"[watch] no new checkpoint under: {run_path}", flush=True)
        except Exception as exc:
            print(f"[watch] warning: {type(exc).__name__}: {exc}", flush=True)
            if args_cli.once:
                raise

        time.sleep(max(args_cli.poll_interval, 1.0))


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
