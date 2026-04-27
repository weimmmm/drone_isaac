from __future__ import annotations

import argparse
import os
import pickle
import re
import sys
from datetime import datetime


TASK_DIR = os.path.abspath(os.path.dirname(__file__))
ENV_DIR = os.path.abspath(os.path.join(TASK_DIR, ".."))
ROOT_DIR = os.path.abspath(os.path.join(TASK_DIR, "..", ".."))
LOCAL_RSL_RL_DIR = os.path.join(ENV_DIR, "rsl_rl")
ISAACLAB_PATH = os.environ.get("ISAACLAB_PATH", "/home/wei/IsaacLab")

ISAACLAB_SOURCE_DIRS = (
    os.path.join(ISAACLAB_PATH, "source", "isaaclab"),
    os.path.join(ISAACLAB_PATH, "source", "isaaclab_assets"),
    os.path.join(ISAACLAB_PATH, "source", "isaaclab_rl"),
    os.path.join(ISAACLAB_PATH, "source", "isaaclab_tasks"),
)

for path in (ROOT_DIR, ENV_DIR, LOCAL_RSL_RL_DIR, *ISAACLAB_SOURCE_DIRS):
    if path not in sys.path:
        sys.path.insert(0, path)


from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Train quadcopter obstacle navigation with rsl_rl SAC.")
parser.set_defaults(video=True)
parser.add_argument("--video", dest="video", action="store_true")
parser.add_argument("--no_video", dest="video", action="store_false")
parser.add_argument("--video_length", type=int, default=1000)
parser.add_argument("--video_interval_iterations", type=int, default=50)
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--task", type=str, default="Isaac-Quadcopter-Obstacles-SAC-v0")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=5000)
parser.add_argument("--resume", action="store_true", default=False)
parser.add_argument("--load_run", type=str, default=None)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--experiment_name", type=str, default=None)
parser.add_argument("--random_steps", type=int, default=None)
parser.add_argument("--learning_starts", type=int, default=None)
parser.add_argument("--batch_size", type=int, default=None)
parser.add_argument("--replay_buffer_size", type=int, default=None)
parser.add_argument("--gradient_steps_per_iteration", type=int, default=None)
parser.add_argument("--disable_alpha_autotune", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab.utils.dict import class_to_dict, print_dict
from isaaclab.utils.io import dump_yaml

from local_rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OffPolicyRunner

import quadcopter_sac  # noqa: F401


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


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


def _find_checkpoint(log_root: str, run_name: str, checkpoint_name: str) -> str:
    if not os.path.isdir(log_root):
        raise ValueError(f"Log directory does not exist: {log_root}")

    matched_runs = [entry.path for entry in os.scandir(log_root) if entry.is_dir() and re.match(run_name, entry.name)]
    if not matched_runs:
        raise ValueError(f"No runs in '{log_root}' match '{run_name}'.")
    matched_runs.sort()
    run_path = matched_runs[-1]

    checkpoints = [name for name in os.listdir(run_path) if re.match(checkpoint_name, name)]
    if not checkpoints:
        raise ValueError(f"No checkpoints in '{run_path}' match '{checkpoint_name}'.")
    checkpoints.sort(key=lambda name: f"{name:0>15}")
    return os.path.join(run_path, checkpoints[-1])


def main():
    env_cfg = _load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    agent_cfg = _load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    print("[INFO] Loaded SAC env_cfg and agent_cfg", flush=True)

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    if args_cli.seed is not None:
        agent_cfg.seed = args_cli.seed
    if args_cli.random_steps is not None:
        agent_cfg.random_steps = args_cli.random_steps
    if args_cli.learning_starts is not None:
        agent_cfg.learning_starts = args_cli.learning_starts
    if args_cli.batch_size is not None:
        agent_cfg.algorithm.batch_size = args_cli.batch_size
    if args_cli.replay_buffer_size is not None:
        agent_cfg.replay_buffer_size = args_cli.replay_buffer_size
    if args_cli.gradient_steps_per_iteration is not None:
        agent_cfg.gradient_steps_per_iteration = args_cli.gradient_steps_per_iteration
    if args_cli.disable_alpha_autotune:
        agent_cfg.algorithm.autotune_alpha = False

    agent_cfg.device = args_cli.device if args_cli.device is not None else agent_cfg.device
    env_cfg.seed = agent_cfg.seed
    env_cfg.device = agent_cfg.device
    env_cfg.sim.device = agent_cfg.device
    env_cfg.auto_reset_done = True
    env_cfg.debug_vis = not bool(getattr(args_cli, "headless", False))

    if args_cli.experiment_name:
        agent_cfg.experiment_name = args_cli.experiment_name
    if args_cli.resume:
        agent_cfg.resume = True
    if args_cli.load_run:
        agent_cfg.load_run = args_cli.load_run
    if args_cli.checkpoint:
        agent_cfg.load_checkpoint = args_cli.checkpoint

    log_root_path = os.path.abspath(os.path.join(ROOT_DIR, "logs", "rsl_rl", agent_cfg.experiment_name))
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
    print(f"[INFO] Logging experiment in directory: {log_dir}", flush=True)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if agent_cfg.resume:
        resume_path = _find_checkpoint(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO] Loading SAC checkpoint from: {resume_path}", flush=True)
    else:
        resume_path = None

    if args_cli.video:
        video_interval_steps = args_cli.video_interval_iterations * agent_cfg.num_steps_per_env
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % video_interval_steps == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OffPolicyRunner(env, class_to_dict(agent_cfg), log_dir=log_dir, device=agent_cfg.device)
    runner.add_git_repo_to_log(__file__)

    if resume_path:
        runner.load(resume_path)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    with open(os.path.join(log_dir, "params", "env.pkl"), "wb") as file:
        pickle.dump(env_cfg, file)
    with open(os.path.join(log_dir, "params", "agent.pkl"), "wb") as file:
        pickle.dump(agent_cfg, file)

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
