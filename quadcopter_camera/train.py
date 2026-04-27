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

for path in (ROOT_DIR, ENV_DIR, LOCAL_RSL_RL_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)


from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Train quadcopter obstacles with the vendored rsl_rl library.")
parser.set_defaults(video=True)
parser.add_argument("--video", dest="video", action="store_true")
parser.add_argument("--no_video", dest="video", action="store_false")
parser.add_argument("--video_length", type=int, default=500)
parser.add_argument("--video_interval_iterations", type=int, default=200)
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--task", type=str, default="Isaac-Quadcopter-Camera-SingleWorld-v0")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--resume", action="store_true", default=False)
parser.add_argument("--load_run", type=str, default=None)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--experiment_name", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# This task always requires cameras because the policy consumes depth images.
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab.utils.dict import class_to_dict, print_dict
from isaaclab.utils.io import dump_yaml

from local_rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

import quadcopter_camera  # noqa: F401


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
    print("[DEBUG] Loaded env_cfg and agent_cfg", flush=True)

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    if args_cli.seed is not None:
        agent_cfg.seed = args_cli.seed

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if args_cli.experiment_name:
        agent_cfg.experiment_name = args_cli.experiment_name
    if args_cli.resume:
        agent_cfg.resume = True
    if args_cli.load_run:
        agent_cfg.load_run = args_cli.load_run
    if args_cli.checkpoint:
        agent_cfg.load_checkpoint = args_cli.checkpoint

    log_root_path = os.path.abspath(os.path.join(ROOT_DIR, "logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)

    print("[DEBUG] Creating gym environment", flush=True)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    print("[DEBUG] Gym environment created", flush=True)

    if agent_cfg.resume:
        resume_path = _find_checkpoint(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
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
        print("[INFO] Recording videos during training.", flush=True)
        print(f"[INFO] Saving one training video every {args_cli.video_interval_iterations} learning iterations.", flush=True)
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    print("[DEBUG] Wrapping environment for local rsl_rl", flush=True)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    print("[DEBUG] Wrapped environment", flush=True)
    print("[DEBUG] Initializing OnPolicyRunner", flush=True)
    runner = OnPolicyRunner(env, class_to_dict(agent_cfg), log_dir=log_dir, device=agent_cfg.device)
    print("[DEBUG] OnPolicyRunner initialized", flush=True)
    runner.add_git_repo_to_log(__file__)

    if resume_path:
        runner.load(resume_path)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    with open(os.path.join(log_dir, "params", "env.pkl"), "wb") as file:
        pickle.dump(env_cfg, file)
    with open(os.path.join(log_dir, "params", "agent.pkl"), "wb") as file:
        pickle.dump(agent_cfg, file)

    print("[DEBUG] Starting runner.learn", flush=True)
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    print("[DEBUG] runner.learn returned", flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
