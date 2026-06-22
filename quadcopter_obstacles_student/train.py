from __future__ import annotations

import argparse
import os
import pickle
import random
import re
import sys
import faulthandler
from datetime import datetime


DEFAULT_REPRO_SEED = 3407
TASK_DIR = os.path.abspath(os.path.dirname(__file__))
ENV_DIR = os.path.abspath(os.path.join(TASK_DIR, ".."))
ROOT_DIR = os.path.abspath(os.path.join(TASK_DIR, "..", ".."))
LOCAL_RSL_RL_DIR = os.path.join(ENV_DIR, "rsl_rl")
ISAACLAB_ROOT = os.environ.get("ISAACLAB_PATH", "/home/wei/IsaacLab")
ISAACLAB_SOURCE_DIRS = [
    os.path.join(ISAACLAB_ROOT, "source", name)
    for name in ("isaaclab", "isaaclab_assets", "isaaclab_rl", "isaaclab_tasks")
]

for path in (ROOT_DIR, ENV_DIR, LOCAL_RSL_RL_DIR, *ISAACLAB_SOURCE_DIRS):
    if path not in sys.path:
        sys.path.insert(0, path)


def _ensure_conda_lib_first() -> None:
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        return

    conda_lib = os.path.join(conda_prefix, "lib")
    ld_paths = os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)
    if ld_paths and ld_paths[0] == conda_lib:
        return

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = os.pathsep.join([conda_lib, *[path for path in ld_paths if path]])
    env["QUADCOPTER_TRAIN_REEXEC"] = "1"
    os.execvpe(sys.executable, [sys.executable, *sys.argv], env)


_ensure_conda_lib_first()


def _bootstrap_repro_env(seed: int) -> None:
    """Set process-level reproducibility env vars before Isaac/Torch imports."""
    required_env = {
        "PYTHONHASHSEED": str(seed),
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    }
    desired_env = os.environ.copy()
    python_paths = [path for path in (ROOT_DIR, ENV_DIR, LOCAL_RSL_RL_DIR, *ISAACLAB_SOURCE_DIRS) if os.path.isdir(path)]
    existing_pythonpath = desired_env.get("PYTHONPATH", "")
    desired_env["PYTHONPATH"] = os.pathsep.join(
        [*python_paths, *[path for path in existing_pythonpath.split(os.pathsep) if path]]
    )
    desired_env.setdefault("PYTHONUNBUFFERED", "1")
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
    if desired_env.get("PYTHONUNBUFFERED") != os.environ.get("PYTHONUNBUFFERED"):
        changed.append("PYTHONUNBUFFERED=1")

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


from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Train quadcopter obstacles with the vendored rsl_rl library.")
parser.set_defaults(video= False) 
parser.add_argument("--video", dest="video", action="store_true")
parser.add_argument("--no_video", dest="video", action="store_false")
parser.add_argument("--video_length", type=int, default=250)
parser.add_argument("--video_interval_iterations", type=int, default=200)
parser.add_argument("--num_envs", type=int, default=8190)
parser.add_argument("--task", type=str, default="Isaac-Quadcopter-Obstacles-Student-v0")
parser.add_argument("--seed", type=int, default=DEFAULT_REPRO_SEED)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--resume", action="store_true", default=False)
parser.add_argument("--load_run", type=str, default=None)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--experiment_name", type=str, default=None)
parser.add_argument("--cfg", type=str, default=os.path.join(TASK_DIR, "cfg", "train.yaml"))
parser.add_argument("--env_cfg_dir", type=str, default=None)
parser.add_argument("--logger", type=str, choices=["tensorboard", "wandb", "neptune"], default=None)
parser.add_argument("--wandb_project", type=str, default=None)
parser.add_argument("--wandb_name", type=str, default=None)
parser.add_argument("--wandb_entity", type=str, default=None)
parser.add_argument("--wandb_mode", type=str, default=None)
parser.add_argument("--wandb_run_id", type=str, default=None)
parser.add_argument("--debug_hang_trace_s", type=float, default=0.0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
_bootstrap_repro_env(args_cli.seed)

if args_cli.video:
    args_cli.enable_cameras = True
if not args_cli.headless and not args_cli.experience:
    # Avoid full editor test discovery in the default GUI experience.
    args_cli.experience = "isaaclab.python.rendering.kit"

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

from isaaclab.utils.dict import class_to_dict, print_dict
from isaaclab.utils.io import dump_yaml

from local_rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

import quadcopter_obstacles_student  # noqa: F401
from quadcopter_obstacles_student.config_utils import apply_env_cfg_dir, load_yaml_cfg


def _configure_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.set_float32_matmul_precision("highest")
    except Exception:
        pass
    torch.use_deterministic_algorithms(False)


_configure_determinism(args_cli.seed)


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


def _apply_wandb_cfg(agent_cfg, wandb_cfg: dict) -> None:
    if not wandb_cfg:
        return

    project = wandb_cfg.get("project")
    name = wandb_cfg.get("name")
    entity = wandb_cfg.get("entity")
    mode = wandb_cfg.get("mode")
    run_id = wandb_cfg.get("run_id")

    agent_cfg.logger = "wandb"
    if project:
        agent_cfg.wandb_project = project
    if name:
        agent_cfg.run_name = name
    if entity:
        os.environ["WANDB_USERNAME"] = str(entity)
    if mode:
        # YAML is the default source of truth; explicit CLI flags are applied later.
        os.environ["WANDB_MODE"] = str(mode)
    if run_id:
        os.environ["WANDB_RUN_ID"] = str(run_id)
        os.environ.setdefault("WANDB_RESUME", "allow")


def main():
    if args_cli.debug_hang_trace_s > 0.0:
        faulthandler.enable()
        faulthandler.dump_traceback_later(args_cli.debug_hang_trace_s, repeat=True)

    yaml_cfg = load_yaml_cfg(args_cli.cfg)

    env_cfg = _load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    agent_cfg = _load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    env_cfg_dir = args_cli.env_cfg_dir or os.path.dirname(args_cli.cfg)
    env_cfg_paths = apply_env_cfg_dir(env_cfg, env_cfg_dir)
    _apply_wandb_cfg(agent_cfg, yaml_cfg.get("wandb", {}))

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    agent_cfg.seed = args_cli.seed

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if args_cli.experiment_name:
        agent_cfg.experiment_name = args_cli.experiment_name
    if args_cli.logger:
        agent_cfg.logger = args_cli.logger
    if args_cli.wandb_project:
        agent_cfg.wandb_project = args_cli.wandb_project
    if args_cli.wandb_name:
        agent_cfg.run_name = args_cli.wandb_name
    if args_cli.wandb_entity:
        os.environ["WANDB_USERNAME"] = args_cli.wandb_entity
    if args_cli.wandb_mode:
        os.environ["WANDB_MODE"] = args_cli.wandb_mode
    if args_cli.wandb_run_id:
        os.environ["WANDB_RUN_ID"] = args_cli.wandb_run_id
        os.environ.setdefault("WANDB_RESUME", "allow")
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
    print(f"[INFO] Logger: {agent_cfg.logger}")
    if args_cli.cfg:
        print(f"[INFO] Config file: {args_cli.cfg}")
    if env_cfg_paths:
        print("[INFO] Environment config files:")
        for path in env_cfg_paths:
            print(f"  - {path}")
    if agent_cfg.logger == "wandb":
        print(f"[INFO] WandB project: {agent_cfg.wandb_project}")
        if agent_cfg.run_name:
            print(f"[INFO] WandB run name: {agent_cfg.run_name}")
        if os.environ.get("WANDB_USERNAME"):
            print(f"[INFO] WandB entity: {os.environ['WANDB_USERNAME']}")
        if os.environ.get("WANDB_MODE"):
            print(f"[INFO] WandB mode: {os.environ['WANDB_MODE']}")
        if os.environ.get("WANDB_RUN_ID"):
            print(f"[INFO] WandB run id: {os.environ['WANDB_RUN_ID']}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    env.unwrapped.seed(agent_cfg.seed)

    if agent_cfg.resume:
        resume_path = _find_checkpoint(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
    else:
        resume_path = None

    if args_cli.video:
        print("[INFO] Recording videos during evaluation only.")
        print(f"[INFO] Evaluation video length: {args_cli.video_length} frames.")
        print(f"[INFO] Evaluation videos will be saved to: {os.path.join(log_dir, 'videos', 'eval')}")

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner_cfg = class_to_dict(agent_cfg)
    runner_cfg["evaluation"] = yaml_cfg.get("evaluation", {})
    if args_cli.video:
        runner_cfg["evaluation"] = dict(runner_cfg["evaluation"])
        runner_cfg["evaluation"]["record_video"] = True
        runner_cfg["evaluation"]["video_length"] = args_cli.video_length
        runner_cfg["evaluation"]["video_interval"] = 1
        runner_cfg["evaluation"]["video_fps"] = int(round(1.0 / env.unwrapped.step_dt))
        runner_cfg["evaluation"]["video_dir"] = os.path.join(log_dir, "videos", "eval")

    runner = OnPolicyRunner(env, runner_cfg, log_dir=log_dir, device=agent_cfg.device)
    runner.add_git_repo_to_log(__file__)

    if resume_path:
        runner.load(resume_path)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    with open(os.path.join(log_dir, "params", "env.pkl"), "wb") as file:
        pickle.dump(env_cfg, file)
    with open(os.path.join(log_dir, "params", "agent.pkl"), "wb") as file:
        pickle.dump(agent_cfg, file)

    try:
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    finally:
        if args_cli.debug_hang_trace_s > 0.0:
            faulthandler.cancel_dump_traceback_later()
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
