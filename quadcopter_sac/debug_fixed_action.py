from __future__ import annotations

import argparse
import os
import sys


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


parser = argparse.ArgumentParser(description="Run a fixed action smoke test for quadcopter_sac.")
parser.add_argument("--task", type=str, default="Isaac-Quadcopter-Obstacles-SAC-v0")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--forward", type=float, default=1.0)
parser.add_argument("--lateral", type=float, default=0.0)
parser.add_argument("--vertical", type=float, default=0.0)
parser.add_argument("--print_every", type=int, default=50)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import quadcopter_sac  # noqa: F401


def _load_cfg_from_registry(task_name: str, entry_point_key: str):
    cfg_entry_point = gym.spec(task_name).kwargs.get(entry_point_key)
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
    env_cfg.device = args_cli.device if args_cli.device is not None else env_cfg.device
    env_cfg.sim.device = env_cfg.device
    env_cfg.debug_vis = not bool(getattr(args_cli, "headless", False))
    env_cfg.auto_reset_done = False

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    obs, _ = env.reset()
    del obs

    raw_env = env.unwrapped
    device = raw_env.device
    actions = torch.zeros((raw_env.num_envs, raw_env.num_actions), device=device)
    actions[:, 0] = float(args_cli.forward)
    if raw_env.num_actions > 1:
        actions[:, 1] = float(args_cli.lateral)
    if raw_env.num_actions > 2:
        actions[:, 2] = float(args_cli.vertical)

    start_pos = raw_env.robot.data.root_pos_w.clone()
    start_radius = torch.linalg.norm(start_pos[:, :2], dim=1)
    start_dist = torch.linalg.norm(raw_env._target_positions_w - start_pos, dim=1)
    print(
        f"[FIXED] start radius={start_radius.mean().item():.3f} "
        f"dist={start_dist.mean().item():.3f}",
        flush=True,
    )

    for step in range(1, int(args_cli.steps) + 1):
        _, _, terminated, truncated, _ = env.step(actions)
        pos = raw_env.robot.data.root_pos_w
        radius = torch.linalg.norm(pos[:, :2], dim=1)
        dist = torch.linalg.norm(raw_env._target_positions_w - pos, dim=1)
        target_dir = (raw_env._target_positions_w - pos) / dist.unsqueeze(1).clamp_min(1e-6)
        vel_goal = torch.sum(raw_env.robot.data.root_lin_vel_w * target_dir, dim=1)
        entered = (
            (pos[:, 0].abs() <= float(raw_env.cfg.obstacle_spawn_range))
            & (pos[:, 1].abs() <= float(raw_env.cfg.obstacle_spawn_range))
        )
        if step % int(args_cli.print_every) == 0 or step == 1 or terminated.any() or truncated.any():
            print(
                f"[FIXED] step={step:04d} radius={radius.mean().item():.3f} "
                f"dist={dist.mean().item():.3f} vel_goal={vel_goal.mean().item():+.3f} "
                f"entered={entered.float().mean().item():.3f} "
                f"done={(terminated | truncated).float().mean().item():.3f}",
                flush=True,
            )
        if terminated.any() or truncated.any():
            break

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
