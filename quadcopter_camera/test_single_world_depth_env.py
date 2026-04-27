from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


TASK_DIR = os.path.abspath(os.path.dirname(__file__))
ENV_DIR = os.path.abspath(os.path.join(TASK_DIR, ".."))
ROOT_DIR = os.path.abspath(os.path.join(TASK_DIR, "..", ".."))

for path in (ROOT_DIR, ENV_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)


from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Smoke test the single-world depth environment.")
parser.add_argument("--num_robots", type=int, default=8)
parser.add_argument("--num_obstacles", type=int, default=20)
parser.add_argument("--steps", type=int, default=20)
parser.add_argument("--save_every", type=int, default=20)
parser.add_argument("--verify_robot_occlusion", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import matplotlib.pyplot as plt

from quadcopter_camera.quadcopter_single_world_depth_env import (
    QuadcopterSingleWorldDepthEnv,
    QuadcopterSingleWorldDepthEnvCfg,
)


def _run_robot_occlusion_check(env: QuadcopterSingleWorldDepthEnv, output_root: Path) -> None:
    """Place robots in a deterministic layout and save depth images for inspection."""
    verify_dir = output_root / "robot_occlusion_check"
    verify_dir.mkdir(parents=True, exist_ok=True)

    root_state = env.robot.data.default_root_state.clone()
    root_state[:, :3] = torch.tensor(
        [
            [0.0, 0.0, 1.2],
            [2.0, 0.0, 1.2],
            [-10.0, -10.0, 1.2],
            [-10.0, 10.0, 1.2],
            [10.0, -10.0, 1.2],
            [10.0, 10.0, 1.2],
            [-15.0, 0.0, 1.2],
            [15.0, 0.0, 1.2],
        ][: env.num_robots],
        device=env.device,
    )
    root_state[:, 3] = 1.0
    root_state[:, 4:7] = 0.0
    root_state[:, 7:] = 0.0
    env.robot.write_root_state_to_sim(root_state)
    env.robot.write_joint_state_to_sim(env.robot.data.default_joint_pos, env.robot.data.default_joint_vel)

    env.target_positions_w[:] = root_state[:, :3]
    env.target_positions_w[:, 0] += 10.0
    env.target_reached[:] = False
    env.done_buf[:] = False
    env.rew_buf[:] = 0.0
    env.episode_length_buf[:] = 0
    env.cmd_vel_b[:] = 0.0
    env.actions[:] = 0.0
    env.depth_stack_needs_fill[:] = True
    env.prev_dist_to_target[:] = torch.linalg.norm(env.target_positions_w - root_state[:, :3], dim=1)

    controller_state = {
        "position": root_state[:, :3].clone(),
        "attitude": torch.zeros((env.num_robots, 3), device=env.device),
    }
    env._controller.reset(controller_state, env_ids=torch.arange(env.num_robots, device=env.device))

    for _ in range(5):
        env.sim.step()
        env.robot.update(env.cfg.physics_dt)
    env._update_depth_stack()
    obs = env._get_observations()

    depth_batch = obs["depth"][:, -1].detach().cpu().numpy()
    for robot_idx in range(env.num_robots):
        plt.imsave(
            verify_dir / f"robot_{robot_idx:02d}.png",
            depth_batch[robot_idx],
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
        )
    print(f"[TEST] occlusion check images saved to: {verify_dir}", flush=True)
    print("[TEST] robot_00 is looking toward +X, robot_01 is placed directly ahead at x=2.0", flush=True)


def main() -> None:
    cfg = QuadcopterSingleWorldDepthEnvCfg(
        num_robots=args_cli.num_robots,
        num_obstacles=args_cli.num_obstacles,
        device=args_cli.device if args_cli.device is not None else "cuda:0",
    )
    env = QuadcopterSingleWorldDepthEnv(cfg)
    output_root = Path("/home/wei/End_to_end/drone_isaac/quadcopter_camera/depth_frames")
    output_root.mkdir(parents=True, exist_ok=True)
    robot_dirs = []
    for robot_idx in range(cfg.num_robots):
        robot_dir = output_root / f"robot_{robot_idx:02d}"
        robot_dir.mkdir(parents=True, exist_ok=True)
        robot_dirs.append(robot_dir)

    print("[TEST] building single-world env", flush=True)
    env.build()

    if args_cli.verify_robot_occlusion:
        _run_robot_occlusion_check(env, output_root)

    print("[TEST] resetting env", flush=True)
    obs = env.reset()
    print(f"[TEST] state shape: {tuple(obs['state'].shape)}", flush=True)
    print(f"[TEST] depth shape: {tuple(obs['depth'].shape)}", flush=True)
    print(f"[TEST] depth mean: {obs['depth'].mean().item():.6f}", flush=True)

    actions = torch.zeros((cfg.num_robots, cfg.action_space), device=env.device)
    for step in range(args_cli.steps):
        obs, rew, done, extras = env.step(actions)
        if step % args_cli.save_every == 0:
            depth_batch = obs["depth"][:, -1].detach().cpu().numpy()
            for robot_idx in range(cfg.num_robots):
                plt.imsave(
                    robot_dirs[robot_idx] / f"step_{step:04d}.png",
                    depth_batch[robot_idx],
                    cmap="gray",
                    vmin=0.0,
                    vmax=1.0,
                )
        print(
            f"[TEST] step={step:03d} "
            f"rew_mean={rew.mean().item():.6f} "
            f"done_count={int(done.sum().item())} "
            f"depth_mean={obs['depth'].mean().item():.6f}",
            flush=True,
        )

    env.close()
    print(f"[TEST] completed, depth frames saved to: {output_root}", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
