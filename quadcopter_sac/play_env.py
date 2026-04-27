from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

import torch


TASK_DIR = os.path.abspath(os.path.dirname(__file__))
ENV_DIR = os.path.abspath(os.path.join(TASK_DIR, ".."))
ROOT_DIR = os.path.abspath(os.path.join(TASK_DIR, "..", ".."))
ISAACLAB_PATH = os.environ.get("ISAACLAB_PATH")

for path in (ROOT_DIR, ENV_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

if ISAACLAB_PATH is not None:
    isaaclab_source = os.path.join(ISAACLAB_PATH, "source", "isaaclab")
    if isaaclab_source not in sys.path:
        sys.path.insert(0, isaaclab_source)


def build_parser():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Play the quadcopter obstacle environment with fixed forward commands.")
    parser.add_argument("--num_envs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--forward_cmd", type=float, default=0.6)
    parser.add_argument("--lateral_cmd", type=float, default=0.0)
    parser.add_argument("--vertical_cmd", type=float, default=0.0)
    parser.add_argument("--num_rays", type=int, default=35)
    parser.add_argument("--sleep", type=float, default=0.02)
    parser.add_argument("--debug_interval", type=int, default=20)
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _get_debug_draw():
    try:
        from isaacsim.util.debug_draw import _debug_draw

        return _debug_draw.acquire_debug_draw_interface()
    except Exception:
        return None


def _compute_front_rays(env, num_rays: int):
    # 仿照 NavRL：直接使用 RayCaster 的真实世界系命中点表示雷达，而不是按 yaw+angle 重建端点。
    if env._lidar is not None and env._lidar.is_initialized:
        world_rays = env._get_lidar_world_rays(env.num_envs)
        if world_rays is not None:
            ray_origins_w, ray_directions_w = world_rays
            ray_origins_w = ray_origins_w[:, :num_rays]
            ray_directions_w = ray_directions_w[:, :num_rays]
            ray_hits_w = env._lidar.data.ray_hits_w[:, :num_rays]
            hit_is_finite = torch.isfinite(ray_hits_w).all(dim=-1, keepdim=True)
            fallback_endpoints_w = ray_origins_w + ray_directions_w * env.cfg.obstacle_detection_range
            ray_endpoints_w = torch.where(hit_is_finite, ray_hits_w, fallback_endpoints_w)
            ray_vectors_w = ray_endpoints_w - ray_origins_w
            ray_distances = torch.linalg.norm(ray_vectors_w, dim=-1, keepdim=True).clamp_max(
                env.cfg.obstacle_detection_range
            )
            safe_directions = ray_vectors_w / torch.clamp(ray_distances, min=1e-6)
            ray_endpoints_w = ray_origins_w + safe_directions * ray_distances
            return ray_origins_w, ray_endpoints_w

    root_pos_w = env.robot.data.root_pos_w
    root_quat_w = env.robot.data.root_quat_w
    yaw = torch.atan2(
        2.0 * (root_quat_w[:, 0] * root_quat_w[:, 3] + root_quat_w[:, 1] * root_quat_w[:, 2]),
        1.0 - 2.0 * (root_quat_w[:, 2].square() + root_quat_w[:, 3].square()),
    )
    ray_angles = (
        torch.arange(num_rays, device=env.device, dtype=torch.float32)
        * (2.0 * torch.pi / float(num_rays))
        - torch.pi
    )
    world_angles = yaw.unsqueeze(1) + ray_angles.unsqueeze(0)
    ray_dirs_w = torch.stack(
        [torch.cos(world_angles), torch.sin(world_angles), torch.zeros_like(world_angles)],
        dim=-1,
    )
    ray_origins_w = root_pos_w.unsqueeze(1).expand(-1, num_rays, -1)
    ray_distances = env._compute_front_ray_distances(root_pos_w, root_quat_w)[:, :num_rays]
    ray_distances = ray_distances * env.cfg.obstacle_detection_range
    ray_endpoints_w = ray_origins_w + ray_dirs_w * ray_distances.unsqueeze(-1)
    return ray_origins_w, ray_endpoints_w


def _draw_front_rays(env, draw_interface, num_rays: int):
    if draw_interface is None:
        return
    try:
        draw_interface.clear_lines()
    except Exception:
        pass

    ray_origins_w, ray_endpoints_w = _compute_front_rays(env, num_rays)
    origins = ray_origins_w.reshape(-1, 3).detach().cpu().tolist()
    endpoints = ray_endpoints_w.reshape(-1, 3).detach().cpu().tolist()
    colors = [(0.2, 0.9, 1.0, 1.0)] * len(origins)
    sizes = [float(env.cfg.debug_lidar_ray_size)] * len(origins)
    try:
        draw_interface.draw_lines(origins, endpoints, colors, sizes)
    except Exception:
        pass


def _print_debug_state(env, action: torch.Tensor, step_count: int):
    import isaaclab.utils.math as math_utils

    root_quat_w = env.robot.data.root_quat_w
    root_lin_vel_w = env.robot.data.root_lin_vel_w
    root_lin_vel_b = math_utils.quat_apply_inverse(root_quat_w, root_lin_vel_w)
    target_vec_w = env._target_positions_w - env.robot.data.root_pos_w
    yaw = torch.atan2(
        2.0 * (root_quat_w[:, 0] * root_quat_w[:, 3] + root_quat_w[:, 1] * root_quat_w[:, 2]),
        1.0 - 2.0 * (root_quat_w[:, 2].square() + root_quat_w[:, 3].square()),
    )

    env_id = 0
    print(
        f"[PLAY DEBUG] step={step_count} env={env_id} "
        f"yaw={yaw[env_id].item():+.3f} "
        f"cmd_b=({action[env_id, 0].item():+.2f}, {action[env_id, 1].item():+.2f}, {action[env_id, 2].item():+.2f}) "
        f"vel_b=({root_lin_vel_b[env_id, 0].item():+.2f}, {root_lin_vel_b[env_id, 1].item():+.2f}, {root_lin_vel_b[env_id, 2].item():+.2f}) "
        f"vel_w=({root_lin_vel_w[env_id, 0].item():+.2f}, {root_lin_vel_w[env_id, 1].item():+.2f}, {root_lin_vel_w[env_id, 2].item():+.2f}) "
        f"target_w=({target_vec_w[env_id, 0].item():+.2f}, {target_vec_w[env_id, 1].item():+.2f}, {target_vec_w[env_id, 2].item():+.2f})"
    )


def main(args_cli):
    from quadcopter_obstacles_env import QuadcopterObstaclesEnv, QuadcopterObstaclesEnvCfg

    env_cfg = QuadcopterObstaclesEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.num_obstacle_rays = args_cli.num_rays
    device = getattr(args_cli, "device", env_cfg.device)
    env_cfg.sim.device = device
    env_cfg.device = device
    env_cfg.seed = args_cli.seed
    env_cfg.auto_reset_done = False
    env_cfg.use_raycast_lidar = True
    env_cfg.debug_vis = not bool(getattr(args_cli, "headless", False))
    env_cfg.viewer_eye = (-48.0, 0.0, 18.0)
    env_cfg.viewer_lookat = (0.0, 0.0, 1.5)

    print("[PLAY] creating environment...", flush=True)
    env = QuadcopterObstaclesEnv(cfg=env_cfg, render_mode=None)
    try:
        print("[PLAY] resetting environment...", flush=True)
        env.reset()
        print("[PLAY] reset complete, entering loop...", flush=True)
        action = torch.tensor(
            [args_cli.forward_cmd, args_cli.lateral_cmd, args_cli.vertical_cmd],
            dtype=torch.float32,
            device=env.device,
        ).unsqueeze(0).repeat(env.num_envs, 1)
        step_count = 0

        while True:
            _, _, terminated, truncated, _ = env.step(action)
            if args_cli.debug_interval > 0 and step_count % args_cli.debug_interval == 0:
                _print_debug_state(env, action, step_count)
            reset_mask = terminated | truncated
            if torch.any(reset_mask):
                env._reset_idx(torch.nonzero(reset_mask, as_tuple=False).squeeze(-1))
            if args_cli.sleep > 0.0:
                time.sleep(args_cli.sleep)
            step_count += 1
    finally:
        env.close()


if __name__ == "__main__":
    parser = build_parser()
    args_cli = parser.parse_args()

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    try:
        main(args_cli)
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
