from __future__ import annotations

import argparse
import csv
import os
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime

import cv2
import numpy as np


TASK_DIR = "/home/wei/End_to_end/drone_isaac/quadcopter_obstacles_teacher"
ENV_DIR = os.path.abspath(os.path.join(TASK_DIR, ".."))
ROOT_DIR = os.path.abspath(os.path.join(TASK_DIR, "..", ".."))
LOCAL_RSL_RL_DIR = os.path.join(ROOT_DIR, "rsl_rl")
DEFAULT_MODEL_PATH = (
    "/home/wei/End_to_end/logs/rsl_rl/quadcopter_obstacles_v5/2026-03-14_02-50-17/model_3000.pt"
)

for path in (ROOT_DIR, ENV_DIR, LOCAL_RSL_RL_DIR, TASK_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)


@dataclass
class EpisodeBuffer:
    episode_id: int
    rows: list[dict[str, object]] = field(default_factory=list)
    image_names: list[str] = field(default_factory=list)
    depth_images_u8: list[np.ndarray] = field(default_factory=list)


class DatasetWriter:
    def __init__(
        self,
        output_root: str,
        num_envs: int,
        success_only: bool = False,
        image_scale: int = 4,
    ):
        self.output_root = output_root
        self.num_envs = num_envs
        self.success_only = success_only
        self.image_scale = max(int(image_scale), 1)
        self.episodes_written = 0
        self.episodes_skipped = 0
        self.next_episode_id = 0
        self.buffers = [EpisodeBuffer(episode_id=self._allocate_episode_id()) for _ in range(num_envs)]
        os.makedirs(self.output_root, exist_ok=True)

    def _allocate_episode_id(self) -> int:
        episode_id = self.next_episode_id
        self.next_episode_id += 1
        return episode_id

    def append(
        self,
        env_index: int,
        step_id: int,
        timestamp: float,
        depth_image_u8: np.ndarray,
        pos_w: np.ndarray,
        target_pos_b: np.ndarray,
        lin_vel_b: np.ndarray,
        teacher_cmd_b: np.ndarray,
        done: bool,
        success: bool,
        collision: bool,
        timeout: bool,
    ) -> None:
        buffer = self.buffers[env_index]
        image_name = f"{len(buffer.rows):06d}.png"
        row = {
            "episode_id": buffer.episode_id,
            "step_id": step_id,
            "timestamp": timestamp,
            "depth_image": image_name,
            "pos_w_x": float(pos_w[0]),
            "pos_w_y": float(pos_w[1]),
            "pos_w_z": float(pos_w[2]),
            "target_pos_b_x": float(target_pos_b[0]),
            "target_pos_b_y": float(target_pos_b[1]),
            "target_pos_b_z": float(target_pos_b[2]),
            "lin_vel_b_x": float(lin_vel_b[0]),
            "lin_vel_b_y": float(lin_vel_b[1]),
            "lin_vel_b_z": float(lin_vel_b[2]),
            "teacher_cmd_b_x": float(teacher_cmd_b[0]),
            "teacher_cmd_b_y": float(teacher_cmd_b[1]),
            "teacher_cmd_b_z": float(teacher_cmd_b[2]),
            "done": bool(done),
            "success": bool(success),
            "collision": bool(collision),
            "timeout": bool(timeout),
        }
        buffer.rows.append(row)
        buffer.image_names.append(image_name)
        buffer.depth_images_u8.append(depth_image_u8.copy())

    def flush_episode(self, env_index: int, suffix: str = "") -> None:
        buffer = self.buffers[env_index]
        if not buffer.rows:
            return

        if not (len(buffer.rows) == len(buffer.image_names) == len(buffer.depth_images_u8)):
            raise RuntimeError(
                f"Episode buffer out of sync for env {env_index}: "
                f"rows={len(buffer.rows)}, image_names={len(buffer.image_names)}, images={len(buffer.depth_images_u8)}"
            )

        if len(set(buffer.image_names)) != len(buffer.image_names):
            raise RuntimeError(f"Duplicate image names detected for env {env_index}, episode {buffer.episode_id}")

        final_row = buffer.rows[-1]
        is_success = bool(final_row["success"])
        is_partial = bool(suffix)
        if self.success_only and (not is_success or is_partial):
            self.episodes_skipped += 1
            self.buffers[env_index] = EpisodeBuffer(episode_id=buffer.episode_id + 1)
            return

        folder_name = f"{env_index}{suffix}"
        folder_path = os.path.join(self.output_root, folder_name)
        os.makedirs(folder_path, exist_ok=False)

        for image_name, depth_image in zip(buffer.image_names, buffer.depth_images_u8):
            image_path = os.path.join(folder_path, image_name)
            if self.image_scale > 1:
                depth_image = cv2.resize(
                    depth_image,
                    (depth_image.shape[1] * self.image_scale, depth_image.shape[0] * self.image_scale),
                    interpolation=cv2.INTER_NEAREST,
                )
            ok = cv2.imwrite(image_path, np.ascontiguousarray(depth_image))
            if not ok:
                raise RuntimeError(f"Failed to write image to {image_path}")

        csv_path = os.path.join(folder_path, "data.csv")
        fieldnames = [
            "episode_id",
            "step_id",
            "timestamp",
            "depth_image",
            "pos_w_x",
            "pos_w_y",
            "pos_w_z",
            "target_pos_b_x",
            "target_pos_b_y",
            "target_pos_b_z",
            "lin_vel_b_x",
            "lin_vel_b_y",
            "lin_vel_b_z",
            "teacher_cmd_b_x",
            "teacher_cmd_b_y",
            "teacher_cmd_b_z",
            "done",
            "success",
            "collision",
            "timeout",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(buffer.rows)

        written_pngs = sorted(name for name in os.listdir(folder_path) if name.endswith(".png"))
        if len(written_pngs) != len(buffer.rows):
            raise RuntimeError(
                f"Written file count mismatch for {folder_path}: "
                f"pngs={len(written_pngs)}, rows={len(buffer.rows)}"
            )

        written_csv_names = [row["depth_image"] for row in buffer.rows]
        if written_pngs != sorted(written_csv_names):
            raise RuntimeError(
                f"CSV/image name mismatch for {folder_path}: "
                f"csv_names={len(written_csv_names)}, pngs={len(written_pngs)}"
            )

        self.episodes_written += 1
        self.buffers[env_index] = EpisodeBuffer(episode_id=self._allocate_episode_id())

    def flush_all_partials(self) -> int:
        partial_count = 0
        for env_index, buffer in enumerate(self.buffers):
            if buffer.rows:
                self.flush_episode(env_index, suffix="_partial")
                partial_count += 1
        return partial_count


def _default_dataset_out() -> str:
    run_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return os.path.join(TASK_DIR, "data", f"eval_{run_stamp}")


def _extract_student_features(policy_obs, actions):
    lin_vel_b = policy_obs[:, 0:3]
    target_pos_b = policy_obs[:, 9:12]
    teacher_cmd_b = actions[:, 0:3]
    return lin_vel_b, target_pos_b, teacher_cmd_b


def _get_policy_obs(obs):
    if hasattr(obs, "keys") and "policy" in obs.keys():
        return obs["policy"]
    if isinstance(obs, dict):
        return obs["policy"]
    return obs


def main():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Evaluate a trained teacher quadcopter obstacles policy and collect expert data.")
    parser.add_argument("--task", type=str, default="Isaac-Quadcopter-Obstacles-Teacher-DepthEval-v0")
    parser.add_argument("--num_envs", type=int, default=128)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--metrics_out",
        type=str,
        default="/home/wei/End_to_end/logs/rsl_rl/quadcopter_obstacles_v5/2026-03-14_02-50-17/eval_teacher_metrics.txt",
    )
    parser.add_argument("--dataset_out", type=str, default=_default_dataset_out())
    parser.add_argument("--save_dataset", action="store_true", default=True)
    parser.add_argument("--no_save_dataset", dest="save_dataset", action="store_false")
    parser.add_argument("--success_only", action="store_true", default=True)
    parser.add_argument("--save_every_n_steps", type=int, default=5)
    parser.add_argument("--save_image_scale", type=int, default=16)
    parser.add_argument("--use_follow_camera_depth", action="store_true", default=True)
    parser.add_argument("--no_use_follow_camera_depth", dest="use_follow_camera_depth", action="store_false")
    parser.add_argument("--show_depth_preview", action="store_true", default=False)
    parser.add_argument("--depth_preview_env_index", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=4000)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    if getattr(args, "headless", False) and args.use_follow_camera_depth:
        args.use_follow_camera_depth = False

    # Isaac Sim startup can fail if the inherited shell cwd no longer exists.
    # Pin the process cwd to the project root before AppLauncher initializes Kit.
    os.chdir(ROOT_DIR)

    os.makedirs(os.path.dirname(args.metrics_out), exist_ok=True)
    log_file = open(args.metrics_out, "w", encoding="utf-8")

    def log(message: str):
        print(message)
        log_file.write(message + "\n")
        log_file.flush()

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    try:
        log("simulation_app_started")

        import gymnasium as gym
        import torch

        from isaaclab.utils.dict import class_to_dict

        from local_rsl_rl import RslRlVecEnvWrapper
        from rsl_rl.runners import OnPolicyRunner

        import quadcopter_obstacles_teacher  # noqa: F401

        log("imports_after_app_ok")

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

        env_cfg = _load_cfg_from_registry(args.task, "env_cfg_entry_point")
        agent_cfg = _load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
        log("configs_loaded")

        env_cfg.scene.num_envs = args.num_envs
        env_cfg.debug_vis = False
        env_cfg.sim.device = args.device
        if hasattr(env_cfg, "auto_reset_done"):
            env_cfg.auto_reset_done = False
        agent_cfg.device = args.device

        if hasattr(env_cfg, "show_depth_preview"):
            env_cfg.show_depth_preview = args.show_depth_preview
        if hasattr(env_cfg, "depth_preview_env_index"):
            env_cfg.depth_preview_env_index = args.depth_preview_env_index

        dataset_writer = None
        if args.save_dataset:
            dataset_writer = DatasetWriter(
                args.dataset_out,
                args.num_envs,
                success_only=args.success_only,
                image_scale=args.save_image_scale,
            )
            log(f"dataset_out={args.dataset_out}")
            log(f"dataset_success_only={args.success_only}")
            log(f"dataset_save_every_n_steps={args.save_every_n_steps}")
            log(f"dataset_save_image_scale={args.save_image_scale}")
            log(f"dataset_use_follow_camera_depth={args.use_follow_camera_depth}")

        env = gym.make(args.task, cfg=env_cfg, render_mode=None)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        log("env_created")
        obs = env.reset()
        env.unwrapped.episode_length_buf.zero_()
        log("env_reset")

        if not hasattr(env.unwrapped, "get_depth_frames"):
            raise RuntimeError(
                "Current evaluation env does not expose get_depth_frames(). "
                "Use Isaac-Quadcopter-Obstacles-Teacher-DepthEval-v0 for dataset collection."
            )

        runner = OnPolicyRunner(env, class_to_dict(agent_cfg), log_dir=None, device=agent_cfg.device)
        runner.load(args.model, load_optimizer=False)
        policy = runner.get_inference_policy(device=agent_cfg.device)
        log("policy_loaded")

        reward_sum = 0.0
        reward_sq_sum = 0.0
        done_count = 0
        success_count = 0.0
        died_count = 0.0
        timeout_count = 0.0
        speed_sum = 0.0
        speed_xy_sum = 0.0
        action_abs_sum = torch.zeros(env.num_actions, device=args.device)
        action_sq_sum = torch.zeros(env.num_actions, device=args.device)
        final_target_distance = None
        final_obstacle_distance = None
        final_wall_distance = None

        step_ids = torch.zeros(args.num_envs, dtype=torch.long, device=args.device)
        finished_envs = torch.zeros(args.num_envs, dtype=torch.bool, device=args.device)
        step_dt = float(env.unwrapped.step_dt)
        step = 0

        with torch.inference_mode():
            while not bool(finished_envs.all().item()):
                if args.max_steps > 0 and step >= args.max_steps:
                    log(f"max_steps_reached={args.max_steps}")
                    break

                actions = policy(obs)
                actions = actions.clone()
                actions[finished_envs] = 0.0

                policy_obs = _get_policy_obs(obs)
                lin_vel_b, target_pos_b, teacher_cmd_b = _extract_student_features(policy_obs, actions)

                obs, rew, dones, extras = env.step(actions)
                if args.use_follow_camera_depth:
                    depth_frames_u8 = None
                else:
                    depth_frames_u8 = env.unwrapped.get_record_depth_frames_u8().detach().cpu().numpy()

                reward_sum += rew.sum().item()
                reward_sq_sum += torch.square(rew).sum().item()
                done_flags = dones.to(torch.bool)
                newly_finished = done_flags & ~finished_envs
                done_step = int(newly_finished.sum().item())
                done_count += done_step

                success_flags = extras["target_reached"].to(torch.bool)
                timeout_flags = extras["time_outs"].to(torch.bool)
                collision_flags = (extras["obstacle_collision"] | extras["wall_collision"]).to(torch.bool)

                if dataset_writer is not None:
                    target_pos_b_np = target_pos_b.detach().cpu().numpy()
                    lin_vel_b_np = lin_vel_b.detach().cpu().numpy()
                    teacher_cmd_b_np = teacher_cmd_b.detach().cpu().numpy()
                    pos_w_np = env.unwrapped.robot.data.root_pos_w.detach().cpu().numpy()
                    done_np = done_flags.detach().cpu().numpy()
                    success_np = success_flags.detach().cpu().numpy()
                    collision_np = collision_flags.detach().cpu().numpy()
                    timeout_np = timeout_flags.detach().cpu().numpy()
                    step_ids_np = step_ids.detach().cpu().numpy()

                    for env_index in range(args.num_envs):
                        if finished_envs[env_index]:
                            continue
                        should_save_this_step = (
                            int(step_ids_np[env_index]) % max(args.save_every_n_steps, 1) == 0
                            or bool(done_np[env_index])
                        )
                        if not should_save_this_step:
                            continue
                        timestamp = float(step_ids_np[env_index] * step_dt)
                        if args.use_follow_camera_depth:
                            depth_image_u8 = env.unwrapped.get_follow_camera_depth_u8(env_index)
                        else:
                            depth_image_u8 = depth_frames_u8[env_index]
                        dataset_writer.append(
                            env_index=env_index,
                            step_id=int(step_ids_np[env_index]),
                            timestamp=timestamp,
                            depth_image_u8=depth_image_u8,
                            pos_w=pos_w_np[env_index],
                            target_pos_b=target_pos_b_np[env_index],
                            lin_vel_b=lin_vel_b_np[env_index],
                            teacher_cmd_b=teacher_cmd_b_np[env_index],
                            done=bool(done_np[env_index]),
                            success=bool(success_np[env_index]),
                            collision=bool(collision_np[env_index]),
                            timeout=bool(timeout_np[env_index]),
                        )

                        if done_np[env_index]:
                            dataset_writer.flush_episode(env_index)

                success_step = float((newly_finished & success_flags).float().sum().item())
                timeout_step = float((newly_finished & timeout_flags).float().sum().item())
                died_step = float((newly_finished & ~success_flags & ~timeout_flags).float().sum().item())

                died_count += died_step
                timeout_count += timeout_step
                success_count += success_step

                lin_vel_w = env.unwrapped.robot.data.root_lin_vel_w
                speed_sum += torch.linalg.norm(lin_vel_w, dim=1).sum().item()
                speed_xy_sum += torch.linalg.norm(lin_vel_w[:, :2], dim=1).sum().item()

                action_abs_sum += actions.abs().sum(dim=0)
                action_sq_sum += torch.square(actions).sum(dim=0)

                target_world = env.unwrapped._target_positions_w
                robot_pos_w = env.unwrapped.robot.data.root_pos_w
                final_target_distance = torch.linalg.norm(target_world - robot_pos_w, dim=1)
                final_obstacle_distance = env.unwrapped._compute_closest_obstacle_signed_distance(robot_pos_w)
                final_wall_distance = env.unwrapped._compute_wall_signed_distance(robot_pos_w)

                step_ids += 1
                finished_envs |= done_flags
                step_ids[newly_finished] = 0
                step += 1

                if step % 250 == 0:
                    mean_reward = reward_sum / (step * args.num_envs)
                    log(
                        f"[eval] step={step} mean_step_reward={mean_reward:.4f} "
                        f"episodes_done={done_count} success={success_count:.2f} "
                        f"died={died_count:.2f} timeout={timeout_count:.2f} "
                        f"finished_envs={int(finished_envs.sum().item())}/{args.num_envs}"
                    )

        total_samples = max(step * args.num_envs, 1)
        mean_reward = reward_sum / total_samples
        reward_std = max(reward_sq_sum / total_samples - mean_reward**2, 0.0) ** 0.5
        avg_speed = speed_sum / total_samples
        avg_speed_xy = speed_xy_sum / total_samples
        action_abs_mean = (action_abs_sum / total_samples).tolist()
        action_rms = torch.sqrt(action_sq_sum / total_samples).tolist()

        log("=== Evaluation Summary ===")
        log(f"model={args.model}")
        log(f"num_envs={args.num_envs}")
        log(f"steps={step}")
        log(f"episodes_done={done_count}")
        log(f"success_count={success_count:.2f}")
        log(f"died_count={died_count:.2f}")
        log(f"timeout_count={timeout_count:.2f}")
        if done_count > 0:
            log(f"success_rate={success_count / done_count:.4f}")
            log(f"died_rate={died_count / done_count:.4f}")
            log(f"timeout_rate={timeout_count / done_count:.4f}")
        log(f"mean_step_reward={mean_reward:.6f}")
        log(f"std_step_reward={reward_std:.6f}")
        log(f"avg_speed_mean={avg_speed:.4f}")
        log(f"avg_speed_xy_mean={avg_speed_xy:.4f}")
        if final_target_distance is not None:
            log(f"final_target_distance_mean={final_target_distance.mean().item():.4f}")
            log(f"final_target_distance_median={final_target_distance.median().item():.4f}")
        if final_obstacle_distance is not None:
            log(f"final_obstacle_distance_mean={final_obstacle_distance.mean().item():.4f}")
            log(f"final_obstacle_distance_median={final_obstacle_distance.median().item():.4f}")
        if final_wall_distance is not None:
            log(f"final_wall_distance_mean={final_wall_distance.mean().item():.4f}")
            log(f"final_wall_distance_median={final_wall_distance.median().item():.4f}")
        log(f"action_abs_mean={action_abs_mean}")
        log(f"action_rms={action_rms}")

        if dataset_writer is not None:
            partial_count = dataset_writer.flush_all_partials()
            log(f"dataset_episodes_written={dataset_writer.episodes_written}")
            log(f"dataset_episodes_skipped={dataset_writer.episodes_skipped}")
            log(f"dataset_partial_episodes_written={partial_count}")

        env.close()
    except Exception as exc:
        log(f"exception={type(exc).__name__}: {exc}")
        log(traceback.format_exc())
        raise
    finally:
        try:
            log("simulation_app_closing")
        except Exception:
            pass
        log_file.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
