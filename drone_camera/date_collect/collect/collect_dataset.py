from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any

COLLECT_DIR = Path(__file__).resolve().parent
DATE_COLLECT_DIR = COLLECT_DIR.parent
DRONE_CAMERA_DIR = DATE_COLLECT_DIR.parent
DRONE_ISAAC_DIR = DRONE_CAMERA_DIR.parent
ROOT_DIR = DRONE_ISAAC_DIR.parent
LOCAL_RSL_RL_DIR = DRONE_ISAAC_DIR / "local_rsl_rl"
RSL_RL_DIR = DRONE_ISAAC_DIR / "rsl_rl"
ISAACLAB_DIR = Path("/home/wei/IsaacLab")
ISAACLAB_SOURCE_DIRS = [
    ISAACLAB_DIR / "source" / "isaaclab",
    ISAACLAB_DIR / "source" / "isaaclab_assets",
    ISAACLAB_DIR / "source" / "isaaclab_rl",
    ISAACLAB_DIR / "source" / "isaaclab_tasks",
]

for path in [ROOT_DIR, DRONE_ISAAC_DIR, DRONE_CAMERA_DIR, LOCAL_RSL_RL_DIR, RSL_RL_DIR, *ISAACLAB_SOURCE_DIRS]:
    path_str = str(path)
    if path.exists() and path_str not in sys.path:
        sys.path.insert(0, path_str)


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
    env["DEPTH_COLLECT_REEXEC"] = "1"
    os.execvpe(sys.executable, [sys.executable, *sys.argv], env)


_ensure_conda_lib_first()

from isaaclab.app import AppLauncher


DEFAULT_TEACHER_MODEL = COLLECT_DIR / "__pycache__" / "model_5300.pt"
DEFAULT_OUTPUT_DIR = COLLECT_DIR / "data"
DEFAULT_ENV_CFG_DIR = DRONE_CAMERA_DIR / "quadcopter_obstacles_student" / "cfg"


parser = argparse.ArgumentParser(description="Collect depth-image sequences with a privileged teacher policy.")
parser.add_argument("--teacher_model", type=Path, default=DEFAULT_TEACHER_MODEL)
parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
parser.add_argument("--env_cfg_dir", type=Path, default=DEFAULT_ENV_CFG_DIR)
parser.add_argument("--task", type=str, default="Isaac-Quadcopter-Obstacles-DepthCollect-v0")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument("--chunk_steps", type=int, default=256)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--run_name", type=int, default=None, help="Optional first numeric trajectory folder index.")
parser.add_argument("--depth_image_height", type=int, default=480)
parser.add_argument("--depth_image_width", type=int, default=640)
parser.add_argument(
    "--depth_camera_offset_pos",
    type=float,
    nargs=3,
    default=(0.12, 0.0, 0.0),
    metavar=("X", "Y", "Z"),
    help="Camera offset on the drone body.",
)
parser.add_argument("--save_trajectories", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--trajectory_image_stride", type=int, default=1)
parser.add_argument("--teacher_action_xy_scale", type=float, default=0.75)
parser.add_argument("--teacher_action_z_scale", type=float, default=1.0)
parser.add_argument("--max_abs_pitch_deg", type=float, default=35.0)
parser.add_argument("--depth_source", type=str, choices=["geometric", "camera"], default="geometric")
parser.add_argument(
    "--sync_depth_camera_pose",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Kept for compatibility with older collection commands.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = args_cli.depth_source == "camera" or bool(args_cli.sync_depth_camera_pose)
if not args_cli.headless and not args_cli.experience:
    args_cli.experience = "isaaclab.python.rendering.kit"

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

from isaaclab.utils.dict import class_to_dict

from local_rsl_rl import RslRlVecEnvWrapper
from rsl_rl.modules import ActorCriticMultiHeadObs

import collection_env  # noqa: F401
from quadcopter_obstacles_student.config_utils import apply_env_cfg_dir

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


def _next_numeric_dir_index(output_dir: Path) -> int:
    numeric_names = [int(path.name) for path in output_dir.iterdir() if path.is_dir() and path.name.isdigit()]
    return max(numeric_names) + 1 if numeric_names else 0


def _tensor_to_numpy(tensor: torch.Tensor, dtype: np.dtype | None = None) -> np.ndarray:
    array = tensor.detach().cpu().numpy()
    return array.astype(dtype) if dtype is not None else array


def _stack_np(items: list[torch.Tensor], dtype: np.dtype | None = None) -> np.ndarray:
    return _tensor_to_numpy(torch.stack(items, dim=0), dtype=dtype)


def _quat_wxyz_to_rpy(quat_wxyz: np.ndarray) -> np.ndarray:
    quat = quat_wxyz.astype(np.float64)
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.rad2deg(np.stack([roll, pitch, yaw], axis=-1)).astype(np.float32)


def _record_obs(obs: dict[str, torch.Tensor], env, capture_depth: bool) -> dict[str, torch.Tensor]:
    unwrapped = env.unwrapped
    record = {key: value.clone() for key, value in obs.items()}
    if "policy_image" in obs:
        record["obs_depth_norm"] = obs["policy_image"].clone()
    elif capture_depth and hasattr(unwrapped, "capture_depth_camera_image"):
        record["obs_depth_norm"] = unwrapped.capture_depth_camera_image().clone()
    record["robot_pos_w"] = unwrapped.robot.data.root_pos_w.clone()
    record["robot_quat_w"] = unwrapped.robot.data.root_quat_w.clone()
    record["robot_lin_vel_w"] = unwrapped.robot.data.root_lin_vel_w.clone()
    return record


def _append_obs_buffers(buffers: dict[str, list[torch.Tensor]], obs: dict[str, torch.Tensor], env, capture_depth: bool) -> None:
    record = _record_obs(obs, env, capture_depth)
    for key, value in record.items():
        buffers.setdefault(key, []).append(value)


def _extras_value(
    extras: dict[str, Any],
    key: str,
    num_envs: int,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    value = extras.get(key)
    if value is None:
        value = torch.zeros(num_envs, device=device)
    if not isinstance(value, torch.Tensor):
        value = torch.tensor(value, device=device)
    value = value.reshape(num_envs)
    return value.to(dtype=dtype) if dtype is not None else value


def _trajectory_csv_header() -> list[str]:
    return [
        "step_in_episode",
        "image",
        "pos_x",
        "pos_y",
        "pos_z",
        "lin_vel_x",
        "lin_vel_y",
        "lin_vel_z",
        "quat_w",
        "quat_x",
        "quat_y",
        "quat_z",
        "roll_deg",
        "pitch_deg",
        "yaw_deg",
        "teacher_action_x",
        "teacher_action_y",
        "teacher_action_z",
        "teacher_action_yaw",
        "teacher_cmd_vel_x",
        "teacher_cmd_vel_y",
        "teacher_cmd_vel_z",
        "distance_to_target",
        "target_reached",
        "done",
    ]


def _append_trajectory_csv_rows(csv_path: Path, rows: list[list[object]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow(_trajectory_csv_header())
        writer.writerows(rows)


def _save_trajectory_frames(output_dir: Path, first_trajectory_index: int, payload: dict[str, np.ndarray]) -> None:
    try:
        from PIL import Image
    except Exception:
        Image = None
        print("[WARN] Pillow is not available; skipping trajectory depth PNG export.", flush=True)

    stride = max(1, int(args_cli.trajectory_image_stride))
    depth = payload["obs_depth_norm"]
    teacher_action = payload["teacher_action"]
    teacher_cmd_vel_b = payload["teacher_cmd_vel_b"]
    episode_id = payload["episode_id"]
    step_in_episode = payload["step_in_episode"]
    robot_pos_w = payload["robot_pos_w"]
    robot_quat_w = payload["robot_quat_w"]
    robot_rpy_deg = payload["robot_rpy_deg"]
    robot_lin_vel_w = payload["robot_lin_vel_w"]
    distance_to_target = payload["distance_to_target"]
    target_reached = payload["target_reached"]
    done = payload["done"]

    rows_by_csv: dict[Path, list[list[object]]] = {}
    max_abs_pitch_deg = float(args_cli.max_abs_pitch_deg)
    num_steps, num_envs = depth.shape[:2]
    for t in range(num_steps):
        if t % stride != 0:
            continue
        for env_idx in range(num_envs):
            if max_abs_pitch_deg > 0.0 and abs(float(robot_rpy_deg[t, env_idx, 1])) > max_abs_pitch_deg:
                continue
            eid = int(episode_id[t, env_idx]) + first_trajectory_index
            frame = int(step_in_episode[t, env_idx])
            traj_dir = output_dir / f"{eid:06d}"
            depth_dir = traj_dir / "depth"
            depth_rel = f"depth/{frame:06d}.png"
            if Image is not None:
                depth_dir.mkdir(parents=True, exist_ok=True)
                image = depth[t, env_idx, 0]
                image_u8 = np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8)
                Image.fromarray(image_u8).save(depth_dir / f"{frame:06d}.png")
            row = [
                frame,
                depth_rel,
                *[float(v) for v in robot_pos_w[t, env_idx]],
                *[float(v) for v in robot_lin_vel_w[t, env_idx]],
                *[float(v) for v in robot_quat_w[t, env_idx]],
                *[float(v) for v in robot_rpy_deg[t, env_idx]],
                *[float(v) for v in teacher_action[t, env_idx]],
                *[float(v) for v in teacher_cmd_vel_b[t, env_idx]],
                float(distance_to_target[t, env_idx]),
                int(target_reached[t, env_idx]),
                int(done[t, env_idx]),
            ]
            rows_by_csv.setdefault(traj_dir / "data.csv", []).append(row)

    for csv_path, rows in rows_by_csv.items():
        _append_trajectory_csv_rows(csv_path, rows)


def _save_chunk(
    output_dir: Path,
    first_trajectory_index: int,
    chunk_idx: int,
    obs_buffers: dict[str, list[torch.Tensor]],
    step_buffers: dict[str, list[torch.Tensor]],
    episode_start: torch.Tensor,
    episode_id: torch.Tensor,
    step_in_episode: torch.Tensor,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir = output_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    payload: dict[str, np.ndarray] = {}
    for key, items in obs_buffers.items():
        dtype = np.float16 if key in {"policy_image", "obs_depth_norm"} else np.float32
        payload[key] = _stack_np(items, dtype=dtype)
    for key, items in step_buffers.items():
        if key in {"done", "target_reached", "episode_start"}:
            dtype = np.bool_
        elif key in {"episode_id", "step_in_episode"}:
            dtype = np.int64
        else:
            dtype = np.float32
        payload[key] = _stack_np(items, dtype=dtype)

    if "obs_depth_norm" not in payload and "policy_image" in payload:
        payload["obs_depth_norm"] = payload["policy_image"]
    payload["episode_start"] = _tensor_to_numpy(episode_start, np.bool_)
    payload["episode_id_last"] = _tensor_to_numpy(episode_id, np.int64)
    payload["step_in_episode_last"] = _tensor_to_numpy(step_in_episode, np.int64)
    payload["robot_rpy_deg"] = _quat_wxyz_to_rpy(payload["robot_quat_w"])

    abs_pitch = np.abs(payload["robot_rpy_deg"][..., 1])
    abs_roll = np.abs(payload["robot_rpy_deg"][..., 0])
    teacher_action = payload.get("teacher_action", np.zeros((*abs_pitch.shape, 4), dtype=np.float32))
    action_xy_sat = np.linalg.norm(teacher_action[..., :2], axis=-1) > 0.95
    print(
        "[INFO] Chunk attitude/action stats: "
        f"abs_pitch_p95={np.percentile(abs_pitch, 95):.2f}deg, "
        f"abs_pitch_max={abs_pitch.max():.2f}deg, "
        f"abs_roll_p95={np.percentile(abs_roll, 95):.2f}deg, "
        f"action_xy_near_sat={action_xy_sat.mean():.4f}",
        flush=True,
    )

    np.savez_compressed(chunks_dir / f"chunk_{chunk_idx:05d}.npz", **payload)
    if args_cli.save_trajectories:
        _save_trajectory_frames(output_dir, first_trajectory_index, payload)


def main() -> None:
    teacher_model = args_cli.teacher_model.expanduser().resolve()
    if not teacher_model.is_file():
        raise FileNotFoundError(f"Teacher checkpoint not found: {teacher_model}")
    if args_cli.steps <= 0 or args_cli.chunk_steps <= 0:
        raise ValueError("--steps and --chunk_steps must be positive.")

    env_cfg = _load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    agent_cfg = _load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    env_cfg_paths = apply_env_cfg_dir(env_cfg, str(args_cli.env_cfg_dir))
    env_cfg.scene.num_envs = int(args_cli.num_envs)
    env_cfg.seed = int(args_cli.seed)
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.device = env_cfg.sim.device
    env_cfg.depth_image_height = int(args_cli.depth_image_height)
    env_cfg.depth_image_width = int(args_cli.depth_image_width)
    env_cfg.depth_camera_offset_pos = tuple(float(v) for v in args_cli.depth_camera_offset_pos)
    env_cfg.depth_source = args_cli.depth_source
    agent_cfg.device = env_cfg.device
    agent_cfg.logger = "tensorboard"

    output_dir = args_cli.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args_cli.run_name is None:
        first_trajectory_index = _next_numeric_dir_index(output_dir)
    else:
        first_trajectory_index = int(args_cli.run_name)
        if first_trajectory_index < 0:
            raise ValueError("--run_name must be a positive integer when using flat trajectory output.")
        first_dir = output_dir / f"{first_trajectory_index:06d}"
        if first_dir.exists():
            raise FileExistsError(
                f"First trajectory directory already exists: {first_dir}. "
                "Omit --run_name to auto-continue after existing numeric folders."
            )

    print(f"[INFO] Collecting flat trajectory folders into: {output_dir}", flush=True)
    print(f"[INFO] First trajectory folder index: {first_trajectory_index}", flush=True)
    print(f"[INFO] Teacher checkpoint: {teacher_model}", flush=True)
    print(f"[INFO] Env cfg dir: {args_cli.env_cfg_dir}", flush=True)
    for path in env_cfg_paths:
        print(f"[INFO] Loaded env cfg: {path}", flush=True)

    print("[INFO] Creating gym env...", flush=True)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    print("[INFO] Gym env created; seeding...", flush=True)
    env.unwrapped.seed(args_cli.seed)
    env.unwrapped._skip_depth_observation = True
    print("[INFO] Wrapping env for RSL-RL...", flush=True)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    print("[INFO] RSL-RL env wrapper ready.", flush=True)

    runner_cfg = class_to_dict(agent_cfg)
    runner_cfg["evaluation"] = {}
    print("[INFO] Creating inference policy module...", flush=True)
    init_obs = env.get_observations()
    policy_cfg = dict(runner_cfg["policy"])
    if runner_cfg.get("empirical_normalization") is not None:
        if policy_cfg.get("actor_obs_normalization") is None:
            policy_cfg["actor_obs_normalization"] = runner_cfg["empirical_normalization"]
        if policy_cfg.get("critic_obs_normalization") is None:
            policy_cfg["critic_obs_normalization"] = runner_cfg["empirical_normalization"]
    policy_cfg.pop("class_name", None)
    policy_cfg = {key: value for key, value in policy_cfg.items() if value is not None}
    policy_module = ActorCriticMultiHeadObs(
        init_obs,
        runner_cfg["obs_groups"],
        env.num_actions,
        **policy_cfg,
    )
    print("[INFO] Loading teacher checkpoint...", flush=True)
    checkpoint = torch.load(str(teacher_model), weights_only=False, map_location="cpu")
    print("[INFO] Teacher checkpoint file loaded; applying state dict...", flush=True)
    state_dict = checkpoint["model_state_dict"]
    first_key = next(iter(state_dict.keys()))
    print(
        f"[INFO] Teacher state dict: {len(state_dict)} tensors, first={first_key}, "
        f"shape={tuple(state_dict[first_key].shape)}",
        flush=True,
    )
    load_result = policy_module.load_state_dict(state_dict, strict=False)
    if hasattr(load_result, "missing_keys"):
        print(
            f"[INFO] Teacher load result: missing={len(load_result.missing_keys)}, "
            f"unexpected={len(load_result.unexpected_keys)}",
            flush=True,
        )
    else:
        print(f"[INFO] Teacher load result: {load_result}", flush=True)
    print("[INFO] Moving teacher policy to device...", flush=True)
    policy_module = policy_module.to(agent_cfg.device)
    policy_module.eval()
    print("[INFO] Teacher checkpoint loaded.", flush=True)
    policy = policy_module.act_inference

    env.unwrapped._skip_depth_observation = False
    print("[INFO] Resetting wrapped env with depth enabled...", flush=True)
    obs = env.reset()
    print("[INFO] Wrapped env reset done; starting collection.", flush=True)
    num_envs = env.num_envs
    device = env.device
    episode_ids = torch.arange(num_envs, device=device, dtype=torch.long)
    next_episode_id = int(num_envs)
    step_counts = torch.zeros(num_envs, device=device, dtype=torch.long)
    episode_start_flags = torch.ones(num_envs, device=device, dtype=torch.bool)

    total_steps = int(args_cli.steps)
    chunk_steps = int(args_cli.chunk_steps)
    collected_steps = 0
    chunk_idx = 0
    capture_depth = True
    try:
        with torch.inference_mode():
            while collected_steps < total_steps:
                current_chunk_steps = min(chunk_steps, total_steps - collected_steps)
                obs_buffers: dict[str, list[torch.Tensor]] = {}
                step_buffers: dict[str, list[torch.Tensor]] = {
                    "teacher_action": [],
                    "teacher_cmd_vel_b": [],
                    "distance_to_target": [],
                    "target_reached": [],
                    "done": [],
                    "episode_id": [],
                    "step_in_episode": [],
                    "episode_start": [],
                }
                for local_step in range(current_chunk_steps):
                    if collected_steps == 0 and local_step == 0:
                        print("[INFO] Collecting first sample...", flush=True)
                    _append_obs_buffers(obs_buffers, obs, env, capture_depth)
                    if collected_steps == 0 and local_step == 0:
                        print("[INFO] First observation recorded; running teacher policy...", flush=True)
                    raw_actions = policy(obs)
                    actions = raw_actions.clone()
                    actions[:, :2] *= float(args_cli.teacher_action_xy_scale)
                    actions[:, 2] *= float(args_cli.teacher_action_z_scale)
                    actions = actions.clamp(-1.0, 1.0)

                    step_buffers["teacher_action"].append(actions.clone())
                    step_buffers["episode_id"].append(episode_ids.clone())
                    step_buffers["step_in_episode"].append(step_counts.clone())
                    step_buffers["episode_start"].append(episode_start_flags.clone())

                    next_obs, _reward, done_long, extras = env.step(actions)
                    if collected_steps == 0 and local_step == 0:
                        print("[INFO] First env step done.", flush=True)
                    done = done_long.bool()
                    step_buffers["teacher_cmd_vel_b"].append(env.unwrapped._cmd_vel_b.clone())
                    step_buffers["done"].append(done.clone())
                    step_buffers["target_reached"].append(
                        _extras_value(extras, "target_reached", num_envs, device, dtype=torch.bool).clone()
                    )
                    step_buffers["distance_to_target"].append(
                        _extras_value(extras, "distance_to_target", num_envs, device, dtype=torch.float32).clone()
                    )

                    step_counts += 1
                    episode_start_flags = torch.zeros_like(episode_start_flags)
                    if done.any():
                        done_ids = torch.nonzero(done, as_tuple=False).squeeze(-1)
                        num_done = int(done_ids.numel())
                        new_ids = torch.arange(next_episode_id, next_episode_id + num_done, device=device, dtype=torch.long)
                        episode_ids[done_ids] = new_ids
                        step_counts[done_ids] = 0
                        episode_start_flags[done_ids] = True
                        next_episode_id += num_done

                    obs = next_obs
                    collected_steps += 1

                _save_chunk(
                    output_dir,
                    first_trajectory_index,
                    chunk_idx,
                    obs_buffers,
                    step_buffers,
                    episode_start_flags,
                    episode_ids,
                    step_counts,
                )
                print(
                    f"[INFO] Saved trajectory batch {chunk_idx:05d}: "
                    f"{collected_steps}/{total_steps} steps, next_episode_id={next_episode_id}",
                    flush=True,
                )
                chunk_idx += 1
    finally:
        env.close()

    print(f"[INFO] Dataset complete. Last allocated trajectory id: {first_trajectory_index + next_episode_id - 1}", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
