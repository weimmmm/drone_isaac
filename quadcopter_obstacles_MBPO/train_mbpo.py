from __future__ import annotations

import argparse
import faulthandler
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import traceback
from datetime import datetime


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

    parser = argparse.ArgumentParser(description="Train the quadcopter obstacle teacher with the staged MBPO/SSAC core.")
    parser.add_argument("--num_envs", type=int, default=512)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--steps_per_epoch", type=int, default=100)
    parser.add_argument("--warmup_epochs", type=int, default=400, help="前多少个 epoch 持续使用 warmup 探索策略。大规模训练默认延长 warmup，让动力学模型先吃够随机探索数据。")
    parser.add_argument("--buffer_min", type=int, default=None)
    parser.add_argument("--replay_buffer_max", type=int, default=None)
    parser.add_argument("--virt_buffer_max", type=int, default=None)
    parser.add_argument("--rollout_horizon", type=int, default=1)
    parser.add_argument("--rollout_batch_size", type=int, default=None)
    parser.add_argument("--model_initial_steps", type=int, default=None)
    parser.add_argument("--model_steps", type=int, default=None)
    parser.add_argument("--model_update_period", type=int, default=None)
    parser.add_argument("--model_start_epoch", type=int, default=0)
    parser.add_argument("--model_start_steps", type=int, default=0)
    parser.add_argument("--solver_start_steps", type=int, default=None)
    parser.add_argument("--warmup_action_std", type=float, default=0.35)
    parser.add_argument("--actor_forward_bias_start", type=float, default=0.5)
    parser.add_argument("--actor_forward_bias_anneal_epochs", type=int, default=200)
    parser.add_argument("--actor_action_hold_steps", type=int, default=1)
    parser.add_argument("--actor_vertical_action_limit", type=float, default=0.25)
    parser.add_argument("--solver_updates_per_step", type=int, default=1)
    parser.add_argument("--sac_batch_size", type=int, default=None)
    parser.add_argument("--real_fraction", type=float, default=None, help="固定真实数据占比；不填则使用线性调度。")
    parser.add_argument("--real_fraction_start", type=float, default=0.9, help="真实数据占比调度起点。0.9 表示真实:虚拟 = 9:1。")
    parser.add_argument("--real_fraction_final", type=float, default=0.5, help="真实数据占比调度终点。0.5 表示真实:虚拟 = 1:1。")
    parser.add_argument("--real_fraction_schedule_epochs", type=int, default=3000, help="真实数据占比从起点过渡到终点所需的 epoch 数。")
    parser.add_argument("--disable_alpha_autotune", dest="disable_alpha_autotune", action="store_true")
    parser.add_argument("--enable_alpha_autotune", dest="disable_alpha_autotune", action="store_false")
    parser.set_defaults(disable_alpha_autotune=False)
    parser.add_argument("--debug_step_logs", action="store_true", default=False)
    parser.add_argument("--logdir", type=str, default=None)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--video_interval", type=int, default=20)
    parser.add_argument("--video_max_steps", type=int, default=None)
    parser.add_argument("--video_fps", type=int, default=30)
    parser.add_argument("--video_format", type=str, default="mp4", choices=("avi", "mp4"))
    parser.add_argument("--video_width", type=int, default=2560)
    parser.add_argument("--video_height", type=int, default=1440)
    parser.add_argument("--video_crf", type=int, default=14)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--checkpoint", type=str, default=None)
    AppLauncher.add_app_launcher_args(parser)
    return parser

def _write_video_with_ffmpeg(frame_dir, video_path, fps, crf):
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise RuntimeError("ffmpeg is not installed or not available in PATH.")

    command = [
        ffmpeg_path,
        "-y",
        "-framerate",
        str(fps),
        "-i",
        os.path.join(frame_dir, "frame_%06d.png"),
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        str(crf),
        video_path,
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "ffmpeg failed to encode the video.")


class LiveTrainingVideoRecorder:
    def __init__(self, env, video_path, fps, video_format, video_crf):
        import cv2

        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        self.cv2 = cv2
        self.env = env
        self.video_path = video_path
        self.fps = fps
        self.video_format = video_format
        self.video_crf = video_crf
        self.writer = None
        self.frame_dir = None
        self.frame_index = 0
        self._opened = False

    def _open(self, frame):
        height, width = frame.shape[:2]
        if self.video_format == "avi":
            fourcc = self.cv2.VideoWriter_fourcc(*"MJPG")
            self.writer = self.cv2.VideoWriter(self.video_path, fourcc, float(self.fps), (width, height))
            if not self.writer.isOpened():
                raise RuntimeError(f"Failed to open video writer for {self.video_path}")
        else:
            ffmpeg_path = shutil.which("ffmpeg")
            if ffmpeg_path is None:
                fourcc = self.cv2.VideoWriter_fourcc(*"mp4v")
                self.writer = self.cv2.VideoWriter(self.video_path, fourcc, float(self.fps), (width, height))
                if not self.writer.isOpened():
                    raise RuntimeError(f"Failed to open fallback mp4 video writer for {self.video_path}")
            else:
                self.frame_dir = tempfile.mkdtemp(prefix="mbpo_live_video_frames_", dir=os.path.dirname(self.video_path))
        self._opened = True

    def capture(self):
        frame = self.env.render()
        if frame is None:
            return False
        if not self._opened:
            self._open(frame)
        frame_bgr = self.cv2.cvtColor(frame, self.cv2.COLOR_RGB2BGR)
        if self.writer is not None:
            self.writer.write(frame_bgr)
        else:
            self.cv2.imwrite(os.path.join(self.frame_dir, f"frame_{self.frame_index:06d}.png"), frame_bgr)
        self.frame_index += 1
        return True

    def close(self):
        if self.writer is not None:
            self.writer.release()
        elif self.frame_dir is not None:
            try:
                if self.frame_index > 0:
                    _write_video_with_ffmpeg(self.frame_dir, self.video_path, self.fps, self.video_crf)
            finally:
                shutil.rmtree(self.frame_dir, ignore_errors=True)
        return self.video_path


def main(args_cli):
    import torch
    from torch.utils.tensorboard import SummaryWriter

    from mbpo_env_adapter import BatchedMBPOAdapter
    from mbpo_core import SMBPOCore
    from mbpo_core.log import default_log as log
    from quadcopter_obstacles_env import QuadcopterObstaclesEnv, QuadcopterObstaclesEnvCfg

    env_cfg = QuadcopterObstaclesEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.auto_reset_done = False
    env_cfg.debug_vis = not bool(getattr(args_cli, "headless", False))
    env_cfg.reward_debug_interval = 200 if args_cli.debug_step_logs else 0
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        env_cfg.device = args_cli.device
    if args_cli.video_interval > 0:
        env_cfg.viewer_resolution = (args_cli.video_width, args_cli.video_height)
    if args_cli.video_max_steps is None:
        args_cli.video_max_steps = int(args_cli.video_fps * 10)

    env_render_mode = "rgb_array" if args_cli.video_interval > 0 else None
    env = QuadcopterObstaclesEnv(cfg=env_cfg, render_mode=env_render_mode)
    adapter = BatchedMBPOAdapter(env)

    resolved_solver_start_steps = (
        args_cli.warmup_epochs * args_cli.steps_per_epoch * args_cli.num_envs
        if args_cli.solver_start_steps is None
        else args_cli.solver_start_steps
    )
    resolved_buffer_min = 65_536 if args_cli.buffer_min is None else args_cli.buffer_min
    resolved_replay_buffer_max = 10_000_000 if args_cli.replay_buffer_max is None else args_cli.replay_buffer_max
    resolved_virt_buffer_max = 700_000 if args_cli.virt_buffer_max is None else args_cli.virt_buffer_max
    resolved_rollout_batch_size = 256 if args_cli.rollout_batch_size is None else args_cli.rollout_batch_size
    resolved_model_initial_steps = 20_000 if args_cli.model_initial_steps is None else args_cli.model_initial_steps
    resolved_model_steps = 5_000 if args_cli.model_steps is None else args_cli.model_steps
    resolved_sac_batch_size = 1_024 if args_cli.sac_batch_size is None else args_cli.sac_batch_size

    alg_cfg = SMBPOCore.Config()
    alg_cfg.steps_per_epoch = args_cli.steps_per_epoch
    alg_cfg.buffer_min = resolved_buffer_min
    alg_cfg.replay_buffer_max = resolved_replay_buffer_max
    alg_cfg.virt_buffer_max = resolved_virt_buffer_max
    alg_cfg.horizon = args_cli.rollout_horizon
    alg_cfg.rollout_batch_size = resolved_rollout_batch_size
    alg_cfg.max_episode_steps = env.unwrapped.max_episode_length
    alg_cfg.model_initial_steps = resolved_model_initial_steps
    alg_cfg.model_steps = resolved_model_steps
    alg_cfg.model_update_period = (
        args_cli.model_update_period
        if args_cli.model_update_period is not None
        else 10 * args_cli.steps_per_epoch * args_cli.num_envs
    )
    alg_cfg.warmup_only_model_fit = False
    model_update_interval_epochs = alg_cfg.model_update_period / max(
        args_cli.steps_per_epoch * args_cli.num_envs,
        1,
    )
    alg_cfg.model_start_epoch = args_cli.model_start_epoch
    alg_cfg.model_start_steps = args_cli.model_start_steps
    alg_cfg.solver_start_steps = resolved_solver_start_steps
    alg_cfg.warmup_epochs = args_cli.warmup_epochs
    alg_cfg.warmup_action_std = args_cli.warmup_action_std
    alg_cfg.actor_forward_bias_start = args_cli.actor_forward_bias_start
    alg_cfg.actor_forward_bias_anneal_epochs = args_cli.actor_forward_bias_anneal_epochs
    alg_cfg.actor_action_hold_steps = args_cli.actor_action_hold_steps
    alg_cfg.actor_vertical_action_limit = args_cli.actor_vertical_action_limit
    alg_cfg.solver_updates_per_step = args_cli.solver_updates_per_step
    if args_cli.real_fraction is None:
        alg_cfg.real_fraction = args_cli.real_fraction_start
        alg_cfg.real_fraction_final = args_cli.real_fraction_final
        alg_cfg.real_fraction_schedule_epochs = args_cli.real_fraction_schedule_epochs
    else:
        alg_cfg.real_fraction = args_cli.real_fraction
        alg_cfg.real_fraction_final = args_cli.real_fraction
        alg_cfg.real_fraction_schedule_epochs = 0
    alg_cfg.sac_cfg.batch_size = resolved_sac_batch_size
    alg_cfg.sac_cfg.autotune_alpha = not args_cli.disable_alpha_autotune
    alg_cfg.sac_cfg.use_lidar_cnn = True
    alg_cfg.sac_cfg.encoder_cfg.lidar_start_idx = adapter._idx_front_rays_start
    alg_cfg.sac_cfg.encoder_cfg.lidar_dim = int(env_cfg.num_obstacle_rays)
    alg_cfg.sac_cfg.critic_cfg.use_lidar_cnn = True
    alg_cfg.sac_cfg.critic_cfg.encoder_cfg.lidar_start_idx = adapter._idx_front_rays_start
    alg_cfg.sac_cfg.critic_cfg.encoder_cfg.lidar_dim = int(env_cfg.num_obstacle_rays)

    log_root = args_cli.logdir or os.path.join(TASK_DIR, "logs", "mbpo")
    os.makedirs(log_root, exist_ok=True)
    if args_cli.resume:
        if args_cli.checkpoint is None:
            raise ValueError("--resume requires --checkpoint")
        run_dir = os.path.dirname(os.path.abspath(args_cli.checkpoint))
    else:
        run_dir = os.path.join(log_root, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(run_dir, exist_ok=True)
    log.setup(run_dir)
    log.message(f"alpha_autotune: {alg_cfg.sac_cfg.autotune_alpha}")
    log.message(
        "resolved training defaults: "
        f"solver_start_steps={alg_cfg.solver_start_steps}, "
        f"warmup_epochs={args_cli.warmup_epochs}, "
        f"warmup_action_std={alg_cfg.warmup_action_std}, "
        f"actor_forward_bias={alg_cfg.actor_forward_bias_start}->0/"
        f"{alg_cfg.actor_forward_bias_anneal_epochs}ep, "
        f"actor_action_hold_steps={alg_cfg.actor_action_hold_steps}, "
        f"actor_vertical_action_limit={alg_cfg.actor_vertical_action_limit}, "
        f"buffer_min={alg_cfg.buffer_min}, "
        f"replay_buffer_max={alg_cfg.replay_buffer_max}, "
        f"virt_buffer_max={alg_cfg.virt_buffer_max}, "
        f"rollout_batch_size={alg_cfg.rollout_batch_size}, "
        f"model_initial_steps={alg_cfg.model_initial_steps}, "
        f"model_steps={alg_cfg.model_steps}, "
        f"model_update_period={alg_cfg.model_update_period}, "
        f"model_update_interval≈{model_update_interval_epochs:.1f}epochs, "
        f"sac_batch_size={alg_cfg.sac_cfg.batch_size}, "
        f"real_fraction={alg_cfg.real_fraction}->{alg_cfg.real_fraction_final} "
        f"over {alg_cfg.real_fraction_schedule_epochs} epochs"
    )
    writer = SummaryWriter(log_dir=run_dir)

    alg = SMBPOCore(
        alg_cfg,
        adapter,
        state_dim=adapter.state_dim,
        action_dim=adapter.action_dim,
        action_space=adapter.action_space,
        check_done_fn=adapter.check_done,
        check_violation_fn=adapter.check_violation,
    )
    alg.to(torch.device(adapter.device))
    alg.debug_step_logs = args_cli.debug_step_logs
    if hasattr(alg.solver, "debug_updates"):
        alg.solver.debug_updates = args_cli.debug_step_logs

    ckpt_path = os.path.join(run_dir, "mbpo_latest.pt")
    if args_cli.resume:
        checkpoint_path = os.path.abspath(args_cli.checkpoint)
        checkpoint = torch.load(checkpoint_path, map_location=adapter.device)
        alg.load_state_dict(checkpoint["alg_state"])
        if "replay_buffer" in checkpoint:
            alg.replay_buffer.load_state_dict(checkpoint["replay_buffer"])
        if "virt_buffer" in checkpoint:
            alg.virt_buffer.load_state_dict(checkpoint["virt_buffer"])
        log.message(f"Resumed MBPO checkpoint from {checkpoint_path}")
        alg.setup()
    else:
        with open(os.path.join(run_dir, "mbpo_env_cfg.pkl"), "wb") as f:
            pickle.dump(env_cfg, f)
        with open(os.path.join(run_dir, "mbpo_alg_cfg.pkl"), "wb") as f:
            pickle.dump(alg_cfg, f)
        alg.setup()

    live_video_recorder = None
    live_video_frame_limit = max(int(args_cli.video_max_steps), 1)

    for epoch in range(args_cli.epochs):
        log.message(f"Beginning MBPO epoch {epoch + 1}/{args_cli.epochs}")
        if (
            args_cli.video_interval > 0
            and live_video_recorder is None
            and ((epoch + 1) % args_cli.video_interval) == 0
        ):
            video_dir = os.path.join(run_dir, "videos")
            video_ext = "avi" if args_cli.video_format == "avi" else "mp4"
            video_path = os.path.join(video_dir, f"train_epoch_{epoch + 1:05d}.{video_ext}")
            live_video_recorder = LiveTrainingVideoRecorder(
                env,
                video_path,
                args_cli.video_fps,
                args_cli.video_format,
                args_cli.video_crf,
            )
            log.message(
                f"Recording live training video from current training state: "
                f"max_frames={live_video_frame_limit} path={video_path}"
            )

        def capture_live_training_frame():
            if live_video_recorder is None:
                return
            if live_video_recorder.frame_index < live_video_frame_limit:
                live_video_recorder.capture()

        steps_before = int(alg.steps_sampled.item())
        episodes_before = int(alg.episodes_sampled.item())
        successes_before = int(alg.n_successes.item())
        violations_before = int(alg.n_violations.item())
        collisions_before = int(alg.n_collisions.item())
        timeouts_before = int(alg.n_timeouts.item())
        too_low_before = int(alg.n_too_low.item())
        too_high_before = int(alg.n_too_high.item())
        out_of_bounds_before = int(alg.n_out_of_bounds.item())
        reward_sum_before = float(alg.env_reward_sum.item())
        reward_count_before = int(alg.env_reward_count.item())
        action_abs_sum_before = float(alg.action_abs_sum.item())
        action_element_count_before = int(alg.action_element_count.item())
        vx_cmd_sum_before = float(alg.vx_cmd_sum.item())
        vy_cmd_sum_before = float(alg.vy_cmd_sum.item())
        vz_cmd_sum_before = float(alg.vz_cmd_sum.item())
        cmd_count_before = int(alg.cmd_count.item())
        inside_map_count_before = int(alg.inside_map_count.item())
        inside_map_sample_count_before = int(alg.inside_map_sample_count.item())
        forward_action_before_bias_sum_before = float(alg.forward_action_before_bias_sum.item())
        forward_action_after_bias_sum_before = float(alg.forward_action_after_bias_sum.item())
        forward_action_count_before = int(alg.forward_action_count.item())
        forward_bias_sum_before = float(alg.forward_bias_sum.item())
        forward_bias_count_before = int(alg.forward_bias_count.item())
        try:
            alg.epoch(on_step=capture_live_training_frame if live_video_recorder is not None else None)
        except Exception:
            if live_video_recorder is not None:
                try:
                    saved_video_path = live_video_recorder.close()
                    log.message(f"Saved partial live training video to {saved_video_path}", flush=True)
                finally:
                    live_video_recorder = None
            log.message("Exception during MBPO epoch:", flush=True)
            log.message(traceback.format_exc(), timestamp=False, flush=True)
            raise
        avg_critic_loss = alg.average_recent_critic_loss()
        if avg_critic_loss is not None:
            log.message(f"Average recent critic loss: {avg_critic_loss}")
            writer.add_scalar("train/critic_loss", float(avg_critic_loss), epoch + 1)
            alg.recent_critic_losses.clear()
        avg_model_loss = alg.average_recent_model_loss()
        if avg_model_loss is not None:
            log.message(f"Average recent model loss: {avg_model_loss}")
            writer.add_scalar("train/model_loss", float(avg_model_loss), epoch + 1)
            model_stats = getattr(alg, "last_model_update_stats", None)
            if model_stats is not None:
                writer.add_scalar("model/loss_mean", float(model_stats["loss_mean"]), epoch + 1)
                writer.add_scalar("model/loss_min", float(model_stats["loss_min"]), epoch + 1)
                writer.add_scalar("model/loss_max", float(model_stats["loss_max"]), epoch + 1)
                writer.add_scalar("model/loss_p50", float(model_stats["loss_p50"]), epoch + 1)
                writer.add_scalar("model/loss_p90", float(model_stats["loss_p90"]), epoch + 1)
                writer.add_scalar("model/loss_p99", float(model_stats["loss_p99"]), epoch + 1)
                writer.add_scalar("model/loss_delta", float(model_stats["loss_delta"]), epoch + 1)
                writer.add_scalar("model/loss_first10", float(model_stats["start_loss_average"]), epoch + 1)
                writer.add_scalar("model/loss_last10", float(model_stats["end_loss_average"]), epoch + 1)
                writer.add_scalar("model/state_mse", float(model_stats["state_mse"]), epoch + 1)
                writer.add_scalar("model/reward_mse", float(model_stats["reward_mse"]), epoch + 1)
                writer.add_scalar("model/nll_mean", float(model_stats["nll_mean"]), epoch + 1)
                writer.add_scalar("model/nll_p50", float(model_stats["nll_p50"]), epoch + 1)
                writer.add_scalar("model/nll_p90", float(model_stats["nll_p90"]), epoch + 1)
                writer.add_scalar("model/nll_p99", float(model_stats["nll_p99"]), epoch + 1)
                writer.add_scalar("model/nll_max", float(model_stats["nll_max"]), epoch + 1)
                writer.add_scalar("model/nll_over_cap_rate", float(model_stats["nll_over_cap_rate"]), epoch + 1)
                writer.add_scalar("model/terminal_batch_rate", float(model_stats["terminal_rate"]), epoch + 1)
                writer.add_scalar("model/violation_batch_rate", float(model_stats["violation_rate"]), epoch + 1)
                writer.add_scalar("model/log_var_mean", float(model_stats["log_var_mean"]), epoch + 1)
                writer.add_scalar("model/log_var_min", float(model_stats["log_var_min"]), epoch + 1)
                writer.add_scalar("model/log_var_max", float(model_stats["log_var_max"]), epoch + 1)
                writer.add_scalar("model/log_var_gap_mean", float(model_stats["log_var_gap_mean"]), epoch + 1)
                writer.add_scalar("model/log_var_gap_min", float(model_stats["log_var_gap_min"]), epoch + 1)
                writer.add_scalar("model/log_var_gap_max", float(model_stats["log_var_gap_max"]), epoch + 1)
                writer.add_scalar("model/min_log_var_param_mean", float(model_stats["min_log_var_param_mean"]), epoch + 1)
                writer.add_scalar("model/max_log_var_param_mean", float(model_stats["max_log_var_param_mean"]), epoch + 1)
            alg.recent_model_losses.clear()
        avg_episode_length = alg.average_episode_length()
        if avg_episode_length is not None:
            writer.add_scalar("train/episode_length_mean", float(avg_episode_length), epoch + 1)
            alg.recent_episode_lengths.clear()
        epoch_steps = int(alg.steps_sampled.item()) - steps_before
        epoch_episodes = int(alg.episodes_sampled.item()) - episodes_before
        epoch_successes = int(alg.n_successes.item()) - successes_before
        epoch_violations = int(alg.n_violations.item()) - violations_before
        epoch_collisions = int(alg.n_collisions.item()) - collisions_before
        epoch_timeouts = int(alg.n_timeouts.item()) - timeouts_before
        epoch_too_low = int(alg.n_too_low.item()) - too_low_before
        epoch_too_high = int(alg.n_too_high.item()) - too_high_before
        epoch_out_of_bounds = int(alg.n_out_of_bounds.item()) - out_of_bounds_before
        epoch_reward_sum = float(alg.env_reward_sum.item()) - reward_sum_before
        epoch_reward_count = int(alg.env_reward_count.item()) - reward_count_before
        epoch_action_abs_sum = float(alg.action_abs_sum.item()) - action_abs_sum_before
        epoch_action_element_count = int(alg.action_element_count.item()) - action_element_count_before
        epoch_vx_cmd_sum = float(alg.vx_cmd_sum.item()) - vx_cmd_sum_before
        epoch_vy_cmd_sum = float(alg.vy_cmd_sum.item()) - vy_cmd_sum_before
        epoch_vz_cmd_sum = float(alg.vz_cmd_sum.item()) - vz_cmd_sum_before
        epoch_cmd_count = int(alg.cmd_count.item()) - cmd_count_before
        epoch_inside_map_count = int(alg.inside_map_count.item()) - inside_map_count_before
        epoch_inside_map_sample_count = int(alg.inside_map_sample_count.item()) - inside_map_sample_count_before
        epoch_forward_action_before_bias_sum = (
            float(alg.forward_action_before_bias_sum.item()) - forward_action_before_bias_sum_before
        )
        epoch_forward_action_after_bias_sum = (
            float(alg.forward_action_after_bias_sum.item()) - forward_action_after_bias_sum_before
        )
        epoch_forward_action_count = int(alg.forward_action_count.item()) - forward_action_count_before
        epoch_forward_bias_sum = float(alg.forward_bias_sum.item()) - forward_bias_sum_before
        epoch_forward_bias_count = int(alg.forward_bias_count.item()) - forward_bias_count_before
        epoch_reward_mean = epoch_reward_sum / max(epoch_reward_count, 1)
        epoch_mean_abs_action = epoch_action_abs_sum / max(epoch_action_element_count, 1)
        epoch_mean_vx_cmd = epoch_vx_cmd_sum / max(epoch_cmd_count, 1)
        epoch_mean_vy_cmd = epoch_vy_cmd_sum / max(epoch_cmd_count, 1)
        epoch_mean_vz_cmd = epoch_vz_cmd_sum / max(epoch_cmd_count, 1)
        epoch_inside_map_rate = epoch_inside_map_count / max(epoch_inside_map_sample_count, 1)
        epoch_mean_forward_action_before_bias = (
            epoch_forward_action_before_bias_sum / max(epoch_forward_action_count, 1)
        )
        epoch_mean_forward_action_after_bias = (
            epoch_forward_action_after_bias_sum / max(epoch_forward_action_count, 1)
        )
        epoch_forward_bias_value = epoch_forward_bias_sum / max(epoch_forward_bias_count, 1)
        epoch_success_rate = epoch_successes / max(epoch_episodes, 1)
        epoch_violation_rate = epoch_violations / max(epoch_episodes, 1)
        epoch_collision_rate = epoch_collisions / max(epoch_episodes, 1)
        epoch_timeout_rate = epoch_timeouts / max(epoch_episodes, 1)
        epoch_out_of_bounds_rate = epoch_out_of_bounds / max(epoch_episodes, 1)
        warmup_steps_done, warmup_steps_target, warmup_ratio = alg.warmup_progress()
        phase_name = "warmup" if alg.in_warmup() else "train"

        writer.add_scalar("train/replay_buffer_size", float(len(alg.replay_buffer)), epoch + 1)
        writer.add_scalar("train/virt_buffer_size", float(len(alg.virt_buffer)), epoch + 1)
        writer.add_scalar("train/real_fraction", float(alg.current_real_fraction()), epoch + 1)
        writer.add_scalar("train/warmup_steps_done", float(warmup_steps_done), epoch + 1)
        writer.add_scalar("train/warmup_steps_target", float(warmup_steps_target), epoch + 1)
        writer.add_scalar("train/warmup_progress", float(warmup_ratio), epoch + 1)
        writer.add_scalar("train/reward_mean", float(epoch_reward_mean), epoch + 1)
        writer.add_scalar("train/steps_per_epoch", float(epoch_steps), epoch + 1)
        writer.add_scalar("train/episodes_per_epoch", float(epoch_episodes), epoch + 1)
        writer.add_scalar("train/successes_per_epoch", float(epoch_successes), epoch + 1)
        writer.add_scalar("train/violations_per_epoch", float(epoch_violations), epoch + 1)
        writer.add_scalar("train/collisions_per_epoch", float(epoch_collisions), epoch + 1)
        writer.add_scalar("train/timeouts_per_epoch", float(epoch_timeouts), epoch + 1)
        writer.add_scalar("train/too_low_per_epoch", float(epoch_too_low), epoch + 1)
        writer.add_scalar("train/too_high_per_epoch", float(epoch_too_high), epoch + 1)
        writer.add_scalar("train/out_of_bounds_per_epoch", float(epoch_out_of_bounds), epoch + 1)
        writer.add_scalar("train/mean_abs_action", float(epoch_mean_abs_action), epoch + 1)
        writer.add_scalar("train/mean_vx_cmd", float(epoch_mean_vx_cmd), epoch + 1)
        writer.add_scalar("train/mean_vy_cmd", float(epoch_mean_vy_cmd), epoch + 1)
        writer.add_scalar("train/mean_vz_cmd", float(epoch_mean_vz_cmd), epoch + 1)
        writer.add_scalar("train/inside_map_rate", float(epoch_inside_map_rate), epoch + 1)
        writer.add_scalar("train/forward_bias_value", float(epoch_forward_bias_value), epoch + 1)
        writer.add_scalar(
            "train/mean_forward_action_before_bias",
            float(epoch_mean_forward_action_before_bias),
            epoch + 1,
        )
        writer.add_scalar(
            "train/mean_forward_action_after_bias",
            float(epoch_mean_forward_action_after_bias),
            epoch + 1,
        )
        writer.add_scalar("train/success_rate_epoch", float(epoch_success_rate), epoch + 1)
        writer.add_scalar("train/violation_rate_epoch", float(epoch_violation_rate), epoch + 1)
        writer.add_scalar("train/collision_rate_epoch", float(epoch_collision_rate), epoch + 1)
        writer.add_scalar("train/timeout_rate_epoch", float(epoch_timeout_rate), epoch + 1)
        writer.add_scalar("train/out_of_bounds_rate_epoch", float(epoch_out_of_bounds_rate), epoch + 1)
        writer.add_scalar("train/episodes_sampled", float(alg.episodes_sampled.item()), epoch + 1)
        writer.add_scalar("train/steps_sampled", float(alg.steps_sampled.item()), epoch + 1)
        writer.add_scalar("train/violations", float(alg.n_violations.item()), epoch + 1)
        writer.add_scalar("train/collisions", float(alg.n_collisions.item()), epoch + 1)
        writer.add_scalar("train/timeouts", float(alg.n_timeouts.item()), epoch + 1)
        writer.add_scalar("train/too_low", float(alg.n_too_low.item()), epoch + 1)
        writer.add_scalar("train/too_high", float(alg.n_too_high.item()), epoch + 1)
        writer.add_scalar("train/out_of_bounds", float(alg.n_out_of_bounds.item()), epoch + 1)
        writer.add_scalar("train/successes", float(alg.n_successes.item()), epoch + 1)
        writer.add_scalar("train/success_rate", alg.success_rate(), epoch + 1)
        writer.add_scalar("train/violation_rate", alg.violation_rate(), epoch + 1)
        writer.add_scalar("train/collision_rate", alg.collision_rate(), epoch + 1)
        writer.add_scalar("train/timeout_rate", alg.timeout_rate(), epoch + 1)
        writer.add_scalar("train/out_of_bounds_rate", alg.out_of_bounds_rate(), epoch + 1)
        writer.flush()
        if live_video_recorder is not None and live_video_recorder.frame_index >= live_video_frame_limit:
            try:
                saved_video_path = live_video_recorder.close()
                log.message(f"Saved live training video to {saved_video_path}")
            except Exception as exc:
                log.message(f"Live training video capture failed: {exc}")
            finally:
                live_video_recorder = None
        log.message(
            f"Phase={phase_name} Warmup={warmup_steps_done}/{warmup_steps_target} "
            f"({warmup_ratio:.3f}) "
            f"Episodes={alg.episodes_sampled.item()} Steps={alg.steps_sampled.item()} "
            f"Successes={alg.n_successes.item()} Violations={alg.n_violations.item()} "
            f"Collisions={alg.n_collisions.item()} Timeouts={alg.n_timeouts.item()} "
            f"TooLow={alg.n_too_low.item()} TooHigh={alg.n_too_high.item()} "
            f"OutOfBounds={alg.n_out_of_bounds.item()} "
            f"SuccessRate={alg.success_rate():.3f} EpochSuccessRate={epoch_success_rate:.3f}"
        )
        log.message(
            f"Epoch {epoch + 1} summary: phase={phase_name} "
            f"warmup={warmup_steps_done}/{warmup_steps_target} ({warmup_ratio:.3f}) "
            f"episodes={epoch_episodes} successes={epoch_successes} "
            f"violations={epoch_violations} collisions={epoch_collisions} timeouts={epoch_timeouts} "
            f"too_low={epoch_too_low} too_high={epoch_too_high} out_of_bounds={epoch_out_of_bounds} "
            f"mean_abs_action={epoch_mean_abs_action:.3f} "
            f"mean_cmd=({epoch_mean_vx_cmd:+.3f},{epoch_mean_vy_cmd:+.3f},{epoch_mean_vz_cmd:+.3f}) "
            f"forward_bias={epoch_forward_bias_value:+.3f} "
            f"forward_action=({epoch_mean_forward_action_before_bias:+.3f}->"
            f"{epoch_mean_forward_action_after_bias:+.3f}) "
            f"inside_map_rate={epoch_inside_map_rate:.3f} "
            f"success_rate={epoch_success_rate:.3f} violation_rate={epoch_violation_rate:.3f} "
            f"collision_rate={epoch_collision_rate:.3f} timeout_rate={epoch_timeout_rate:.3f} "
            f"out_of_bounds_rate={epoch_out_of_bounds_rate:.3f}"
        )
        if ((epoch + 1) % args_cli.save_interval) == 0 or (epoch + 1) == args_cli.epochs:
            torch.save(
                {
                    "alg_state": alg.state_dict(),
                    "replay_buffer": alg.replay_buffer.state_dict(),
                    "virt_buffer": alg.virt_buffer.state_dict(),
                    "epoch": epoch + 1,
                },
                ckpt_path,
            )
            log.message(f"Saved checkpoint to {ckpt_path}")

    if live_video_recorder is not None:
        try:
            saved_video_path = live_video_recorder.close()
            log.message(f"Saved final partial live training video to {saved_video_path}")
        except Exception as exc:
            log.message(f"Final live training video capture failed: {exc}")

    writer.close()
    env.close()


def main_cli():
    from isaaclab.app import AppLauncher

    faulthandler.enable()
    parser = build_parser()
    args_cli = parser.parse_args()
    if args_cli.video_interval > 0:
        setattr(args_cli, "enable_cameras", True)
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    try:
        main(args_cli)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main_cli()
