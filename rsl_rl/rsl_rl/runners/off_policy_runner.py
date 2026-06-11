from __future__ import annotations

import os
import statistics
import time
from collections import deque

import torch

from rsl_rl.algorithms import SAC
from rsl_rl.modules import ActorCriticSAC
from rsl_rl.storage.replay_buffer import ReplayBuffer
from rsl_rl.utils import resolve_obs_groups, store_code_state

try:
    from isaaclab.utils.math import quat_apply_inverse
except Exception:
    quat_apply_inverse = None


class OffPolicyRunner:
    """Off-policy runner for SAC-style algorithms on vectorized IsaacLab environments."""

    def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu"):
        self.cfg = train_cfg
        self.alg_cfg = dict(train_cfg["algorithm"])
        self.policy_cfg = dict(train_cfg["policy"])
        self.device = device
        self.env = env
        self.log_dir = log_dir
        self.writer = None
        self.disable_logs = False
        self.logger_type = str(self.cfg.get("logger", "tensorboard")).lower()
        self.current_learning_iteration = 0
        self.tot_timesteps = 0
        self.tot_time = 0
        self.save_interval = int(self.cfg["save_interval"])
        self.num_steps_per_env = int(self.cfg["num_steps_per_env"])
        self.max_iterations = int(self.cfg.get("max_iterations", 0))
        self.replay_buffer_sample_interval = max(int(self.cfg.get("replay_buffer_sample_interval", 1)), 1)
        warmup_iterations = self.cfg.get("warmup_iterations", None)
        if warmup_iterations is not None:
            self.random_steps = int(warmup_iterations) * int(self.env.num_envs) * self.num_steps_per_env
            replay_buffer_size = int(self.cfg["replay_buffer_size"])
            self.learning_starts = min(max(self.random_steps // self.replay_buffer_sample_interval, 1), replay_buffer_size)
        else:
            self.random_steps = int(self.cfg.get("random_steps", 10_000))
            self.learning_starts = int(self.cfg.get("learning_starts", self.random_steps))
        self._warmup_done_reported = self.random_steps <= 0
        self._print_warmup_budget()
        self.gradient_steps_per_iteration = int(self.cfg.get("gradient_steps_per_iteration", self.num_steps_per_env))
        self.random_forward_action_mean = float(self.cfg.get("random_forward_action_mean", 0.0))
        self.random_forward_action_std = float(self.cfg.get("random_forward_action_std", 1.0))
        self.random_lateral_action_std = float(self.cfg.get("random_lateral_action_std", 1.0))
        self.random_vertical_action_std = float(self.cfg.get("random_vertical_action_std", 1.0))
        self.warmup_action_mode = str(self.cfg.get("warmup_action_mode", "random")).lower()
        self.warmup_target_action_scale = float(self.cfg.get("warmup_target_action_scale", 0.6))
        self.warmup_lateral_noise_std = float(self.cfg.get("warmup_lateral_noise_std", 0.15))
        self.warmup_vertical_gain = float(self.cfg.get("warmup_vertical_gain", 0.6))
        self.warmup_vertical_noise_std = float(self.cfg.get("warmup_vertical_noise_std", self.random_vertical_action_std))
        self.warmup_min_height = float(self.cfg.get("warmup_min_height", 0.8))
        self.git_status_repos = []

        obs = self.env.get_observations().to(self.device)
        self.cfg["obs_groups"] = resolve_obs_groups(obs, self.cfg["obs_groups"], ["critic"])
        self.alg = self._construct_algorithm(obs)
        self.replay_buffer = ReplayBuffer(
            obs,
            self.env.num_envs,
            self.cfg["replay_buffer_size"],
            storage_device=self.cfg.get("replay_buffer_device", "cpu"),
            sample_device=self.device,
        )

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        self._prepare_logging_writer()
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf,
                high=int(self.env.max_episode_length),
            )

        obs = self.env.get_observations().to(self.device)
        self.train_mode()
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        successbuffer = deque(maxlen=100)
        ep_infos = []
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + int(num_learning_iterations)
        for it in range(start_iter, tot_iter):
            start = time.time()
            for _ in range(self.num_steps_per_env):
                env_step_index = self.tot_timesteps // self.env.num_envs
                self.alg.policy.update_normalization(obs)
                if self.tot_timesteps < self.random_steps:
                    actions = self._sample_warmup_actions()
                else:
                    with torch.no_grad():
                        actions = self.alg.act(obs)
                next_obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                next_obs = next_obs.to(self.device)
                
                # Handling terminal observations correctly for SAC/off-policy bootstrapping
                real_next_obs = next_obs
                if "terminal_obs" in extras and "terminal_obs_mask" in extras:
                    mask = extras["terminal_obs_mask"].to(self.device)
                    trunc_idx = mask.nonzero(as_tuple=False).flatten()
                    if trunc_idx.numel() > 0:
                        real_next_obs = {}
                        for k in next_obs.keys():
                            real_next_obs[k] = next_obs[k].clone()
                            if k in extras["terminal_obs"]:
                                real_next_obs[k][trunc_idx] = extras["terminal_obs"][k].to(self.device)

                rewards = rewards.to(self.device)
                dones = dones.to(self.device)
                should_store_transition = ((env_step_index + 1) % self.replay_buffer_sample_interval) == 0
                if should_store_transition or torch.any(dones):
                    self.replay_buffer.add(obs, actions, rewards, dones, real_next_obs, extras)
                obs = next_obs

                cur_reward_sum += rewards
                cur_episode_length += 1
                if "episode" in extras:
                    ep_infos.append(extras["episode"])
                elif "log" in extras:
                    ep_infos.append(extras["log"])
                new_ids = (dones > 0).nonzero(as_tuple=False).flatten()
                if new_ids.numel() > 0:
                    rewbuffer.extend(cur_reward_sum[new_ids].detach().cpu().numpy().tolist())
                    lenbuffer.extend(cur_episode_length[new_ids].detach().cpu().numpy().tolist())
                    if "episode_success" in extras:
                        successbuffer.extend(
                            extras["episode_success"][new_ids].float().detach().cpu().numpy().tolist()
                        )
                    cur_reward_sum[new_ids] = 0
                    cur_episode_length[new_ids] = 0
                self.tot_timesteps += self.env.num_envs

            collection_time = time.time() - start
            start = time.time()
            if self.tot_timesteps >= self.random_steps and not self._warmup_done_reported:
                print(f"[INFO] SAC warmup finished at {self.tot_timesteps} timesteps; starting gradient updates.", flush=True)
                self._warmup_done_reported = True
            if self.tot_timesteps >= self.random_steps and len(self.replay_buffer) >= max(self.learning_starts, self.alg.batch_size):
                loss_dict = self.alg.update(self.replay_buffer, self.gradient_steps_per_iteration)
            else:
                loss_dict = {}
            learn_time = time.time() - start
            self.current_learning_iteration = it
            self.tot_time += collection_time + learn_time

            if self.log_dir is not None:
                self.log(
                    {
                        "it": it,
                        "tot_iter": tot_iter,
                        "num_learning_iterations": num_learning_iterations,
                        "collection_time": collection_time,
                        "learn_time": learn_time,
                        "loss_dict": loss_dict,
                        "rewbuffer": rewbuffer,
                        "lenbuffer": lenbuffer,
                        "successbuffer": successbuffer,
                        "ep_infos": ep_infos,
                    }
                )
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))
                if it == start_iter:
                    store_code_state(self.log_dir, self.git_status_repos)
            ep_infos.clear()

        if self.log_dir is not None:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))
        self.close()

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        if self.writer is None:
            return
        it = locs["it"]
        fps = int(self.env.num_envs * self.num_steps_per_env / max(locs["collection_time"] + locs["learn_time"], 1e-6))
        scalars = {}
        for key, value in locs["loss_dict"].items():
            scalars[f"Loss/{key}"] = value
        scalars["Train/replay_buffer_size"] = len(self.replay_buffer)
        scalars["Perf/total_fps"] = fps
        scalars["Perf/collection_time"] = locs["collection_time"]
        scalars["Perf/learning_time"] = locs["learn_time"]
        if locs["rewbuffer"]:
            scalars["Train/mean_reward"] = statistics.mean(locs["rewbuffer"])
            scalars["Train/mean_episode_length"] = statistics.mean(locs["lenbuffer"])
        if locs["successbuffer"]:
            scalars["Train/success_rate"] = statistics.mean(locs["successbuffer"])
        for ep_info in locs["ep_infos"]:
            for key, value in ep_info.items():
                if not isinstance(value, torch.Tensor):
                    value = torch.tensor(value, device=self.device)
                scalar_key = key if "/" in key else "Episode/" + key
                scalars[scalar_key] = value.float().mean()
        if self.logger_type == "wandb" and hasattr(self.writer, "add_scalars"):
            self.writer.add_scalars(scalars, global_step=it)
        else:
            for key, value in scalars.items():
                self.writer.add_scalar(key, value, it)
        self.writer.flush()

        mean_reward = statistics.mean(locs["rewbuffer"]) if locs["rewbuffer"] else 0.0
        success_rate = statistics.mean(locs["successbuffer"]) if locs["successbuffer"] else 0.0
        title = f" SAC iteration {it}/{locs['tot_iter']} ".center(width)
        print(
            f"{'#' * width}\n"
            f"{title}\n\n"
            f"{'Computation:':>{pad}} {fps} steps/s "
            f"(collection {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"
            f"{'Replay buffer:':>{pad}} {len(self.replay_buffer)}\n"
            f"{'Mean reward:':>{pad}} {mean_reward:.3f}\n"
            f"{'Success rate:':>{pad}} {success_rate:.3f}\n"
            f"{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"
        )

    def save(self, path: str, infos=None):
        torch.save(
            {
                "model_state_dict": self.alg.policy.state_dict(),
                "algorithm_state_dict": self.alg.state_dict(),
                "iter": self.current_learning_iteration,
                "infos": infos,
            },
            path,
        )
        if self.writer is not None and self.logger_type in ["neptune", "wandb"] and not self.disable_logs:
            self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path: str, load_optimizer: bool = True, map_location: str | None = None):
        loaded = torch.load(path, weights_only=False, map_location=map_location)
        self.alg.policy.load_state_dict(loaded["model_state_dict"])
        if load_optimizer and "algorithm_state_dict" in loaded:
            self.alg.load_state_dict(loaded["algorithm_state_dict"])
        self.current_learning_iteration = int(loaded.get("iter", 0))
        return loaded.get("infos")

    def get_inference_policy(self, device=None):
        self.eval_mode()
        if device is not None:
            self.alg.policy.to(device)
        return self.alg.policy.act_inference

    def train_mode(self):
        self.alg.policy.train()

    def eval_mode(self):
        self.alg.policy.eval()

    def close(self):
        if self.writer is None:
            return
        if hasattr(self.writer, "stop"):
            self.writer.stop()
        else:
            self.writer.close()
        self.writer = None

    def add_git_repo_to_log(self, repo_file_path):
        self.git_status_repos.append(repo_file_path)

    def _print_warmup_budget(self):
        if self.max_iterations <= 0:
            return
        steps_per_iteration = self.env.num_envs * self.num_steps_per_env
        if steps_per_iteration <= 0:
            return
        total_steps_budget = self.max_iterations * steps_per_iteration
        warmup_finish_iteration = (self.random_steps + steps_per_iteration - 1) // steps_per_iteration
        if total_steps_budget <= self.random_steps:
            print(
                "[WARN] SAC warmup is longer than the configured run: "
                f"random_steps={self.random_steps}, total_steps={total_steps_budget}, "
                f"num_envs={self.env.num_envs}, num_steps_per_env={self.num_steps_per_env}. "
                f"Increase max_iterations to more than {warmup_finish_iteration} or reduce random_steps.",
                flush=True,
            )
        else:
            print(
                "[INFO] SAC warmup budget: "
                f"random_steps={self.random_steps}, warmup_finish_iteration={warmup_finish_iteration}, "
                f"total_steps={total_steps_budget}.",
                flush=True,
            )

    def _construct_algorithm(self, obs):
        if self.cfg.get("empirical_normalization") is not None:
            self.policy_cfg.setdefault("actor_obs_normalization", self.cfg["empirical_normalization"])
            self.policy_cfg.setdefault("critic_obs_normalization", self.cfg["empirical_normalization"])
        self.policy_cfg = {key: value for key, value in self.policy_cfg.items() if value is not None}
        policy_class = globals()[self.policy_cfg.pop("class_name")]
        policy = policy_class(obs, self.cfg["obs_groups"], self.env.num_actions, **self.policy_cfg).to(self.device)
        alg_class = globals()[self.alg_cfg.pop("class_name")]
        return alg_class(policy, device=self.device, **self.alg_cfg)

    def _prepare_logging_writer(self):
        if self.log_dir is not None and self.writer is None:
            if self.logger_type == "neptune":
                from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter

                self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "wandb":
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter

                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "tensorboard":
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise ValueError("Logger type not found. Please choose 'neptune', 'wandb' or 'tensorboard'.")

    def _sample_warmup_actions(self) -> torch.Tensor:
        if self.warmup_action_mode in {"target", "target_directed", "goal"}:
            actions = self._sample_target_warmup_actions()
            if actions is not None:
                return actions
        return self._sample_random_warmup_actions()

    def _sample_random_warmup_actions(self) -> torch.Tensor:
        actions = torch.zeros((self.env.num_envs, self.env.num_actions), device=self.device)
        actions[:, 0] = (
            torch.randn(self.env.num_envs, device=self.device) * self.random_forward_action_std
            + self.random_forward_action_mean
        )
        if self.env.num_actions > 1:
            actions[:, 1] = torch.randn(self.env.num_envs, device=self.device) * self.random_lateral_action_std
        if self.env.num_actions > 2:
            actions[:, 2] = torch.randn(self.env.num_envs, device=self.device) * self.random_vertical_action_std
        if self.env.num_actions > 3:
            actions[:, 3:] = torch.empty(
                (self.env.num_envs, self.env.num_actions - 3),
                device=self.device,
            ).uniform_(-1.0, 1.0)
        return actions.clamp(-1.0, 1.0)

    def _sample_target_warmup_actions(self) -> torch.Tensor | None:
        if quat_apply_inverse is None:
            return None

        unwrapped_env = getattr(self.env, "unwrapped", self.env)
        target_positions = getattr(unwrapped_env, "_target_positions_w", None)
        cfg = getattr(unwrapped_env, "cfg", None)
        if target_positions is None or cfg is None:
            return None

        try:
            robot = unwrapped_env.robot
            root_pos_w = robot.data.root_pos_w.to(self.device)
            root_quat_w = robot.data.root_quat_w.to(self.device)
        except Exception:
            return None

        target_positions = target_positions.to(self.device)
        if root_pos_w.shape[0] != self.env.num_envs or target_positions.shape[0] != self.env.num_envs:
            return None

        actions = torch.zeros((self.env.num_envs, self.env.num_actions), device=self.device)
        target_vec_b = quat_apply_inverse(root_quat_w, target_positions - root_pos_w)
        target_xy_b = target_vec_b[:, :2]
        target_xy_norm = torch.linalg.norm(target_xy_b, dim=1, keepdim=True).clamp_min(1.0e-6)
        target_xy_dir = target_xy_b / target_xy_norm

        if self.env.num_actions > 0:
            actions[:, 0] = target_xy_dir[:, 0] * self.warmup_target_action_scale
        if self.env.num_actions > 1:
            actions[:, 1] = target_xy_dir[:, 1] * self.warmup_target_action_scale
            actions[:, 1] += torch.randn(self.env.num_envs, device=self.device) * self.warmup_lateral_noise_std
        if self.env.num_actions > 2:
            flight_min_height = float(getattr(cfg, "flight_min_height", 0.1))
            flight_max_height = float(getattr(cfg, "flight_max_height", 4.0))
            cmd_vel_z_max = max(float(getattr(cfg, "cmd_vel_z_max", 1.0)), 1.0e-6)
            min_warmup_height = max(self.warmup_min_height, flight_min_height + 0.2)
            max_warmup_height = max(min_warmup_height, flight_max_height - 0.2)
            target_height = target_positions[:, 2].clamp(min=min_warmup_height, max=max_warmup_height)
            z_action = (target_height - root_pos_w[:, 2]) * self.warmup_vertical_gain / cmd_vel_z_max
            z_action += torch.randn(self.env.num_envs, device=self.device) * self.warmup_vertical_noise_std
            actions[:, 2] = z_action
        if self.env.num_actions > 3:
            actions[:, 3:] = torch.empty(
                (self.env.num_envs, self.env.num_actions - 3),
                device=self.device,
            ).uniform_(-0.2, 0.2)
        return actions.clamp(-1.0, 1.0)
