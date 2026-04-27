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
        self.current_learning_iteration = 0
        self.tot_timesteps = 0
        self.tot_time = 0
        self.save_interval = int(self.cfg["save_interval"])
        self.num_steps_per_env = int(self.cfg["num_steps_per_env"])
        self.replay_buffer_sample_interval = max(int(self.cfg.get("replay_buffer_sample_interval", 1)), 1)
        warmup_iterations = self.cfg.get("warmup_iterations", None)
        if warmup_iterations is not None:
            self.random_steps = int(warmup_iterations) * int(self.env.num_envs) * self.num_steps_per_env
            replay_buffer_size = int(self.cfg["replay_buffer_size"])
            self.learning_starts = min(max(self.random_steps // self.replay_buffer_sample_interval, 1), replay_buffer_size)
        else:
            self.random_steps = int(self.cfg.get("random_steps", 10_000))
            self.learning_starts = int(self.cfg.get("learning_starts", self.random_steps))
        self.gradient_steps_per_iteration = int(self.cfg.get("gradient_steps_per_iteration", self.num_steps_per_env))
        self.random_forward_action_mean = float(self.cfg.get("random_forward_action_mean", 0.0))
        self.random_forward_action_std = float(self.cfg.get("random_forward_action_std", 1.0))
        self.random_lateral_action_std = float(self.cfg.get("random_lateral_action_std", 1.0))
        self.random_vertical_action_std = float(self.cfg.get("random_vertical_action_std", 1.0))
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
                rewards = rewards.to(self.device)
                dones = dones.to(self.device)
                should_store_transition = ((env_step_index + 1) % self.replay_buffer_sample_interval) == 0
                if should_store_transition or torch.any(dones):
                    self.replay_buffer.add(obs, actions, rewards, dones, next_obs, extras)
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

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        if self.writer is None:
            return
        it = locs["it"]
        fps = int(self.env.num_envs * self.num_steps_per_env / max(locs["collection_time"] + locs["learn_time"], 1e-6))
        for key, value in locs["loss_dict"].items():
            self.writer.add_scalar(f"Loss/{key}", value, it)
        self.writer.add_scalar("Train/replay_buffer_size", len(self.replay_buffer), it)
        self.writer.add_scalar("Perf/total_fps", fps, it)
        self.writer.add_scalar("Perf/collection_time", locs["collection_time"], it)
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], it)
        if locs["rewbuffer"]:
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), it)
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), it)
        if locs["successbuffer"]:
            self.writer.add_scalar("Train/success_rate", statistics.mean(locs["successbuffer"]), it)
        for ep_info in locs["ep_infos"]:
            for key, value in ep_info.items():
                if not isinstance(value, torch.Tensor):
                    value = torch.tensor(value, device=self.device)
                self.writer.add_scalar(key if "/" in key else "Episode/" + key, value.float().mean(), it)
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

    def add_git_repo_to_log(self, repo_file_path):
        self.git_status_repos.append(repo_file_path)

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
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

    def _sample_warmup_actions(self) -> torch.Tensor:
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
