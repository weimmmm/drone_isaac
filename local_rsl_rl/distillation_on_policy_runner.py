from __future__ import annotations

import os
from collections import deque
import time

import torch
import torch.nn.functional as F
from tensordict import TensorDict

from rsl_rl.runners import OnPolicyRunner
from rsl_rl.utils import split_and_pad_trajectories
from rsl_rl.utils import store_code_state
from rsl_rl.utils import unpad_trajectories


class DistillationOnPolicyRunner(OnPolicyRunner):
    """PPO runner with an annealed teacher distillation loss for standalone students."""

    def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device="cpu"):
        super().__init__(env, train_cfg, log_dir=log_dir, device=device)

        self.distill_loss_coef_init = float(self.cfg.get("distill_loss_coef_init", 0.0))
        self.distill_loss_coef_final = float(self.cfg.get("distill_loss_coef_final", 0.0))
        self.distill_loss_anneal_steps = max(int(self.cfg.get("distill_loss_anneal_steps", 1)), 1)
        self.distill_num_learning_epochs = max(int(self.cfg.get("distill_num_learning_epochs", 1)), 1)
        self.distill_num_mini_batches = max(int(self.cfg.get("distill_num_mini_batches", 1)), 1)
        self.distill_loss_type = str(self.cfg.get("distill_loss_type", "mse")).lower()
        if self.distill_loss_type not in {"mse", "huber"}:
            raise ValueError(f"Unsupported distillation loss type: {self.distill_loss_type}")

        distill_lr = self.cfg.get("distill_learning_rate")
        if distill_lr is None:
            distill_lr = self.alg.learning_rate
        self.distill_learning_rate = float(distill_lr)
        self.distill_sample_counter = 0

    def _update_latest_symlink(self) -> None:
        latest_path = getattr(self, "latest_symlink_path", None)
        if not latest_path or self.log_dir is None:
            return
        try:
            if os.path.lexists(latest_path):
                if os.path.isdir(latest_path) and not os.path.islink(latest_path):
                    return
                os.unlink(latest_path)
            os.symlink(self.log_dir, latest_path)
        except OSError as exc:
            print(f"[WARN] Failed to update latest symlink '{latest_path}': {exc}")

    def _get_distill_loss_coef(self) -> float:
        progress = min(max(float(self.distill_sample_counter) / self.distill_loss_anneal_steps, 0.0), 1.0)
        return (1.0 - progress) * self.distill_loss_coef_init + progress * self.distill_loss_coef_final

    def _distill_loss(self, predicted_actions: torch.Tensor, teacher_actions: torch.Tensor) -> torch.Tensor:
        if self.distill_loss_type == "huber":
            return F.huber_loss(predicted_actions, teacher_actions)
        return F.mse_loss(predicted_actions, teacher_actions)

    def _masked_distill_loss(
        self, predicted_actions: torch.Tensor, teacher_actions: torch.Tensor, masks: torch.Tensor | None = None
    ) -> torch.Tensor:
        if masks is not None:
            predicted_actions = predicted_actions[masks]
            teacher_actions = teacher_actions[masks]
        return self._distill_loss(predicted_actions, teacher_actions)

    def _predict_distillation_actions(
        self,
        obs_batch,
        masks_batch: torch.Tensor | None = None,
        hidden_states=None,
    ) -> torch.Tensor:
        if not self.alg.policy.is_recurrent:
            return self.alg.policy.act_inference(obs_batch)
        features = self.alg.policy.get_actor_features(obs_batch)
        out_mem = self.alg.policy.memory_a(features, masks_batch, hidden_states).squeeze(0)
        return self.alg.policy.actor(out_mem)

    def _recurrent_distillation_generator(
        self,
        obs_buffer: dict[str, list[torch.Tensor]],
        teacher_action_buffer: list[torch.Tensor],
        num_mini_batches: int,
        num_epochs: int,
    ):
        num_steps = len(teacher_action_buffer)
        stacked_obs = TensorDict(
            {key: torch.stack(value, dim=0) for key, value in obs_buffer.items()},
            batch_size=[num_steps, self.env.num_envs],
            device=self.device,
        )
        stacked_teacher_actions = torch.stack(teacher_action_buffer, dim=0)
        dones = self.alg.storage.dones[:num_steps]

        padded_obs, trajectory_masks = split_and_pad_trajectories(stacked_obs, dones)
        padded_teacher_actions, _ = split_and_pad_trajectories(stacked_teacher_actions, dones)

        if self.env.num_envs < num_mini_batches:
            raise ValueError(
                f"Recurrent distillation requires num_envs >= num_mini_batches, got "
                f"num_envs={self.env.num_envs}, num_mini_batches={num_mini_batches}."
            )

        mini_batch_size = self.env.num_envs // num_mini_batches
        dones_flat = dones.squeeze(-1)
        last_was_done = torch.zeros_like(dones_flat, dtype=torch.bool)
        last_was_done[1:] = dones_flat[:-1]
        last_was_done[0] = True

        for _ in range(num_epochs):
            first_traj = 0
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size
                trajectories_batch_size = torch.sum(last_was_done[:, start:stop])
                last_traj = first_traj + trajectories_batch_size

                obs_batch = padded_obs[:, first_traj:last_traj]
                teacher_batch = padded_teacher_actions[:, first_traj:last_traj]
                masks_batch = trajectory_masks[:, first_traj:last_traj]

                last_was_done_env = last_was_done.permute(1, 0)
                hid_a_batch = [
                    saved_hidden_states.permute(2, 0, 1, 3)[last_was_done_env][first_traj:last_traj]
                    .transpose(1, 0)
                    .contiguous()
                    for saved_hidden_states in self.alg.storage.saved_hidden_states_a
                ]
                hid_a_batch = hid_a_batch[0] if len(hid_a_batch) == 1 else hid_a_batch

                yield obs_batch, teacher_batch, masks_batch, hid_a_batch
                first_traj = last_traj

    def _reduce_distill_gradients(self) -> None:
        grads = [param.grad.view(-1) for param in self.alg.policy.parameters() if param.grad is not None]
        if not grads:
            return
        all_grads = torch.cat(grads)
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size

        offset = 0
        for param in self.alg.policy.parameters():
            if param.grad is None:
                continue
            numel = param.numel()
            param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
            offset += numel

    def _run_distillation_update(self, obs_buffer: dict[str, list[torch.Tensor]], teacher_action_buffer: list[torch.Tensor]):
        coef = self._get_distill_loss_coef()
        if coef <= 0.0 or not teacher_action_buffer:
            return {
                "distill_loss": 0.0,
                "distill_coef": coef,
            }

        total_samples = sum(actions.shape[0] for actions in teacher_action_buffer)
        self.distill_sample_counter += total_samples
        mean_distill_loss = 0.0
        num_updates = 0
        original_lrs = [param_group["lr"] for param_group in self.alg.optimizer.param_groups]
        for param_group in self.alg.optimizer.param_groups:
            param_group["lr"] = self.distill_learning_rate

        try:
            debug_logs = bool(self.cfg.get("debug_logs", False))
            if self.alg.policy.is_recurrent:
                batch_generator = self._recurrent_distillation_generator(
                    obs_buffer,
                    teacher_action_buffer,
                    self.distill_num_mini_batches,
                    self.distill_num_learning_epochs,
                )
                for batch_idx, (obs_batch, teacher_batch, masks_batch, hidden_states_batch) in enumerate(batch_generator):
                    if debug_logs:
                        print(
                            f"[DEBUG][distill.runner] distill_batch_ready idx={batch_idx} obs_batch={tuple(obs_batch.batch_size)} teacher={tuple(teacher_batch.shape)} masks={tuple(masks_batch.shape)}",
                            flush=True,
                        )
                    predicted_actions = self._predict_distillation_actions(
                        obs_batch, masks_batch=masks_batch, hidden_states=hidden_states_batch
                    )
                    if debug_logs:
                        print(
                            f"[DEBUG][distill.runner] distill_forward_done idx={batch_idx} predicted={tuple(predicted_actions.shape)}",
                            flush=True,
                        )
                    teacher_batch_unpadded = unpad_trajectories(teacher_batch, masks_batch).squeeze(0)
                    distill_loss = self._distill_loss(predicted_actions, teacher_batch_unpadded)
                    total_loss = coef * distill_loss
                    if debug_logs:
                        print(
                            f"[DEBUG][distill.runner] distill_loss_done idx={batch_idx} loss={float(distill_loss.detach()):.6f}",
                            flush=True,
                        )

                    self.alg.optimizer.zero_grad()
                    total_loss.backward()
                    if debug_logs:
                        print(f"[DEBUG][distill.runner] distill_backward_done idx={batch_idx}", flush=True)
                    if self.is_distributed:
                        self._reduce_distill_gradients()
                    torch.nn.utils.clip_grad_norm_(self.alg.policy.parameters(), self.alg.max_grad_norm)
                    self.alg.optimizer.step()
                    if debug_logs:
                        print(f"[DEBUG][distill.runner] distill_step_done idx={batch_idx}", flush=True)

                    mean_distill_loss += distill_loss.item()
                    num_updates += 1
            else:
                flat_obs = {key: torch.cat(value, dim=0) for key, value in obs_buffer.items()}
                flat_teacher_actions = torch.cat(teacher_action_buffer, dim=0)
                mini_batch_size = max(total_samples // self.distill_num_mini_batches, 1)
                for _ in range(self.distill_num_learning_epochs):
                    permutation = torch.randperm(total_samples, device=self.device)
                    for start in range(0, total_samples, mini_batch_size):
                        batch_ids = permutation[start : start + mini_batch_size]
                        obs_batch = {key: value[batch_ids] for key, value in flat_obs.items()}
                        teacher_batch = flat_teacher_actions[batch_ids]

                        predicted_actions = self._predict_distillation_actions(obs_batch)
                        distill_loss = self._distill_loss(predicted_actions, teacher_batch)
                        total_loss = coef * distill_loss

                        self.alg.optimizer.zero_grad()
                        total_loss.backward()
                        if self.is_distributed:
                            self._reduce_distill_gradients()
                        torch.nn.utils.clip_grad_norm_(self.alg.policy.parameters(), self.alg.max_grad_norm)
                        self.alg.optimizer.step()

                        mean_distill_loss += distill_loss.item()
                        num_updates += 1
        finally:
            for param_group, lr in zip(self.alg.optimizer.param_groups, original_lrs, strict=False):
                param_group["lr"] = lr

        if num_updates > 0:
            mean_distill_loss /= num_updates

        return {
            "distill_loss": mean_distill_loss,
            "distill_coef": coef,
        }

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):  # noqa: C901
        self._prepare_logging_writer()
        # Prefer explicit runner cfg debug flag, then fall back to wrapped env cfg.
        env_cfg = getattr(self.env, "cfg", None)
        if env_cfg is None and hasattr(self.env, "unwrapped"):
            env_cfg = getattr(self.env.unwrapped, "cfg", None)
        debug_logs = bool(self.cfg.get("debug_logs", False) or getattr(env_cfg, "debug_logs", False))
        self.alg.debug_logs = debug_logs

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs = self.env.get_observations().to(self.device)
        self.train_mode()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        if self.alg.rnd:
            erewbuffer = deque(maxlen=100)
            irewbuffer = deque(maxlen=100)
            cur_ereward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
            cur_ireward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            if debug_logs:
                print(
                    f"[DEBUG][distill.runner] iter={it} rollout_start num_steps_per_env={self.num_steps_per_env}",
                    flush=True,
                )
            start = time.time()
            distill_obs_buffer: dict[str, list[torch.Tensor]] = {key: [] for key in obs.keys()}
            teacher_action_buffer: list[torch.Tensor] = []

            with torch.inference_mode():
                for rollout_step in range(self.num_steps_per_env):
                    # 这里必须对当前观测做真正的快照，而不只是 detach。
                    # 特别是 policy_image 对应的历史深度帧缓冲会在后续 step 中被原地复用/覆盖，
                    # 如果这里只保存 view，蒸馏阶段读到的就不是“当时那一帧”的观测了。
                    current_obs = {key: value.detach().clone() for key, value in obs.items()}
                    actions = self.alg.act(obs)
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    intrinsic_rewards = self.alg.intrinsic_rewards if self.alg.rnd else None

                    teacher_actions = extras.get("teacher_actions")
                    if teacher_actions is not None:
                        for key, value in current_obs.items():
                            distill_obs_buffer[key].append(value)
                        teacher_action_buffer.append(teacher_actions.detach().to(self.device).clone())

                    if self.log_dir is not None:
                        if "episode" in extras:
                            ep_infos.append(extras["episode"])
                        elif "log" in extras:
                            ep_infos.append(extras["log"])
                        if self.alg.rnd:
                            cur_ereward_sum += rewards
                            cur_ireward_sum += intrinsic_rewards  # type: ignore
                            cur_reward_sum += rewards + intrinsic_rewards
                        else:
                            cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                        if self.alg.rnd:
                            erewbuffer.extend(cur_ereward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            irewbuffer.extend(cur_ireward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            cur_ereward_sum[new_ids] = 0
                            cur_ireward_sum[new_ids] = 0
                    if debug_logs and ((rollout_step + 1) % max(1, self.num_steps_per_env // 4) == 0):
                        print(
                            f"[DEBUG][distill.runner] iter={it} rollout_step={rollout_step + 1}/{self.num_steps_per_env}",
                            flush=True,
                        )

                stop = time.time()
                collection_time = stop - start
                start = stop
                self.alg.compute_returns(obs)
                if debug_logs:
                    print(
                        f"[DEBUG][distill.runner] iter={it} rollout_done collection_time={collection_time:.3f}s",
                        flush=True,
                    )

            if debug_logs:
                print(f"[DEBUG][distill.runner] iter={it} ppo_update_start", flush=True)
            ppo_start = time.time()
            loss_dict = self.alg.update()
            ppo_stop = time.time()
            if debug_logs:
                print(
                    f"[DEBUG][distill.runner] iter={it} ppo_update_done elapsed={ppo_stop - ppo_start:.3f}s",
                    flush=True,
                )

            if debug_logs:
                print(
                    f"[DEBUG][distill.runner] iter={it} distill_update_start batches={len(teacher_action_buffer)}",
                    flush=True,
                )
            distill_start = time.time()
            loss_dict.update(self._run_distillation_update(distill_obs_buffer, teacher_action_buffer))
            distill_stop = time.time()
            if debug_logs:
                print(
                    f"[DEBUG][distill.runner] iter={it} distill_update_done elapsed={distill_stop - distill_start:.3f}s",
                    flush=True,
                )

            stop = time.time()
            learn_time = stop - start
            if debug_logs:
                print(
                    f"[DEBUG][distill.runner] iter={it} update_done learn_time={learn_time:.3f}s",
                    flush=True,
                )
            self.current_learning_iteration = it
            if self.log_dir is not None and not self.disable_logs:
                self.log(locals())
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            ep_infos.clear()
            if it == start_iter and not self.disable_logs:
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        super().log(locs, width=width, pad=pad)
        if self.writer is not None:
            self.writer.add_scalar("Distill/coef", locs["loss_dict"]["distill_coef"], locs["it"])
            self.writer.add_scalar("Distill/loss", locs["loss_dict"]["distill_loss"], locs["it"])

    def save(self, path: str, infos=None):
        saved_dict = {
            "model_state_dict": self.alg.policy.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "distill_sample_counter": self.distill_sample_counter,
            "infos": infos,
        }
        if hasattr(self.alg, "rnd") and self.alg.rnd:
            saved_dict["rnd_state_dict"] = self.alg.rnd.state_dict()
            saved_dict["rnd_optimizer_state_dict"] = self.alg.rnd_optimizer.state_dict()
        torch.save(saved_dict, path)
        self._update_latest_symlink()

        if self.logger_type in ["neptune", "wandb"] and not self.disable_logs:
            self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path: str, load_optimizer: bool = True, map_location: str | None = None):
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)
        resumed_training = self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])
        if hasattr(self.alg, "rnd") and self.alg.rnd:
            self.alg.rnd.load_state_dict(loaded_dict["rnd_state_dict"])
        if load_optimizer and resumed_training:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            if hasattr(self.alg, "rnd") and self.alg.rnd:
                self.alg.rnd_optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])
        if resumed_training:
            self.current_learning_iteration = loaded_dict["iter"]
            self.distill_sample_counter = int(loaded_dict.get("distill_sample_counter", 0))
        return loaded_dict["infos"]
