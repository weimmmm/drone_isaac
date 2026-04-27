from __future__ import annotations

import torch
import torch.nn.functional as F


class SAC:
    def __init__(
        self,
        policy,
        device: str = "cpu",
        learning_rate: float = 3.0e-4,
        critic_learning_rate: float | None = None,
        alpha_learning_rate: float | None = None,
        gamma: float = 0.99,
        tau: float = 0.005,
        batch_size: int = 1024,
        target_entropy: float | None = None,
        init_alpha: float = 0.2,
        min_alpha: float = 0.02,
        autotune_alpha: bool = True,
        actor_update_interval: int = 1,
        max_grad_norm: float | None = 1.0,
        **kwargs,
    ):
        del kwargs
        self.policy = policy
        self.device = torch.device(device)
        self.learning_rate = float(learning_rate)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.batch_size = int(batch_size)
        self.autotune_alpha = bool(autotune_alpha)
        self.actor_update_interval = max(int(actor_update_interval), 1)
        self.update_counter = 0
        self.max_grad_norm = max_grad_norm
        self.target_entropy = float(target_entropy) if target_entropy is not None else -float(policy.num_actions)
        self.min_log_alpha = torch.tensor(float(min_alpha)).log().to(self.device)

        self.actor_parameters = list(self.policy.actor_encoder.parameters()) + list(self.policy.actor.parameters())
        self.critic_parameters = (
            list(self.policy.q1_encoder.parameters())
            + list(self.policy.q2_encoder.parameters())
            + list(self.policy.q1.parameters())
            + list(self.policy.q2.parameters())
        )
        self.actor_optimizer = torch.optim.Adam(self.actor_parameters, lr=self.learning_rate)
        critic_lr = self.learning_rate if critic_learning_rate is None else float(critic_learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic_parameters, lr=critic_lr)
        self.log_alpha = torch.tensor(float(init_alpha)).log().to(self.device).requires_grad_(True)
        alpha_lr = self.learning_rate if alpha_learning_rate is None else float(alpha_learning_rate)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def act(self, obs):
        return self.policy.act(obs)

    def update(self, replay_buffer, gradient_steps: int = 1) -> dict[str, float]:
        metrics = []
        actor_metrics = []
        for _ in range(int(gradient_steps)):
            batch = replay_buffer.sample(self.batch_size)
            obs = batch["obs"]
            actions = batch["actions"]
            rewards = batch["rewards"]
            dones = batch["dones"]
            time_outs = batch["time_outs"]
            next_obs = batch["next_obs"]
            not_done = 1.0 - torch.clamp(dones - time_outs, min=0.0, max=1.0)

            with torch.no_grad():
                next_actions, next_log_prob = self.policy.sample(next_obs)
                target_q1, target_q2 = self.policy.target_q_values(next_obs, next_actions)
                target_q = torch.min(target_q1, target_q2) - self.alpha.detach() * next_log_prob
                target = rewards + not_done * self.gamma * target_q

            q1, q2 = self.policy.q_values(obs, actions)
            critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.critic_parameters, float(self.max_grad_norm))
            self.critic_optimizer.step()

            self.update_counter += 1
            should_update_actor = (self.update_counter % self.actor_update_interval) == 0
            if should_update_actor:
                new_actions, log_prob = self.policy.sample(obs)
                q1_pi, q2_pi = self.policy.q_values(obs, new_actions)
                actor_loss = (self.alpha.detach() * log_prob - torch.min(q1_pi, q2_pi)).mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.actor_parameters, float(self.max_grad_norm))
                self.actor_optimizer.step()

                if self.autotune_alpha:
                    alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
                    self.alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    self.alpha_optimizer.step()
                    self.log_alpha.data.clamp_(min=float(self.min_log_alpha.item()))
                else:
                    alpha_loss = torch.zeros((), device=self.device)
                actor_metrics.append(
                    {
                        "actor": float(actor_loss.detach().item()),
                        "alpha_loss": float(alpha_loss.detach().item()),
                    }
                )
            else:
                actor_loss = torch.zeros((), device=self.device)
                alpha_loss = torch.zeros((), device=self.device)

            self.policy.update_targets(self.tau)
            metrics.append(
                {
                    "critic": float(critic_loss.detach().item()),
                    "actor": float(actor_loss.detach().item()),
                    "alpha": float(self.alpha.detach().item()),
                    "alpha_loss": float(alpha_loss.detach().item()),
                    "q_mean": float(torch.min(q1, q2).detach().mean().item()),
                    "target_q_mean": float(target.detach().mean().item()),
                    "actor_updates": float(should_update_actor),
                }
            )

        result = {key: sum(item[key] for item in metrics) / len(metrics) for key in metrics[0]}
        if actor_metrics:
            result["actor"] = sum(item["actor"] for item in actor_metrics) / len(actor_metrics)
            result["alpha_loss"] = sum(item["alpha_loss"] for item in actor_metrics) / len(actor_metrics)
        return result

    def state_dict(self) -> dict:
        return {
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach(),
            "update_counter": self.update_counter,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state_dict["critic_optimizer"])
        self.alpha_optimizer.load_state_dict(state_dict["alpha_optimizer"])
        self.log_alpha.data.copy_(state_dict["log_alpha"].to(self.device))
        self.log_alpha.data.clamp_(min=float(self.min_log_alpha.item()))
        self.update_counter = int(state_dict.get("update_counter", 0))
