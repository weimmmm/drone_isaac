from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn


@dataclass
class SafetyFilterConfig:
    num_depth_columns: int = 16
    risk_threshold: float = 0.5
    lateral_step: float = 0.25
    forward_brake: float = 0.20
    reverse_speed: float = 0.35
    correction_iters: int = 4
    center_crop_ratio: float = 0.6
    near_clip: float = 0.2
    far_clip: float = 8.0


class SafetyFilterNetwork(nn.Module):
    """SIGN-style one-step risk predictor from depth image and candidate action."""

    def __init__(self, action_dim: int = 3, num_columns: int = 16):
        super().__init__()
        self.num_columns = num_columns
        self.depth_net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.action_net = nn.Sequential(
            nn.Linear(action_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 + 8, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        self.last_depth_vector: torch.Tensor | None = None

    def depth_to_vector(self, depth_metric: torch.Tensor, center_crop_ratio: float = 0.6) -> torch.Tensor:
        if depth_metric.ndim != 3:
            raise ValueError(f"Expected depth tensor [B, H, W], got shape {tuple(depth_metric.shape)}")

        batch_size, height, width = depth_metric.shape
        crop_height = max(1, int(round(height * center_crop_ratio)))
        top = max(0, (height - crop_height) // 2)
        bottom = min(height, top + crop_height)
        cropped = depth_metric[:, top:bottom]

        column_edges = torch.linspace(0, width, self.num_columns + 1, device=depth_metric.device).long()
        min_values = []
        for idx in range(self.num_columns):
            start = int(column_edges[idx].item())
            end = max(start + 1, int(column_edges[idx + 1].item()))
            min_values.append(cropped[:, :, start:end].amin(dim=(1, 2)))
        return torch.stack(min_values, dim=1)

    def forward(self, depth_metric: torch.Tensor, action: torch.Tensor, center_crop_ratio: float = 0.6) -> torch.Tensor:
        depth_vector = self.depth_to_vector(depth_metric, center_crop_ratio=center_crop_ratio)
        self.last_depth_vector = depth_vector
        depth_feat = self.depth_net(depth_vector.unsqueeze(1))
        action_feat = self.action_net(torch.clamp(action, -1.0, 1.0))
        combined = torch.cat([depth_feat, action_feat], dim=1)
        return self.classifier(combined)


class SafetyActionFilter:
    """Runtime action corrector that applies a trained safety network."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str | torch.device = "cpu",
        config: SafetyFilterConfig | None = None,
    ):
        self.device = torch.device(device)
        self.config = config or SafetyFilterConfig()
        self.model = SafetyFilterNetwork(action_dim=3, num_columns=self.config.num_depth_columns).to(self.device)
        checkpoint = torch.load(Path(checkpoint_path), map_location=self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    def _proximity_to_metric(self, depth_proximity: torch.Tensor) -> torch.Tensor:
        near = self.config.near_clip
        far = self.config.far_clip
        return near + (1.0 - depth_proximity.clamp(0.0, 1.0)) * (far - near)

    @torch.no_grad()
    def predict_risk(self, depth_proximity: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        depth_metric = self._proximity_to_metric(depth_proximity.to(self.device))
        return self.model(depth_metric, action.to(self.device), center_crop_ratio=self.config.center_crop_ratio)

    @torch.no_grad()
    def correct_actions(self, depth_proximity: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        proposed = action.to(self.device).clone()
        risk = self.predict_risk(depth_proximity, proposed)
        corrected = proposed.clone()
        initial_risk = risk.clone()

        safe_mask = risk.squeeze(-1) <= self.config.risk_threshold
        if safe_mask.all():
            return corrected, {
                "raw_action": proposed,
                "corrected_action": corrected,
                "risk": risk,
                "was_corrected": torch.zeros_like(safe_mask, dtype=torch.bool),
            }

        depth_vec = self.model.last_depth_vector
        if depth_vec is None:
            raise RuntimeError("Safety filter did not produce a depth vector.")

        half = depth_vec.shape[1] // 2
        left_clearance = depth_vec[:, :half].amin(dim=1)
        right_clearance = depth_vec[:, half:].amin(dim=1)
        side_dir = torch.where(left_clearance >= right_clearance, 1.0, -1.0)

        for _ in range(self.config.correction_iters):
            active = risk.squeeze(-1) > self.config.risk_threshold
            if not active.any():
                break

            corrected[active, 1] = torch.clamp(
                corrected[active, 1] + self.config.lateral_step * side_dir[active], -1.0, 1.0
            )
            corrected[active, 0] = torch.clamp(corrected[active, 0] - self.config.forward_brake, -1.0, 1.0)
            corrected[active, 2] = torch.clamp(corrected[active, 2] * 0.5, -1.0, 1.0)
            risk = self.predict_risk(depth_proximity, corrected)

        still_risky = risk.squeeze(-1) > self.config.risk_threshold
        if still_risky.any():
            corrected[still_risky, 0] = -self.config.reverse_speed
            corrected[still_risky, 2] = 0.0
            risk = self.predict_risk(depth_proximity, corrected)

        return corrected, {
            "raw_action": proposed,
            "corrected_action": corrected,
            "risk": risk,
            "initial_risk": initial_risk,
            "was_corrected": ~safe_mask,
            "depth_vector": self.model.last_depth_vector,
        }
