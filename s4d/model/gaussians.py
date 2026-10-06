"""Container for a decoded dynamic Gaussian scene."""

from __future__ import annotations

from dataclasses import dataclass

import torch

SCENE_GROUP = 0
DYNAMIC_GROUP = 1


@dataclass
class GaussianSet:
    xyz: torch.Tensor  # (B, N, 3) centres at t0, world frame
    scales: torch.Tensor  # (B, N, 3)
    quats: torch.Tensor  # (B, N, 4) wxyz
    opacity: torch.Tensor  # (B, N)
    rgb: torch.Tensor  # (B, N, 3) in [0, 1]
    delta01: torch.Tensor  # (B, N, 3) displacement t0 -> t1 (zero for the scene group)
    delta12: torch.Tensor  # (B, N, 3) displacement t1 -> t2
    group: torch.Tensor  # (N,) long, SCENE_GROUP or DYNAMIC_GROUP

    @property
    def num_gaussians(self) -> int:
        return int(self.xyz.shape[1])

    def xyz_at(self, t: int) -> torch.Tensor:
        if t == 0:
            return self.xyz
        if t == 1:
            return self.xyz + self.delta01
        if t == 2:
            return self.xyz + self.delta01 + self.delta12
        raise ValueError(t)

    def xyz_sequence(self) -> torch.Tensor:
        """(B, 3, N, 3) centres at t0, t1, t2."""
        return torch.stack([self.xyz_at(t) for t in range(3)], dim=1)

    def displacement(self, src: int, dst: int) -> torch.Tensor:
        return self.xyz_at(dst) - self.xyz_at(src)

    def detach_geometry(self) -> GaussianSet:
        """Detach everything except the displacements (used by the motion loss)."""
        return GaussianSet(
            self.xyz.detach(),
            self.scales.detach(),
            self.quats.detach(),
            self.opacity.detach(),
            self.rgb.detach(),
            self.delta01,
            self.delta12,
            self.group,
        )
