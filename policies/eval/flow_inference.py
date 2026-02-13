from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper


@dataclass
class FlowSampleCfg:
    """Sampling config for flow-matching ODE inference."""
    ode_steps: int = 10                    # number of ODE steps
    method: str = "euler"                  # torchdiffeq method name if using flow_matching
    use_flow_matching_solver: bool = True  # whether to use flow-matching ODE solver
    noise_std: float = 1.0                 # std for x0 ~ N(0, I)


class _VelocityWrapper(torch.nn.Module):
    """
    Adapts your policy.predict_velocity(x_t, t, obs_memory) to the interface expected by ODE solvers:
        v(x, t, **extras)
    """
    def __init__(self, policy: torch.nn.Module):
        super().__init__()
        self.policy = policy

    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras) -> torch.Tensor:
        # ODESolver may pass t as scalar tensor; your policy expects (B,) tensor.
        if not torch.is_tensor(t):
            t = torch.tensor(t, device=x.device, dtype=x.dtype)
        if t.ndim == 0:
            t = t.expand(x.shape[0])
        elif t.ndim == 1 and t.shape[0] != x.shape[0]:
            t = t.expand(x.shape[0])

        obs_memory = extras["obs_memory"]
        return self.policy.predict_velocity(x_t=x, t=t, obs_memory=obs_memory)


@torch.no_grad()
def sample_action_sequence(
    *,
    policy: torch.nn.Module,
    obs_memory: torch.Tensor,
    pred_horizon: int,
    action_dim: int,
    cfg: FlowSampleCfg,
    device: torch.device,
) -> torch.Tensor:
    """
    Returns:
        actions_abs10: (B, pred_horizon, action_dim) tensor
    """
    B = obs_memory.shape[0]
    x = torch.randn((B, pred_horizon, action_dim), device=device) * float(cfg.noise_std)

    # flow_matching docs: solver.sample returns last timestep when return_intermediates=False
    class FMWrapper(ModelWrapper):
        def __init__(self, model: torch.nn.Module):
            super().__init__(model)

        def forward(self, x: torch.Tensor, t: torch.Tensor, **extras) -> torch.Tensor:
            return model(x, t, **extras)

    model = _VelocityWrapper(policy)
    velocity_model = FMWrapper(model)
    solver = ODESolver(velocity_model=velocity_model)

    # Fixed-step Euler: step_size = 1 / ode_steps, integrate over [0, 1]
    step_size = 1.0 / float(max(1, int(cfg.ode_steps)))
    time_grid = torch.tensor([0.0, 1.0], device=device)

    x1 = solver.sample(
        x_init=x,
        step_size=step_size,
        method=str(cfg.method),
        time_grid=time_grid,
        obs_memory=obs_memory,
    )
    return x1