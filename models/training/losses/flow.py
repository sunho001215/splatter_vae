import torch
import torch.nn.functional as F

FLOW_PAIR_WEIGHTS = (0.4, 0.4, 0.2)

def compute_optical_flow_loss(
    predicted_flows: torch.Tensor,
    target_flows: torch.Tensor,
    rendered_coverage: torch.Tensor,
    predicted_valid_mask: torch.Tensor,
    foreground_mask: torch.Tensor,
    alpha_threshold: float = 0.01,
    smooth_l1_beta: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # Robust alpha-blended flow supervision over finite, foreground, visible pixels.
    # All source correspondence and coverage inputs are detached defensively so
    # opacity or source geometry cannot reduce the objective by hiding motion.
    if predicted_flows.shape != target_flows.shape:
        raise ValueError(
            f"Predicted and target flow shapes differ: {predicted_flows.shape} and {target_flows.shape}."
        )
    if predicted_flows.dim() != 6 or predicted_flows.shape[3] != 2:
        raise ValueError(f"Expected flow as (B,3,A,2,H,W), got {tuple(predicted_flows.shape)}.")
    expected_auxiliary = (*predicted_flows.shape[:3], 1, *predicted_flows.shape[-2:])
    for name, value in (
        ("rendered_coverage", rendered_coverage),
        ("predicted_valid_mask", predicted_valid_mask),
        ("foreground_mask", foreground_mask),
    ):
        if tuple(value.shape) != expected_auxiliary:
            raise ValueError(
                f"Expected {name} as {expected_auxiliary}, got {tuple(value.shape)}."
            )

    target = target_flows.detach().to(
        device=predicted_flows.device,
        dtype=predicted_flows.dtype,
    )
    coverage = rendered_coverage.detach().to(
        device=predicted_flows.device,
        dtype=predicted_flows.dtype,
    ).clamp(0.0, 1.0).squeeze(3)
    renderer_valid = predicted_valid_mask.detach().to(
        device=predicted_flows.device,
        dtype=torch.bool,
    ).squeeze(3)
    foreground = foreground_mask.detach().to(
        device=predicted_flows.device,
        dtype=torch.bool,
    ).squeeze(3)

    target_finite = torch.isfinite(target).all(dim=3)
    predicted_finite = torch.isfinite(predicted_flows).all(dim=3)
    visible = coverage > float(alpha_threshold)
    valid = target_finite & predicted_finite & renderer_valid & foreground & visible
    weights = coverage * valid.to(dtype=predicted_flows.dtype)

    safe_target = torch.where(torch.isfinite(target), target, torch.zeros_like(target))
    safe_predicted = torch.nan_to_num(
        predicted_flows,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    component_penalty = F.smooth_l1_loss(
        safe_predicted,
        safe_target,
        beta=float(smooth_l1_beta),
        reduction="none",
    ).mean(dim=3)
    endpoint_error = (safe_predicted - safe_target).square().sum(dim=3).sqrt()

    pair_losses = []
    metrics: dict[str, torch.Tensor] = {}
    pair_names = ("01", "12", "02")
    for pair_idx, pair_name in enumerate(pair_names):
        pair_weights = weights[:, pair_idx]
        denominator = pair_weights.sum().clamp_min(1.0)
        pair_losses.append(
            (component_penalty[:, pair_idx] * pair_weights).sum() / denominator
        )
        metrics[f"flow_epe_{pair_name}"] = (
            endpoint_error[:, pair_idx] * pair_weights
        ).sum() / denominator

    loss_weights = predicted_flows.new_tensor(FLOW_PAIR_WEIGHTS)
    flow_loss = (torch.stack(pair_losses) * loss_weights).sum()
    eligible = target_finite & foreground
    metrics["flow_visible_fraction"] = (
        valid.to(dtype=predicted_flows.dtype).sum()
        / eligible.to(dtype=predicted_flows.dtype).sum().clamp_min(1.0)
    )
    return flow_loss, metrics


