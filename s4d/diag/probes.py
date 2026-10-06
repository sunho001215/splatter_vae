"""Linear probes (ridge regression R^2) and cross-view retrieval on frozen states."""

from __future__ import annotations

import torch


def ridge_fit(x: torch.Tensor, y: torch.Tensor, lam: float = 1e-2) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Closed-form ridge with standardised inputs. Returns (W, x_mean, x_std) so that y ~ ((x-m)/s) @ W[:-1] + W[-1]."""
    x = x.double()
    y = y.double()
    mean, std = x.mean(0), x.std(0).clamp_min(1e-6)
    z = torch.cat(((x - mean) / std, torch.ones(len(x), 1, dtype=x.dtype)), 1)
    reg = lam * torch.eye(z.shape[1], dtype=x.dtype)
    reg[-1, -1] = 0.0
    W = torch.linalg.solve(z.T @ z + reg, z.T @ y)
    return W, mean, std


def ridge_predict(W: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    z = torch.cat(((x.double() - mean) / std, torch.ones(len(x), 1, dtype=torch.float64)), 1)
    return z @ W


def r2_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    target = target.double()
    ss_res = ((pred - target) ** 2).sum()
    ss_tot = ((target - target.mean(0)) ** 2).sum().clamp_min(1e-12)
    return float(1.0 - ss_res / ss_tot)


def probe_targets(probe_state: torch.Tensor, stride: torch.Tensor) -> dict[str, torch.Tensor]:
    """Probe state contains hand xyz, gripper, object xyz/quaternion.

    ``stride`` is the adjacent-frame time interval in seconds. Velocity is m/s.
    """
    hand = probe_state[:, :, 0:3]
    return {
        "hand_pos": hand[:, 1],
        "obj_pos": probe_state[:, 1, 4:7],
        "hand_vel": (hand[:, 2] - hand[:, 0]) / (2.0 * stride.float()[:, None]),
    }


def fit_and_score(
    train_states: torch.Tensor, train_targets: dict, eval_sets: dict[str, tuple[torch.Tensor, dict]], lam: float = 1e-2
) -> dict[str, float]:
    """train_states (N,D); eval_sets name -> (states, targets). Returns {f"r2_{target}_{set}": value}."""
    out = {}
    for name, y in train_targets.items():
        W, m, s = ridge_fit(train_states, y, lam)
        for set_name, (xs, ys) in eval_sets.items():
            out[f"r2_{name}_{set_name}"] = r2_score(ridge_predict(W, m, s, xs), ys[name])
    return out


def retrieval_top1(query: torch.Tensor, gallery: torch.Tensor) -> float:
    """query/gallery (M,D) states of the same M windows from two cameras; top-1 accuracy by cosine similarity."""
    q = torch.nn.functional.normalize(query.float(), dim=-1)
    g = torch.nn.functional.normalize(gallery.float(), dim=-1)
    nn_idx = (q @ g.T).argmax(1)
    return float((nn_idx == torch.arange(len(q))).float().mean())


def cross_view_retrieval(states: torch.Tensor, n_train_cams: int) -> dict[str, float]:
    """states (M, V_all, D) with the first n_train_cams columns being training cameras.

    Averages top-1 over all ordered train->train pairs and over held-out->train pairs.
    """
    M, V, _ = states.shape
    tt, et = [], []
    for a in range(V):
        for b in range(n_train_cams):
            if a == b:
                continue
            acc = retrieval_top1(states[:, a], states[:, b])
            (tt if a < n_train_cams else et).append(acc)
    out = {"retrieval_top1_train": sum(tt) / max(1, len(tt)), "retrieval_num_states": float(M)}
    if et:
        out["retrieval_top1_heldout"] = sum(et) / len(et)
    return out
