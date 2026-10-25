"""Hard negatives for the encoder pilot hn (2026-10-25): a second, frame-disjoint window from the same episode, kept only
if the ROBOT moved between the two windows' last frames.

The rule reads robot state alone, the four values the RL policy receives as proprioception: end-effector (hand)
position in metres and the gripper opening on a [0, 1] scale. Object state, pixels and every other simulator field are
never read, so changing them cannot change a decision. The same rule applies to DROID robot states (end-effector
position and the normalised gripper position) through ``robot_state``.

For a sample (episode, first frame t0, stride s) the candidates are the first frames t0' of every stride-s window inside
the episode with 2s + 2 <= |t0' - t0| <= max_gap simulator steps (the windows share no frame). A candidate is valid if
the hand moved by >= min_hand_m or the gripper opening changed by >= min_gripper between frame t0 + 2s and frame
t0' + 2s; one valid candidate is drawn uniformly. Without a valid candidate the negative is invalid (masked out).
"""

from __future__ import annotations

import numpy as np
import torch

T_WINDOW = 3
# robot-state columns of each regime's probe_state: Meta-World obs[0:4] = hand xyz, gripper opening (the RL proprio
# indices 0-3, s4d/rl/env.py); DROID probe_state = cartesian_position (xyz, Euler xyz), gripper_position (PointWorld
# conversion, s4d/data/droid/convert.py)
ROBOT_COLUMNS = {"metaworld": (0, 1, 2, 3), "droid": (0, 1, 2, 6)}
DEFAULTS = {"max_gap": 40, "min_hand_m": 0.01, "min_gripper": 0.1}


def robot_state(probe_state, regime: str = "metaworld"):
    """(..., Dp) probe / observation state -> (..., 4) robot state [hand x, y, z, gripper in [0, 1]]."""
    cols = list(ROBOT_COLUMNS[regime])
    return probe_state[..., cols]


def robot_moved(a, b, min_hand_m: float, min_gripper: float):
    """Elementwise over leading dims of (..., 4) robot states: hand displacement >= min_hand_m (metres) or gripper change
    >= min_gripper. Works on NumPy arrays and tensors."""
    if torch.is_tensor(a):
        hand = torch.linalg.vector_norm(a[..., :3] - b[..., :3], dim=-1)
        grip = (a[..., 3] - b[..., 3]).abs()
    else:
        hand = np.linalg.norm(np.asarray(a)[..., :3] - np.asarray(b)[..., :3], axis=-1)
        grip = np.abs(np.asarray(a)[..., 3] - np.asarray(b)[..., 3])
    return (hand >= min_hand_m) | (grip >= min_gripper)


def candidate_starts(t0: int, stride: int, length: int, max_gap: int) -> np.ndarray:
    """First frames t0' of the stride-``stride`` windows inside an episode of ``length`` frames with
    2 * stride + 2 <= |t0' - t0| <= max_gap."""
    last = length - 1 - (T_WINDOW - 1) * stride
    starts = np.arange(0, max(last + 1, 0))
    gap = np.abs(starts - t0)
    return starts[(gap >= (T_WINDOW - 1) * stride + 2) & (gap <= max_gap)]


class HardNegativeSampler:
    """``robot`` maps episode -> (L, 4) robot states (``robot_state`` of every frame)."""

    def __init__(self, robot: dict[str, np.ndarray], max_gap: int = 40, min_hand_m: float = 0.01,
                 min_gripper: float = 0.1):
        if max_gap < 1 or min_hand_m < 0 or min_gripper < 0:
            raise ValueError("hard-negative thresholds must be non-negative and max_gap >= 1")
        self.robot = {ep: np.asarray(states, dtype=np.float64) for ep, states in robot.items()}
        if any(states.ndim != 2 or states.shape[1] != 4 for states in self.robot.values()):
            raise ValueError("robot states must be (L, 4): hand xyz and gripper opening")
        self.max_gap, self.min_hand_m, self.min_gripper = int(max_gap), float(min_hand_m), float(min_gripper)

    def candidates(self, episode: str, t0: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
        """(gap-eligible first frames, their validity under the robot-state rule)."""
        states = self.robot[episode]
        starts = candidate_starts(t0, stride, len(states), self.max_gap)
        last = (T_WINDOW - 1) * stride
        valid = robot_moved(states[t0 + last][None], states[starts + last], self.min_hand_m, self.min_gripper)
        return starts, np.asarray(valid, dtype=bool)

    def sample(self, episode: str, t0: int, stride: int, generator: torch.Generator | None = None) -> int | None:
        """A uniform draw among the valid candidates (torch RNG: the DataLoader worker's seeded generator), or None."""
        starts, valid = self.candidates(episode, t0, stride)
        chosen = starts[valid]
        if not len(chosen):
            return None
        return int(chosen[int(torch.randint(len(chosen), (1,), generator=generator))])

    def hand_displacement(self, episode: str, t0: int, t0_neg: int, stride: int) -> float:
        last = (T_WINDOW - 1) * stride
        states = self.robot[episode]
        return float(np.linalg.norm(states[t0 + last, :3] - states[t0_neg + last, :3]))

    def statistics(self, samples: list[tuple[str, int, int]]) -> dict:
        """Before training: over (episode, t0, stride) samples, the fraction of gap-eligible candidate pairs the robot
        rule keeps, the fraction of samples with at least one valid negative, and the hand displacement (m) among the
        kept pairs (all kept pairs, and weighted as the sampler draws them: 1 / valid count per sample)."""
        total = kept = with_valid = 0
        disp, weights = [], []
        for episode, t0, stride in samples:
            starts, valid = self.candidates(episode, t0, stride)
            total += len(starts)
            kept += int(valid.sum())
            if valid.any():
                with_valid += 1
                last = (T_WINDOW - 1) * stride
                states = self.robot[episode]
                d = np.linalg.norm(states[starts[valid] + last, :3] - states[t0 + last, :3], axis=-1)
                disp.append(d)
                weights.append(np.full(len(d), 1.0 / len(d)))
        disp = np.concatenate(disp) if disp else np.zeros(0)
        weights = np.concatenate(weights) if weights else np.zeros(0)
        qs = (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)

        def weighted_quantiles(values, w):
            if not len(values):
                return [None] * len(qs)
            order = np.argsort(values)
            cum = np.cumsum(w[order]) / w.sum()
            return [float(values[order][min(int(np.searchsorted(cum, q)), len(values) - 1)]) for q in qs]

        edges = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, float("inf")]
        hist = np.histogram(disp, bins=edges)[0].tolist() if len(disp) else [0] * (len(edges) - 1)
        return {
            "samples": len(samples), "candidate_pairs": total, "kept_pairs": kept,
            "kept_fraction": kept / total if total else None,
            "samples_with_valid_negative": with_valid,
            "valid_negative_fraction": with_valid / len(samples) if samples else None,
            "hand_displacement_m_quantiles": dict(zip(map(str, qs), np.quantile(disp, qs).tolist() if len(disp) else [None] * len(qs))),
            "hand_displacement_m_quantiles_as_sampled": dict(zip(map(str, qs), weighted_quantiles(disp, weights))),
            "hand_displacement_m_histogram": {"edges": edges[:-1] + ["inf"], "counts": hist},
            "rule": {"max_gap": self.max_gap, "min_hand_m": self.min_hand_m, "min_gripper": self.min_gripper},
        }
