"""Offline-RL transitions of the real-world recordings (definitions and evidence: docs/realworld_fql.md, "Rewards,
terminals and transitions").

A TIMELINE is the sequence of observation times of one viewpoint condition: a camera's own native frames (single-view
encoders; the viewpoint is fixed for the whole timeline, so an observation history and its next observation always
come from that camera) or the synchronised multi-camera steps of ``store.sync_index`` (multi-view encoders, reference
camera times). Per timeline:

* observation k = frame(s) at tau_k + proprioception at tau_k (``actions.step_table``);
* transition k -> k+1 exists iff ``step_table`` marks it valid (same recording segment, no dropped frame, fresh state,
  only sent+engaged commands in the step); pauses between recording segments are never crossed;
* observation histories (3 frames, consecutive policy steps) are padded with the first frame of the run, as at an
  episode start;
* rewards, masks (``REWARDS``): default ``"ogbench_final_success"`` - for an episode whose outcome is SUCCESS, the
  success state is the recording's FINAL observation (last frame of the last segment), and only if that frame is
  reached by a valid transition (an unusable tail never moves the label earlier: the timeline then has no success
  and is reported); r_k = success(s_{k+1}) - 1 and mask_k = 1 - success(s_{k+1}) (FQL's single-task OGBench
  convention: -1 per step, no bootstrap from success). A FAILED episode has no success state (its last transition is
  a truncation, mask 1). ``"sparse_final"``: r_k = success(s_{k+1}), same masks. The files carry no outcome marker,
  so outcomes must be given explicitly (per-episode labels, or the explicit assumption that all recordings are
  successful demonstrations; open decision).
* multi-view observation time = the LATEST constituent frame of the synchronised step (causal: no camera image is
  newer than the proprioception and the start of the action interval).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from s4d.rwlearn import actions as A
from s4d.rwlearn.store import FrameStore, history, runs_from, sync_index

REWARDS = ("ogbench_final_success", "sparse_final")
MULTIVIEW = "sync"


@dataclass
class Timeline:
    name: str  # camera serial, or "sync"
    times: np.ndarray  # (K,) ns
    segment: np.ndarray  # (K,)
    run: np.ndarray  # (K,)
    frames: np.ndarray  # (K, V) native frame index per camera used (V = 1 for a camera timeline)
    cameras: list


def timelines(store: FrameStore, ep: str, multiview: bool, max_sync_ns: int = 30_000_000) -> list[Timeline]:
    if multiview:
        s = sync_index(store, ep, max_sync_ns)
        # runs from the reference times (pauses, dropped steps); step_table re-checks step lengths on these times
        return [Timeline(MULTIVIEW, s.latest_times, s.segment, s.run, s.frames, s.cameras)]
    out = []
    for cam in store.cameras(ep):
        t, seg = store.timestamps(ep, cam), store.segments(ep, cam)
        out.append(Timeline(cam, t, seg, runs_from(t, seg), np.arange(len(t))[:, None], [cam]))
    return out


def episode_transitions(raw: A.RawEpisode, tl: Timeline, contract: A.ActionContract,
                        reward: str = "ogbench_final_success", *, success: bool = True) -> dict:
    """Arrays over the K observations of one timeline (``transition_ok[k]``: k -> k+1 is a transition)."""
    if reward not in REWARDS:
        raise ValueError(f"reward {reward!r} not in {REWARDS}")
    tab = A.step_table(raw, tl.times, tl.segment, nominal_period_s=contract.nominal_period_s,
                       max_step_factor=contract.max_step_s / contract.nominal_period_s)
    K = len(tl.times)
    same_run = np.zeros(K, bool)
    same_run[:-1] = tl.run[:-1] == tl.run[1:]
    ok = tab.transition_ok & same_run & np.r_[tab.state_ok[1:], False]  # the next observation needs a state too
    reachable = np.zeros(K, bool)
    reachable[1:] = ok[:-1]
    outcome = bool(success)
    success = np.zeros(K, bool)
    # the final observation must be the RECORDING's end: every constituent camera's last recorded frame (a store or
    # synchronisation that dropped tail frames leaves the success unlabelled instead of moving it earlier)
    at_end = bool(K) and all(int(tl.frames[K - 1, v]) == len(raw.cameras[cam]["t"]) - 1
                             for v, cam in enumerate(tl.cameras))
    final_reached = bool(K and reachable[K - 1] and at_end)
    if outcome and final_reached:
        success[K - 1] = True  # the recording's final observation
    nxt = np.r_[success[1:], False]
    rew = (nxt.astype(np.float32) - 1.0) if reward == "ogbench_final_success" else nxt.astype(np.float32)
    action = np.zeros((K, A.ACTION_DIM), np.float32)
    action[ok] = contract.normalise(tab.action_phys[ok]).astype(np.float32)
    if ok.any() and np.abs(action[ok]).max() > 1.0:
        raise ValueError(f"{raw.path}/{tl.name}: a recorded command lies outside the contract bounds "
                         f"(max |a| {np.abs(action[ok]).max():.3f}); the teleop limits differ from the contract")
    terminal = ok & ~np.r_[ok[1:], False]  # last transition of a contiguous chain (frame-stack boundary)
    return {"times": tl.times, "run": tl.run, "proprio": np.nan_to_num(tab.proprio).astype(np.float32),
            "state_ok": tab.state_ok, "transition_ok": ok, "action": action,
            "action_phys": np.nan_to_num(tab.action_phys).astype(np.float32),
            "measured_pose": tab.measured_pose.astype(np.float64), "command_pose": tab.command_pose,
            "reward": np.where(ok, rew, 0).astype(np.float32), "mask": np.where(ok, 1.0 - nxt, 0).astype(np.float32),
            "success": success, "terminal": terminal, "reasons": tab.reasons,
            "label": ("success" if success.any() else "failure" if not outcome else "success_frame_unreachable")}


def history_indices(run: np.ndarray, ks: np.ndarray, length: int = 3) -> np.ndarray:
    """(len(ks), length) frame indices of the observation histories ending at ``ks`` (oldest first)."""
    return np.array([history(run, int(k), length) for k in ks], np.int64).reshape(len(ks), length)
