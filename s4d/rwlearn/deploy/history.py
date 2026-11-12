"""Bounded, in-memory camera history (science side, no I/O). It assembles exactly the trained observation history:
``history_length`` consecutive camera frames per camera at the trained spacing, oldest first.
Missing frames never compress the spacing: a gap ends the run, and a history is either complete inside one run or,
only with ``pad_run_start=True``, padded with the first frame of the run exactly as training does
(``s4d.rwlearn.store.history``: frame-stack reset at a recording run start / episode start). No implicit padding,
timestamp substitution or dropped-frame compression.

Multi-view cross-camera synchronisation (user directive 2026-11-12, evening) is a WARNING, never a refusal: a window
whose views are not mutual-nearest frames (an offset of at least half a frame period between two cameras of a step,
the training ``mutual_sync`` bound) or whose declared M5 robot-surface motion exceeds the trained 5 mm is served as
assembled; ``ObservationWindow.sync_report`` measures it (per-camera capture times, pairwise offsets, the per-step
spread, the M5 evidence when declared) so callers count and log it. Every per-camera check is unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from s4d.rwlearn.deploy.contracts import InputSpec
from s4d.rwlearn.deploy.errors import (
    ContractViolation,
    IncompleteHistory,
    IrregularHistory,
    SessionMismatch,
    Unsynchronized,
)
from s4d.rwlearn.store import history as training_history


@dataclass(frozen=True)
class Frame:
    seq: int  # CAMERA acquisition sequence (RealSense frame number), not a request counter; drops leave gaps
    timestamp_ns: int  # common host monotonic clock (the recorder's convention); never arrival time
    rgb: np.ndarray
    depth: np.ndarray | None = None
    valid: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ObservationWindow:
    spec: InputSpec
    frames: dict[str, tuple[Frame, ...]]
    # optional M5 evidence (informational): one robot-surface spread per assembled step, None where not declared,
    # inf = no valid robot state
    surface_disp_m: tuple[float | None, ...] | None = None
    padded: bool = False  # run-start padding (training rule) was applied

    @property
    def frame_ids(self) -> dict:
        return {c: [int(f.seq) for f in self.frames[c]] for c in self.spec.camera_ids}

    @property
    def timestamps_ns(self) -> np.ndarray:
        return np.array([[f.timestamp_ns for f in self.frames[c]] for c in self.spec.camera_ids], np.int64).T

    @property
    def observation_time_ns(self) -> int:
        return int(self.timestamps_ns[-1].max())  # training multi-view causality: latest constituent frame

    def binding(self, reference_id: str | int, *, pose_time_ns: int, gripper_time_ns: int) -> dict:
        """IDs to accompany the measured flange pose / gripper register (and their robot-clock sample times on the
        same host monotonic clock) at ``observation_time_ns``. Echoed unchanged by ``DeployPolicy.infer``."""
        return {"reference_id": reference_id, "frame_ids": self.frame_ids,
                "observation_time_ns": self.observation_time_ns, "session_id": self.spec.session_id,
                "pose_time_ns": pose_time_ns, "gripper_time_ns": gripper_time_ns}

    def validate(self) -> None:
        s = self.spec
        if not s.session_id or not s.camera_ids or set(self.frames) != set(s.camera_ids):
            raise SessionMismatch("window cameras must exactly match a bound session")
        pads = set()
        for c in s.camera_ids:
            fs = self.frames[c]
            if len(fs) != s.history_length:
                raise IncompleteHistory("incomplete history (pad only at a run start with pad_run_start=True)")
            ids = np.array([f.seq for f in fs])
            p = int(np.sum(ids[1:] == ids[0]))  # leading duplicates of the run's first frame
            if p and not self.padded:
                raise ContractViolation("duplicate frames in an unpadded history")
            if any(fs[j] is not fs[0] for j in range(1, p + 1)):
                raise ContractViolation("padding must repeat the run's first frame")
            pads.add(p)
            distinct = fs[p:]
            if not np.all(np.diff([f.seq for f in distinct]) == s.frame_spacing):
                raise IrregularHistory("missing camera frames must not change trained spacing")
            dt = np.diff([f.timestamp_ns for f in distinct]) / 1e9
            if not np.all(np.abs(dt - s.history_spacing_s) <= s.history_spacing_s * s.timestamp_tolerance_fraction):
                raise IrregularHistory("irregular timestamp history at trained frame spacing")
            for f in distinct:
                validate_frame(f, s)
                if "depth" in s.modalities and f.metadata["native_correction"] != s.native_depth_correction[c]:
                    raise ContractViolation("frame native correction belongs to another camera")
        if self.padded and pads == {0}:
            raise ContractViolation("window marked padded without padding")
        if len(pads) != 1:
            raise Unsynchronized("multi-view padding must be step-aligned across cameras")
        if s.views > 1:
            # Cross-camera offsets and M5 motion are measured by sync_report (warnings), never refused here; only
            # malformed evidence is.
            if self.surface_disp_m is not None:
                if len(self.surface_disp_m) != s.history_length:
                    raise ContractViolation("M5 evidence must give one value (or None) per assembled step")
                for d in self.surface_disp_m:
                    if d is not None and (isinstance(d, bool) or not isinstance(d, (int, float, np.floating))
                                          or np.isnan(d) or d < 0):
                        raise ContractViolation("M5 evidence values are nonnegative metres, inf (no robot state) "
                                                "or None (not declared)")
        elif self.surface_disp_m is not None:
            raise ContractViolation("single-view windows take no M5 evidence")

    def sync_report(self) -> dict | None:
        """Cross-camera synchronisation of a multi-view window (None for single-view), measured, never enforced:
        ``capture_ns`` (end step, per camera), ``age_rel_ms`` (each end frame's age relative to the observation time
        = the newest end frame), ``offsets_ms`` (end step, pairwise ``a|b`` = t_a - t_b, cameras in session order),
        ``step_spread_ms`` (max - min capture time of every step, oldest first), ``max_offset_ms`` (largest offset of
        any view from the reference = first camera, over all steps), ``bound_ms`` (the training mutual-nearest bound,
        half a frame period), ``mismatch`` (``max_offset_ms`` >= the bound: the former strict refusal),
        ``surface_disp_m`` / ``motion_over_bound`` (declared M5 evidence per step: metres, ``"no_state"`` = no valid
        robot state, None = not declared) and ``warnings``: the reasons the former strict contract would have refused
        this window. JSON-safe (finite numbers only)."""
        s = self.spec
        if s.views <= 1:
            return None
        cams = list(s.camera_ids)
        t = self.timestamps_ns
        end = t[-1]
        obs = int(end.max())
        spread = (t.max(1) - t.min(1)) / 1e6
        ref_offset = float(np.abs(t - t[:, :1]).max()) / 1e6  # the former refusal: |t_view - t_reference| per step
        bound = s.frame_period_s * 1e3 / 2
        disp = None if self.surface_disp_m is None else [None if d is None else float(d) for d in self.surface_disp_m]
        limit = float(s.max_set_motion_m) if s.max_set_motion_m is not None else np.inf
        over = disp is not None and any(d is not None and d > limit for d in disp)
        mismatch = ref_offset >= bound
        warnings = []
        if mismatch:
            warnings.append(f"views not mutual-nearest: offset from the reference camera up to {ref_offset:.1f} ms "
                            f">= {bound:.1f} ms (half a frame period)")
        if over:
            worst = max(d for d in disp if d is not None)
            warnings.append("M5: robot surface moved " + ("(no valid robot state)" if not np.isfinite(worst)
                                                           else f"{worst * 1e3:.1f} mm") +
                            f" > {limit * 1e3:.0f} mm within a step")
        if disp is not None:
            disp = [d if d is None or np.isfinite(d) else "no_state" for d in disp]
        return {"cameras": cams, "frame_ids": {c: int(self.frames[c][-1].seq) for c in cams},
                "capture_ns": {c: int(end[i]) for i, c in enumerate(cams)},
                "age_rel_ms": {c: (obs - int(end[i])) / 1e6 for i, c in enumerate(cams)},
                "offsets_ms": {f"{a}|{b}": (int(end[i]) - int(end[j])) / 1e6
                               for i, a in enumerate(cams) for j, b in enumerate(cams) if i < j},
                "step_spread_ms": [float(x) for x in spread], "max_offset_ms": ref_offset,
                "bound_ms": float(bound), "mismatch": mismatch, "surface_disp_m": disp,
                "motion_over_bound": bool(over), "warnings": warnings}

    def observation(self) -> dict:
        """The ``OfflinePolicy.act`` observation: rgb (T, V, S, S, 3) uint8 [+ depth (T, V, S, S) float32 m]
        [+ K (V, 3, 3), c2w (V, 4, 4) float64 from the session calibration]."""
        self.validate()
        s = self.spec
        out = {"rgb": np.stack([np.stack([f.rgb for f in self.frames[c]]) for c in s.camera_ids], 1)}
        if "depth" in s.modalities:
            out["depth"] = np.stack([np.stack([f.depth for f in self.frames[c]]) for c in s.camera_ids], 1)
        if s.needs_calibration:
            out["K"] = np.array([s.calibration[c]["K"] for c in s.camera_ids], np.float64)
            out["c2w"] = np.array([s.calibration[c]["c2w"] for c in s.camera_ids], np.float64)
        return out


def validate_frame(f: Frame, spec: InputSpec):
    if type(f.seq) is not int or f.seq < 0 or type(f.timestamp_ns) is not int or f.timestamp_ns < 0:
        raise ContractViolation("frame seq and common monotonic timestamp_ns must be nonnegative integers")
    rgb = np.asarray(f.rgb)
    if rgb.shape != (*spec.image_hw, 3) or rgb.dtype != np.uint8:
        raise ContractViolation("frame rgb must match the preprocessed training grid and uint8 dtype")
    if "depth" in spec.modalities:
        if f.depth is None or f.valid is None:
            raise ContractViolation("required depth and validity missing")
        d, valid = np.asarray(f.depth), np.asarray(f.valid)
        if d.shape != spec.image_hw or d.dtype != np.float32 or valid.shape != d.shape or valid.dtype != bool:
            raise ContractViolation("depth float32 metres and bool validity must match the training grid")
        if not np.isfinite(d).all() or (d < 0).any() or not np.array_equal(valid, d > 0):
            raise ContractViolation("depth invalid semantics violated (0 invalid, positive finite metres valid)")
        if f.metadata.get("depth_source") != spec.deployment_depth:
            raise ContractViolation("deployment depth must be explicitly labelled native_realsense")
        # Every frame must certify the same sensor correction selected at session startup.
        if f.metadata.get("native_correction") not in spec.native_depth_correction.values():
            raise ContractViolation("frame native correction differs from the session contract")
    elif f.depth is not None or f.valid is not None:
        raise ContractViolation("unexpected depth would be silently ignored by this RGB-only condition")


class HistoryWindow:
    """Per-session frame cache: ``push`` validated frames per camera, ``assemble`` the trained history ending at given
    camera frames, ``reset`` explicitly (a new run). A frame whose seq does not follow the previous one by the trained
    spacing, or whose timestamp gap exceeds ``spec.run_gap_factor`` periods, starts a new run (``store.runs_from``)."""

    def __init__(self, spec: InputSpec, *, capacity: int | None = None):
        if not spec.session_id:
            raise SessionMismatch("bind InputSpec to a session first")
        self.spec = spec
        span = (spec.history_length - 1) * spec.frame_spacing + 1
        self.capacity = int(capacity if capacity is not None else 2 * span + 2)
        if self.capacity < span:
            raise ContractViolation("history capacity shorter than trained history span")
        self.reset()

    def reset(self) -> None:
        self._frames = {c: {} for c in self.spec.camera_ids}
        self._run = {c: {} for c in self.spec.camera_ids}  # seq -> seq of its run's first frame

    def push(self, camera_id: str, frame: Frame) -> None:
        if camera_id not in self._frames:
            raise ContractViolation("camera is not in this session")
        validate_frame(frame, self.spec)
        if "depth" in self.spec.modalities and \
                frame.metadata["native_correction"] != self.spec.native_depth_correction[camera_id]:
            raise ContractViolation("native correction belongs to another camera")
        cache = self._frames[camera_id]
        run_start = frame.seq
        if cache:
            last = cache[max(cache)]
            if frame.seq <= last.seq or frame.timestamp_ns <= last.timestamp_ns:
                raise ContractViolation("duplicate or reordered camera frame; reset explicitly for a new sequence")
            gap_s = (frame.timestamp_ns - last.timestamp_ns) / 1e9
            if frame.seq - last.seq == self.spec.frame_spacing and \
                    gap_s <= self.spec.run_gap_factor * self.spec.history_spacing_s:
                run_start = self._run[camera_id][last.seq]
        cache[frame.seq] = Frame(frame.seq, frame.timestamp_ns, np.array(frame.rgb, copy=True),
                                 None if frame.depth is None else np.array(frame.depth, copy=True),
                                 None if frame.valid is None else np.array(frame.valid, copy=True),
                                 dict(frame.metadata))
        self._run[camera_id][frame.seq] = run_start
        while len(cache) > self.capacity:
            oldest = min(cache)
            del cache[oldest], self._run[camera_id][oldest]

    def assemble(self, end_ids: dict[str, int], *, surface_disp_m=None, pad_run_start: bool = False
                 ) -> ObservationWindow:
        """History ending at camera frames ``end_ids`` (multi-view: each camera's own end frame, e.g. its latest valid
        frame; cross-camera offsets are measured by ``sync_report``, not refused). ``surface_disp_m`` (optional
        multi-view M5 evidence, informational): a sequence with one value per assembled step (oldest first) or a dict
        {reference-camera seq: displacement} (steps without an entry: None). ``pad_run_start``: apply the training
        run-start padding when fewer than ``history_length`` frames (steps) of the current run exist; otherwise such a
        history raises."""
        s = self.spec
        if set(end_ids) != set(s.camera_ids):
            raise ContractViolation("end_ids must identify every session camera exactly")
        available = {}
        for c in s.camera_ids:
            end = end_ids[c]
            if type(end) is not int or end not in self._frames[c]:
                raise IncompleteHistory("end_ids must be cached camera acquisition integers (missing camera frame)")
            available[c] = (end - self._run[c][end]) // s.frame_spacing + 1  # frames of the run up to end
        steps = min(available.values())  # a synchronised run starts once every camera is inside its run
        if steps < s.history_length and not pad_run_start:
            raise IncompleteHistory("incomplete history or missing camera frame at trained spacing")
        frames = {}
        for c in s.camera_ids:
            end = end_ids[c]
            run = np.zeros(s.history_length, np.int64)  # index space: 0 .. L-1 ending at end
            first = s.history_length - min(steps, s.history_length)
            run[:first] = -1  # before the (synchronised) run start
            idx = training_history(run, s.history_length - 1, s.history_length)
            wanted = [end - (s.history_length - 1 - i) * s.frame_spacing for i in idx]
            if any(k not in self._frames[c] for k in wanted):
                raise IncompleteHistory("incomplete history or missing camera frame at trained spacing")
            frames[c] = tuple(self._frames[c][k] for k in wanted)
        padded = steps < s.history_length
        if isinstance(surface_disp_m, dict):
            ref = s.camera_ids[0]
            surface_disp_m = [surface_disp_m.get(f.seq) for f in frames[ref]]
            if all(d is None for d in surface_disp_m):
                surface_disp_m = None
        window = ObservationWindow(s, frames, None if surface_disp_m is None else tuple(surface_disp_m), padded)
        window.validate()
        return window
