"""Our method's real-world window loader over the compact training store (``s4d.realworld.train_store``).

Produces exactly the batch contract of the former window export (``RealworldCacheDataset``; track form of
``s4d.data.contract``): images (T,V,3,S,S) uint8, depth (T,V,1,S,S) metres, K (128-pixel grid, continuous), w2c/c2w,
motion_score, track_src/track_disp (P,V,N,3), track_weight, track_count, gripper_points (T,G,3), gripper_weight
(P,V,G), probe_state (T,7). Nothing window-level is stored: windows are enumerated at construction from the per-episode
sync table, robot-state flags, recording segments and the track chunk tables (``enumerate_windows``), and every target
is derived per item (``camera_targets``).

Window rule (the former export's, plus the recording-segment check): the canonical timeline is the reference camera
(the store's first camera); window ``(t, t+s, t+2s)`` for every stride ``s``; kept when the robot state is valid at the
three times, EVERY consecutive reference frame inside the window span is at most ``max_gap_factor`` median periods
after its predecessor (no window across a recording pause or a dropped reference frame, so the canonical times are
evenly spaced; stricter than the former export, which checked only the three sampled frames against
``max_gap_factor * s`` periods), the three reference frames lie in one recording segment, and enough cameras are
ELIGIBLE: synchronised at all three times (stored mutual nearest within
``max_sync_ns``; a smaller loader bound filters further), the three frames in one recording segment and in one
TrackEverything query window (the latest such window). All selected cameras must be eligible unless ``num_views`` is
set, which then needs ``num_views`` eligible cameras and draws them per (index, epoch).
``validation_tail_frames`` labels windows lying entirely in the last frames ``validation`` and drops straddling ones.

Targets per camera (moving ``track_motion`` tracks of the chosen query window, base-frame metres, never rescaled):
``track_targets`` (source visibility = valid tracker status), pairs across a motion-segment break weighted 0, and the
square crop handled as in the DROID square-crop profile: a pair is supervised only if the track projects inside the
stored image (z > 0) at BOTH endpoints; the motion score splats only tracks inside the image
(``export.splat_score_given_motion`` with segment-aware motion); the gripper FK points (reference-camera time) are
depth-tested (2 cm) against each camera's NATIVE-resolution depth of its synchronised frame (stored per sync set by the
writer: the export's test, exactly) and a gripper pair is supervised only if inside the stored image at both endpoints.
"""

from __future__ import annotations

import hashlib

import numpy as np
import torch
from torch.utils.data import Dataset

from s4d.data.contract import PAIRS, validate_batch
from s4d.data.droid.largescale.crop import pair_membership
from s4d.data.droid.pointworld import project, track_targets
from s4d.realworld.export import segment_motion, splat_score_given_motion
from s4d.realworld.geometry import invert_T
from s4d.realworld.tracking import VALID
from s4d.realworld.train_store import CompactStore


def stable_hash(*parts) -> int:
    return int.from_bytes(hashlib.blake2b("|".join(map(str, parts)).encode(), digest_size=8).digest(), "little")


def inside_image(points: np.ndarray, K: np.ndarray, w2c: np.ndarray, size: int) -> np.ndarray:
    """(N,3) base-frame points -> (N,) bool: in front of the camera and projecting into the stored S x S image."""
    uv, z = project(points, K, w2c)
    finite = np.isfinite(uv).all(-1) & np.isfinite(z)
    pix = np.floor(np.where(finite[:, None], uv, -1.0))
    return finite & (z > 0) & (pix >= 0).all(-1) & (pix < size).all(-1)


def chunk_for(chunk_frames: np.ndarray, chunk_query: np.ndarray, frames) -> int:
    """Index of the latest query window containing all ``frames`` (-1 when none)."""
    hit = np.all([(chunk_frames == f).any(1) for f in frames], 0)
    if not hit.any():
        return -1
    cand = np.flatnonzero(hit)
    return int(cand[np.argmax(chunk_query[cand])])


def enumerate_windows(stamps, ref_segment, robot_ok, frames, matched, segments, chunk_tables, *, strides,
                      max_gap_factor, min_views) -> tuple[list[list[int]], dict]:
    """Windows of one episode: rows ``[t0, s, chunk_0, ..., chunk_{V-1}]`` (chunk -1: camera not eligible).

    ``stamps``/``ref_segment``/``robot_ok`` (n,) on the reference timeline; ``frames`` (n,V) synchronised frame of each
    selected camera (-1 none), ``matched`` (n,V) within the loader's sync bound; ``segments[v]`` (F_v,) recording segment
    of camera v's frames; ``chunk_tables[v]`` = (chunk_frames (C,L), chunk_query (C,)). A window needs ``min_views``
    eligible cameras (module docstring)."""
    stamps = np.asarray(stamps, np.int64)
    n, V = frames.shape
    period = float(np.median(np.diff(stamps))) if n > 1 else 0.0
    counts = {"windows_dropped_for_time_gaps": 0, "dropped_recording_segment": 0, "dropped_too_few_views": 0}
    rows = []
    for s in strides:
        for t0 in range(n - 2 * s):
            canon = [t0, t0 + s, t0 + 2 * s]
            if not robot_ok[canon].all():
                continue
            if max_gap_factor is not None and np.diff(stamps[t0:t0 + 2 * s + 1]).max() > max_gap_factor * period:
                counts["windows_dropped_for_time_gaps"] += 1
                continue
            if len(set(np.asarray(ref_segment)[canon].tolist())) != 1:
                counts["dropped_recording_segment"] += 1
                continue
            chunks = []
            for v in range(V):
                fr = frames[canon, v]
                ok = bool(matched[canon, v].all()) and fr[0] < fr[1] < fr[2]
                ok = ok and len(set(np.asarray(segments[v])[fr].tolist())) == 1
                chunks.append(chunk_for(*chunk_tables[v], fr) if ok else -1)
            if sum(k >= 0 for k in chunks) < min_views:
                counts["dropped_too_few_views"] += 1
                continue
            rows.append([t0, s, *chunks])
    return rows, counts


def camera_targets(pts, valid, seg, grip, gvis, K, w2c, depth, size: int) -> dict:
    """Targets of one camera for one window. ``pts`` (3,J,3) base-frame track positions at the window's times,
    ``valid`` (3,J) valid tracker status, ``seg`` (3,J) motion segments (-1 invalid), ``grip`` (3,G,3) gripper FK points,
    ``gvis`` (3,G) their native-depth visibility in this camera, ``K``/``w2c`` of the stored image, ``depth`` (3,S,S).
    Returns motion_score (3,1,S,S), track_src/track_disp (P,J,3), track_weight (P,J), gripper_weight (P,G)."""
    inside = np.stack([inside_image(pts[t], K, w2c, size) for t in range(3)])
    g_inside = np.stack([inside_image(grip[t], K, w2c, size) for t in range(3)])
    gmotion = segment_motion(grip, np.zeros(grip.shape[:2], np.int16))
    score = splat_score_given_motion(np.concatenate([pts, grip], 1), np.concatenate([valid & inside, gvis], 1),
                                     np.concatenate([segment_motion(pts, seg), gmotion]), K, w2c, depth)
    src, disp, weight = track_targets(pts, valid)
    for pair, (a, b, _) in enumerate(PAIRS):
        cross = (seg[a] < 0) | (seg[a] != seg[b])  # never across a motion-segment break
        src[pair, cross], disp[pair, cross], weight[pair, cross] = 0, 0, 0
    weight *= pair_membership(inside, PAIRS)  # crop: inside the stored image at both endpoints
    gweight = (gvis[[a for a, _, _ in PAIRS]] & pair_membership(g_inside, PAIRS)).astype(np.float32)
    return {"motion_score": score, "track_src": src, "track_disp": disp, "track_weight": weight,
            "gripper_weight": gweight}


class RealworldWindowDataset(Dataset):
    def __init__(self, store, *, episodes=None, split: str = "train", strides=(1, 2), max_sync_ns: int | None = None,
                 max_gap_factor: float | None = 1.5, validation_tail_frames: int = 0, cameras=None,
                 num_views: int | None = None, max_tracks: int | None = None, seed: int = 0, validate: bool = True):
        if split not in ("train", "validation", "all"):
            raise ValueError("split must be train, validation or all")
        self.store = store if isinstance(store, CompactStore) else CompactStore(store)
        self.episodes = list(episodes or self.store.episodes())
        self.strides = tuple(int(s) for s in strides)
        self.seed, self.validate = int(seed), bool(validate)
        self.num_views = None if num_views is None else int(num_views)
        if self.num_views is not None and self.num_views < 1:
            raise ValueError("num_views must be >= 1")
        self.epoch_tagged_indices = self.num_views is not None  # a new viewpoint draw per visit (trainer sampler)
        rows, self.cams = [], []
        self.counts = {"windows_dropped_for_time_gaps": 0, "dropped_recording_segment": 0, "dropped_too_few_views": 0}
        capacity = 1
        for e, ep in enumerate(self.episodes):
            ef = self.store.episode(ep)
            cams = list(cameras or ef.cameras)
            if self.num_views is None and self.cams and len(cams) != len(self.cams[0]):
                raise ValueError("every episode must provide the same number of cameras (or set num_views)")
            if self.num_views is not None and self.num_views > len(cams):
                raise ValueError(f"{ep}: num_views {self.num_views} exceeds its {len(cams)} cameras")
            self.cams.append(cams)
            vidx = [ef.cameras.index(c) for c in cams]
            tol = ef.max_sync_ns if max_sync_ns is None else int(max_sync_ns)
            if tol > ef.max_sync_ns:
                raise ValueError(f"{ep}: max_sync_ns {tol} exceeds the store's bound {ef.max_sync_ns}")
            sync = ef.sync()
            frames, err = sync["frames"][:, vidx], sync["error_ns"][:, vidx]
            ref = ef.reference_camera
            robot = ef.robot(ref)
            tables = [ef.track_index(c) for c in cams]
            for t in tables:
                capacity = max(capacity, int(np.diff(t["chunk_offset"]).max(initial=0)))
            windows, counts = enumerate_windows(
                ef.timestamps(ref), ef.segments(ref), robot["ok"] & (robot["sync_err_ns"] <= tol), frames,
                (frames >= 0) & (err <= tol), [ef.segments(c) for c in cams],
                [(t["chunk_frames"], t["chunk_query"]) for t in tables], strides=self.strides,
                max_gap_factor=max_gap_factor, min_views=self.num_views or len(cams))
            for k, v in counts.items():
                self.counts[k] += v
            n = len(ef.timestamps(ref))
            boundary = n - int(validation_tail_frames) if validation_tail_frames else n
            for t0, s, *chunks in windows:
                if t0 + 2 * s < boundary or t0 >= boundary:  # straddling windows are dropped (no shared frame)
                    rows.append((e, t0, s, chunks, 1 if t0 >= boundary else 0))
        width = max((len(c) for c in self.cams), default=0)
        table = np.full((len(rows), 4 + width), -1, np.int64)
        for i, (e, t0, s, chunks, val) in enumerate(rows):
            table[i, :3] = (e, t0, s)
            table[i, 3:3 + len(chunks)] = chunks
            table[i, -1] = val
        if split != "all":
            table = table[table[:, -1] == (1 if split == "validation" else 0)]
        self.table = table
        self.capacity = capacity if max_tracks is None else int(max_tracks)
        self.store.close()  # no open handle / cached array is inherited by DataLoader workers
        if not len(self.table):
            raise ValueError(f"empty split {split}")

    def __len__(self) -> int:
        return len(self.table)

    def window(self, index: int) -> dict:
        e, t0, s, *rest = (int(x) for x in self.table[index])
        chunks = rest[:-1][:len(self.cams[e])]
        return {"episode": self.episodes[e], "canonical_indices": [t0, t0 + s, t0 + 2 * s], "stride": s,
                "chunks": chunks, "split": "validation" if rest[-1] else "train"}

    def _select(self, ntracks: int, *key) -> np.ndarray:
        if ntracks <= self.capacity:
            return np.arange(ntracks)
        rng = np.random.default_rng(stable_hash(self.seed, *key))
        return np.sort(rng.choice(ntracks, self.capacity, replace=False))

    def __getitem__(self, index) -> dict:
        index, epoch = (int(index[0]), int(index[1])) if isinstance(index, tuple) else (int(index), 0)
        w = self.window(index)
        ep, s, chunks = w["episode"], w["stride"], w["chunks"]
        e = int(self.table[index, 0])
        ef = self.store.episode(ep)
        cams = self.cams[e]
        views = [v for v, k in enumerate(chunks) if k >= 0]
        if self.num_views is not None:
            rng = np.random.default_rng(stable_hash(self.seed, "views", ep, w["canonical_indices"][0], s, epoch))
            views = sorted(rng.choice(views, self.num_views, replace=False).tolist())
        canon = np.array(w["canonical_indices"])
        ref_robot = ef.robot(ef.reference_camera)
        grip = ref_robot["gripper_points"][canon].astype(np.float32)  # (T,G,3)
        probe = ref_robot["probe_state"][canon].astype(np.float32)
        sync = ef.sync()
        frames_all, gvis_all = sync["frames"], sync["gripper_visible"]
        S, N = ef.image_size, self.capacity
        out = {k: [] for k in ("images", "depth", "K", "w2c", "motion_score", "track_src", "track_disp",
                               "track_weight", "track_count", "gripper_weight")}
        frame_ids = []
        for v in views:
            cam = cams[v]
            col = ef.cameras.index(cam)
            fr = frames_all[canon, col].astype(np.int64)
            frame_ids.append(tuple(int(x) for x in fr))
            depth = ef.depth(cam, fr)
            K = ef.intrinsics(cam).astype(np.float32)
            w2c = ef.w2c(cam).astype(np.float32)
            tr = ef.tracks(cam)
            c = chunks[v]
            lo, hi = int(tr["chunk_offset"][c]), int(tr["chunk_offset"][c + 1])
            pos = [int(np.flatnonzero(tr["chunk_frames"][c] == f)[0]) for f in fr]
            keep = lo + self._select(hi - lo, ep, cam, c)
            pts = tr["xyz"][keep][:, pos].transpose(1, 0, 2).astype(np.float32)  # (T,J,3)
            valid = tr["status"][keep][:, pos].T == VALID
            seg = tr["segment"][keep][:, pos].T.astype(np.int64)
            tg = camera_targets(pts, valid, seg, grip, gvis_all[canon, col], K, w2c, depth, S)
            pad = ((0, 0), (0, N - pts.shape[1]))
            out["track_src"].append(np.pad(tg["track_src"], (*pad, (0, 0))))
            out["track_disp"].append(np.pad(tg["track_disp"], (*pad, (0, 0))))
            out["track_weight"].append(np.pad(tg["track_weight"], pad))
            out["track_count"].append(pts.shape[1])
            out["gripper_weight"].append(tg["gripper_weight"])
            out["motion_score"].append(tg["motion_score"])
            out["images"].append(ef.rgb(cam, fr).transpose(0, 3, 1, 2))
            out["depth"].append(depth[:, None])
            out["K"].append(K)
            out["w2c"].append(w2c)
        sample = {
            "images": torch.from_numpy(np.ascontiguousarray(np.stack(out["images"], 1))),
            "depth": torch.from_numpy(np.stack(out["depth"], 1).astype(np.float32)),
            "K": torch.from_numpy(np.stack(out["K"])),
            "w2c": torch.from_numpy(np.stack(out["w2c"])),
            "c2w": torch.from_numpy(np.stack([invert_T(T) for T in out["w2c"]]).astype(np.float32)),
            "motion_score": torch.from_numpy(np.stack(out["motion_score"], 1)),
            "track_src": torch.from_numpy(np.stack(out["track_src"], 1)),
            "track_disp": torch.from_numpy(np.stack(out["track_disp"], 1)),
            "track_weight": torch.from_numpy(np.stack(out["track_weight"], 1)),
            "track_count": torch.tensor(out["track_count"], dtype=torch.int64),
            "gripper_points": torch.from_numpy(grip),
            "gripper_weight": torch.from_numpy(np.stack(out["gripper_weight"], 1)),
            "probe_state": torch.from_numpy(probe),
        }
        sample["meta"] = {"task": "realworld", "task_name": ef.task, "episode": ef.episode_id,
                          "t_indices": tuple(int(x) for x in canon), "stride": s,
                          "camera_ids": tuple(cams[v] for v in views), "frame_indices": tuple(frame_ids)}
        if self.validate:
            validate_batch({k: v if k == "meta" else v.unsqueeze(0) for k, v in sample.items()})
        return sample
