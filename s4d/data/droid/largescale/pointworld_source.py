"""Lazy, schema-checked access to released PointWorld-DROID per-episode files (never modified).

Released member paths (nvidia/PointWorld-DROID @ dd9aaeec): ``droid/flows-fs-optimized/<episode>_flows.h5``,
``droid/depth_320x180/<episode>_depth.h5`` and ``droid/cameras/<episode>_cameras.json``; they ship inside
``tar.zst`` packages, so a large-scale run reads them from an extracted tree (``layout`` below; one extraction of the
PointWorld annotations, no copy of DROID itself). Schema facts used here are in docs/POINTWORLD_SCHEMA.md and
docs/droid_largescale.md:

* flow root attrs: ``uuid``, ``scene_path``, ``hdf5_path``, ``canonical_timestamps`` (JSON list, int ms),
  ``ext1_cam_serial``, ``ext2_cam_serial``; clip groups ``"s:e"`` (half-open on the canonical timeline) with
  ``camera_<serial>_ext`` groups only (no wrist tracks) holding ``intrinsic`` (3x3, 320x180, integer centres),
  ``extrinsic`` (4x4 **w2c**), ``scene_flows`` (T,N,3) float16 world **positions**, ``scene_visibility``,
  ``scene_depth_valid_mask``; per-clip ``gripper_pose`` (T,7 xyz+qxyzw), ``gripper_positions`` (finger angle).
* depth: groups ``<serial>+ext`` and ``<serial>+wrist`` with ``depth`` (F,180,320) uint16 mm (0 invalid) and
  ``timestamps`` (F,) int64 ms.
* cameras JSON: ``optimization_success`` and ``<serial>: {"optimized_extrinsics": 4x4}`` for external serials.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from s4d.data.droid.largescale.handles import HandlePool

DEFAULT_LAYOUT = {
    "flows": "flows-fs-optimized/{episode}_flows.h5",
    "depth": "depth_320x180/{episode}_depth.h5",
    "cameras": "cameras/{episode}_cameras.json",
}
NATIVE_WH = (320, 180)


def clip_bounds(key: str) -> tuple[int, int]:
    start, end = map(int, key.split(":"))
    if start < 0 or end <= start:
        raise ValueError(f"invalid half-open clip {key!r}")
    return start, end


def sorted_clips(keys) -> list[str]:
    """HDF5 iterates lexicographically ("15:26" < "5:16"); clips are ordered by their integer bounds."""
    return sorted((k for k in keys if ":" in k), key=clip_bounds)


def parse_depth_group(name: str) -> tuple[str, str]:
    serial, sep, role = name.partition("+")
    if not sep or role not in ("ext", "wrist") or not serial.isdigit():
        raise ValueError(f"unexpected PointWorld depth group {name!r}")
    return serial, role


@dataclass(frozen=True)
class EpisodeSummary:
    episode: str
    uuid: str
    scene_path: str
    hdf5_path: str
    canonical_timestamps: tuple[int, ...]
    ext_serials: tuple[str, str]  # (ext1, ext2) from flow attrs
    wrist_serials: tuple[str, ...]  # from depth groups "<serial>+wrist"
    clips: tuple[str, ...]
    track_counts: dict  # clip -> {serial: N}
    depth_lengths: dict  # "<serial>+<role>" -> frames
    optimization_success: bool


def summarize_episode(root: str | Path, episode: str, layout: dict = DEFAULT_LAYOUT) -> EpisodeSummary:
    """Read only attributes, shapes and the small camera JSON; checks the cross-file schema of one episode."""
    root = Path(root)
    flows_path = root / layout["flows"].format(episode=episode)
    depth_path = root / layout["depth"].format(episode=episode)
    cameras = json.loads((root / layout["cameras"].format(episode=episode)).read_text())
    with h5py.File(flows_path, "r") as flow, h5py.File(depth_path, "r") as depth:
        attrs = flow.attrs
        ext = (str(attrs["ext1_cam_serial"]), str(attrs["ext2_cam_serial"]))
        stamps = tuple(int(x) for x in json.loads(attrs["canonical_timestamps"]))
        if any(b <= a for a, b in zip(stamps, stamps[1:])):
            raise ValueError(f"{episode}: canonical timestamps are not strictly increasing")
        clips = sorted_clips(flow.keys())
        counts = {}
        for key in clips:
            start, end = clip_bounds(key)
            if end > len(stamps):
                raise ValueError(f"{episode}: clip {key} exceeds the canonical timeline ({len(stamps)})")
            groups = sorted(g for g in flow[key] if g.startswith("camera_"))
            expected = sorted(f"camera_{s}_ext" for s in ext)
            if groups != expected:
                raise ValueError(f"{episode} clip {key}: camera groups {groups} != ext1/ext2 attrs {expected}")
            counts[key] = {}
            for serial in ext:
                cam = flow[key][f"camera_{serial}_ext"]
                T, N, three = cam["scene_flows"].shape
                if three != 3 or T != end - start:
                    raise ValueError(f"{episode} clip {key}: scene_flows {cam['scene_flows'].shape} != ({end - start},N,3)")
                counts[key][serial] = int(N)
        roles = {}
        lengths = {}
        for name in depth:
            serial, role = parse_depth_group(name)
            roles.setdefault(role, []).append(serial)
            lengths[name] = int(depth[name]["depth"].shape[0])
            if depth[name]["timestamps"].shape != (lengths[name],):
                raise ValueError(f"{episode}: {name} depth/timestamps length mismatch")
        if sorted(roles.get("ext", [])) != sorted(ext):
            raise ValueError(f"{episode}: depth ext serials {roles.get('ext')} != flow ext serials {ext}")
        missing = [s for s in ext if s not in cameras]
        if missing:
            raise ValueError(f"{episode}: camera JSON misses external serials {missing}")
        if str(cameras.get("uuid", attrs["uuid"])) != str(attrs["uuid"]):
            raise ValueError(f"{episode}: camera JSON uuid != flow uuid")
        return EpisodeSummary(
            episode=episode,
            uuid=str(attrs["uuid"]),
            scene_path=str(attrs["scene_path"]),
            hdf5_path=str(attrs.get("hdf5_path", "")),
            canonical_timestamps=stamps,
            ext_serials=ext,
            wrist_serials=tuple(sorted(roles.get("wrist", []))),
            clips=tuple(clips),
            track_counts=counts,
            depth_lengths=lengths,
            optimization_success=bool(cameras.get("optimization_success", False)),
        )


def _close(handle) -> None:
    handle.close()


class PointWorldReader:
    """Bounded lazy reads of flow/depth HDF5 files: at most ``max_open`` files per process, small chunk caches."""

    def __init__(self, root: str | Path, layout: dict = DEFAULT_LAYOUT, *, max_open: int = 8, chunk_cache_bytes: int = 1 << 20):
        self.root, self.layout = Path(root), dict(layout)
        self.chunk_cache_bytes = int(chunk_cache_bytes)
        # one pool per file kind: a window holds its episode's flow *and* depth file at the same time
        self.flow_files = HandlePool(self._open_flows, _close, max_open=max_open)
        self.depth_files = HandlePool(self._open_depth, _close, max_open=max_open)
        self._cameras: dict[str, dict] = {}

    def _open(self, kind: str, episode: str):
        path = self.root / self.layout[kind].format(episode=episode)
        return h5py.File(path, "r", rdcc_nbytes=self.chunk_cache_bytes, rdcc_nslots=521)

    def _open_flows(self, episode: str):
        return self._open("flows", episode)

    def _open_depth(self, episode: str):
        return self._open("depth", episode)

    @property
    def files(self) -> HandlePool:  # diagnostics: the larger of the two pools
        return max(self.flow_files, self.depth_files, key=len)

    def flow(self, episode: str):
        return self.flow_files.get(episode)

    def depth(self, episode: str):
        return self.depth_files.get(episode)

    def cameras(self, episode: str) -> dict:
        if episode not in self._cameras:
            if len(self._cameras) >= 64:
                self._cameras.pop(next(iter(self._cameras)))
            path = self.root / self.layout["cameras"].format(episode=episode)
            self._cameras[episode] = json.loads(path.read_text())
        return self._cameras[episode]

    def depth_frames(self, episode: str, group: str, frames) -> tuple[np.ndarray, np.ndarray]:
        """(len(frames),180,320) metres (0 invalid) and their timestamps; reads only the selected frames."""
        g = self.depth(episode)[group]
        frames = np.asarray(frames, np.int64)
        order = np.argsort(frames)
        unique, inverse = np.unique(frames[order], return_inverse=True)
        data = g["depth"][unique]  # h5py fancy indexing needs increasing unique indices
        out = np.empty((len(frames),) + data.shape[1:], np.float32)
        out[order] = data[inverse].astype(np.float32) / 1000.0
        return out, g["timestamps"][:][frames]

    def depth_timestamps(self, episode: str, group: str) -> np.ndarray:
        return self.depth(episode)[group]["timestamps"][:]

    def close(self) -> None:
        self.flow_files.close()
        self.depth_files.close()
        self._cameras.clear()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_cameras"] = {}
        return state
