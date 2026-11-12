"""Recorded replay (dry run) of the robot-side client on a recorded ``panda_teleop_episode`` H5: the same client
preprocessing as live deployment, with the recorded colour JPEG decoded by the training adapter (PIL), the recorded
native z16 depth aligned like the live client (default ``rs_align``: validated rs.align emulation -> colour-camera z;
or ``sensor_depth``: the release stage kernel), and the measured robot
state picked by the training binding rule (``actions.step_table``: latest robot / gripper sample at or before the
observation time). No camera, robot, network or torch. Multi-view M5 evidence comes from a released training-store
episode (``m5_from_store``) because the URDF surface model is not part of the client.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from s4d.realworld.frame_preprocessing import IMAGE_SIZE
from s4d.rwlearn.deploy.contracts import NATIVE_DEPTH
from s4d.rwlearn.deploy.history import Frame
from s4d.rwlearn.deploy.preprocessing import preprocess_frame
from s4d.rwlearn.deploy.realsense import (
    align_recorded_depth,
    aligned_native_depth,
    color_ray_denominator,
    emulate_rs_align,
)

ALIGNMENTS = ("rs_align", "sensor_depth")

EPISODE = "recording"


class RecordedEpisode:
    def __init__(self, path):
        import h5py

        from s4d.realworld.adapters.panda_teleop import PandaTeleopH5Adapter
        from s4d.rwlearn import actions as A

        self.path = str(path)
        self.adapter = PandaTeleopH5Adapter({"episodes": {EPISODE: self.path}})
        self.meta = self.adapter.read_meta(EPISODE)
        self.raw = A.load_raw(self.path)
        self._rigs = {g.rig_id: g for g in self.meta.rgbd}
        self._denominators = {}  # camera -> ((K, T, grid), colour-z denominator): computed once, as RealSenseRGBD does
        with h5py.File(self.path, "r") as f:
            self._numbers = {cam: np.asarray(f["cameras"][f"cam_{cam}"]["frame_number"][()], np.int64)[
                self.raw.cameras[cam]["paired_colour"]] for cam in self._rigs}

    def cameras(self) -> list[str]:
        return sorted(self._rigs)

    def timestamps_ns(self, cam: str) -> np.ndarray:
        """Colour frame times of the adapter stream (recorder host CLOCK_MONOTONIC), stream index order."""
        return np.asarray(self.meta.stream_timestamps_ns[self._rigs[cam].color_stream], np.int64)

    def frame_numbers(self, cam: str) -> np.ndarray:
        """Device colour frame numbers of the adapter stream (the ``Frame.seq`` of replay)."""
        return self._numbers[cam]

    def K_native(self, cam: str) -> np.ndarray:
        return self.meta.stream(self._rigs[cam].color_stream).intrinsics.K()

    def native_hw(self, cam: str) -> tuple[int, int]:
        i = self.meta.stream(self._rigs[cam].color_stream).intrinsics
        return int(i.height), int(i.width)

    def native(self, cam: str, index: int, *, depth: bool, alignment: str = "rs_align") -> dict:
        """Colour frame ``index`` (stream order) decoded as the training adapter does [+ the recorded native depth
        aligned to it: colour-camera z metres with validity]: the client's input before preprocessing."""
        if alignment not in ALIGNMENTS:
            raise ValueError(f"alignment must be one of {ALIGNMENTS}")
        g = self._rigs[cam]
        (_, t, rgb), = self.adapter.iter_frames(EPISODE, g.color_stream, index, index + 1)
        out = {"rgb": rgb, "timestamp_ns": int(t), "seq": int(self._numbers[cam][index])}
        if depth:
            ds = self.meta.stream(g.depth_stream)
            (_, td, raw), = self.adapter.iter_frames(EPISODE, g.depth_stream, index, index + 1)
            Kc, Kd = self.K_native(cam), ds.intrinsics.K()
            if alignment == "rs_align":
                T = g.T()
                z16 = emulate_rs_align(raw, Kd, Kc, T, rgb.shape[:2], ds.depth_scale_m)
                key = (np.asarray(Kc, np.float64).tobytes(), np.asarray(T, np.float64).tobytes(), z16.shape)
                if self._denominators.get(cam, (None,))[0] != key:
                    self._denominators[cam] = (key, color_ray_denominator(Kc, T, z16.shape))
                d, v = aligned_native_depth(z16, ds.depth_scale_m, Kc, T, denominator=self._denominators[cam][1])
            else:
                aligned = align_recorded_depth(raw, Kd, Kc, g.T(), rgb.shape[:2], ds.depth_scale_m)
                d, v = aligned["depth"], aligned["valid"]
            out.update(depth=d, valid=v, depth_timestamp_ns=int(td), alignment=alignment)
        return out

    def frame(self, cam: str, index: int, *, depth: bool, correction: dict | None = None,
              out_size: int = IMAGE_SIZE, alignment: str = "rs_align") -> Frame:
        """The preprocessed ``Frame`` the client sends for colour frame ``index`` of ``cam``."""
        n = self.native(cam, index, depth=depth, alignment=alignment)
        p = preprocess_frame(n["rgb"], depth=n.get("depth"), valid=n.get("valid"), out_size=out_size,
                             native_correction=correction if depth else None)
        metadata = {"depth_source": NATIVE_DEPTH, "native_correction": correction,
                    "alignment": f"{alignment} (recorded replay)",
                    "depth_timestamp_ns": n["depth_timestamp_ns"]} if depth else {}
        return Frame(n["seq"], n["timestamp_ns"], p["rgb"], p.get("depth"), p.get("valid"), metadata)

    def state_at(self, t_ns: int) -> dict:
        """Measured flange pose7 + gripper register bound to observation time ``t_ns`` with their sample times."""
        from s4d.rwlearn import actions as A

        i = int(A.latest_at_or_before(self.raw.robot_t, np.array([t_ns], np.int64))[0])
        j = int(A.latest_at_or_before(self.raw.gripper_t, np.array([t_ns], np.int64))[0])
        if i < 0 or j < 0:
            raise ValueError("no robot or gripper sample at or before the observation time")
        return {"pose7": self.raw.measured[i].copy(), "pose_time_ns": int(self.raw.robot_t[i]),
                "gripper_register": float(self.raw.gripper_position[j]), "gripper_time_ns": int(self.raw.gripper_t[j])}


def m5_from_store(store_episode_h5, recording: RecordedEpisode) -> dict:
    """The training store's M5 sets of this recording: ``{"cameras", "steps": [{cam: stream index}], "disp": {reference
    seq: surface_disp_m}}`` (store frame indices are the adapter's colour stream indices; checked on timestamps)."""
    from s4d.realworld.train_store import EpisodeFile

    store = EpisodeFile(store_episode_h5)
    try:
        store.require_geometry()
        sets, cams = store.sync(), list(store.cameras)
        if "surface_disp_m" not in sets:
            raise ValueError("training store episode without M5 robot motion (schema < 1.1.0)")
        for c in cams:
            if not np.array_equal(store.timestamps(c), recording.timestamps_ns(c)):
                raise ValueError(f"{c}: store and recording frame times differ")
        frames, disp = np.asarray(sets["frames"], np.int64), np.asarray(sets["surface_disp_m"], np.float64)
    finally:
        store.close()
    keep = (frames >= 0).all(1)
    ref = recording.frame_numbers(cams[0])
    return {"cameras": cams, "steps": [dict(zip(cams, map(int, row))) for row in frames[keep]],
            "disp": {int(ref[row[0]]): float(d) for row, d in zip(frames[keep], disp[keep])}}


def native_hw_of(path) -> dict:
    rec = RecordedEpisode(Path(path))
    return {c: rec.native_hw(c) for c in rec.cameras()}
