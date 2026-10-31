"""Adapter for raw DROID episodes (droid_raw 1.0.1 layout: metadata_<uuid>.json, trajectory.h5, recordings/SVO/*.svo).

Used to validate the real-world pipeline on genuine stereo pairs from one rig (ZED 2 external cameras). It reads the
raw side-by-side SVO stream (unrectified), the ZED factory calibration of each serial (``SN<serial>.conf`` as
downloaded by the ZED SDK from calib.stereolabs.com; parsed by ``droid_svo.zed_stereo_calibration``) and the robot
state, and offers the DROID-3D per-episode extrinsics (``extrinsics.jsonl``, world->camera view matrices of the
SDK-rectified left camera) as provided camera poses. Frames are decoded once per camera by the isolated
``svo_decode`` worker into ``cache_dir`` (never inside the source tree) and served from a memory map.

Configuration::

    adapter:
      name: droid_raw_svo
      config:
        episodes: {<uuid>: <raw episode dir>}
        calibration_dir: <dir with SN<serial>.conf>
        cameras: [ext1, ext2]            # roles from the metadata; the moving wrist camera is not a fixed rig
        cache_dir: <decoded-frame cache, outside the source tree>
        decoder: {python: <env with PyAV>, timeout_s: 1800}
        droid3d_extrinsics: <extrinsics.jsonl>   # optional: provided camera poses
        droid3d_intrinsics: <camera_intrinsics.jsonl>  # optional: SDK baseline cross-check
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

from s4d.realworld.adapters.base import EpisodeAdapter, register_adapter
from s4d.realworld.adapters.droid_svo import index_svo, zed_stereo_calibration
from s4d.realworld.contract import (
    CameraStream,
    EpisodeMeta,
    Intrinsics,
    Provenance,
    RobotDescription,
    RobotTrajectory,
    SourceFile,
    StereoGeometry,
)

ROLES = ("ext1", "ext2", "wrist")
VIEW_KEYS = {"ext1": "T0_ext1_in_world", "ext2": "T1_ext2_in_world"}


def _jsonl_record(path: str | Path, episode_id: str) -> dict | None:
    key = f'"{episode_id}"'
    with open(path) as f:
        for line in f:
            if key in line:
                record = json.loads(line)
                if record.get("episode_uuid") == episode_id:
                    return record
    return None


@register_adapter
class DroidRawSvoAdapter(EpisodeAdapter):
    name = "droid_raw_svo"
    version = "1"

    def __init__(self, config: dict):
        super().__init__(config)
        self.episodes = {k: Path(v) for k, v in config["episodes"].items()}
        self.cameras = tuple(config.get("cameras", ("ext1", "ext2")))
        if any(r not in ROLES for r in self.cameras):
            raise ValueError(f"cameras must be among {ROLES}")
        self.cache_dir = Path(config["cache_dir"])
        for src in self.episodes.values():
            if self.cache_dir.resolve().is_relative_to(src.resolve()):
                raise ValueError("cache_dir must be outside the source episode directories")
        self._frames: dict = {}

    # ------------------------------------------------------------ helpers

    def _metadata(self, episode_id: str) -> dict:
        path = self.episodes[episode_id] / f"metadata_{episode_id}.json"
        meta = json.loads(path.read_text())
        if meta["uuid"] != episode_id:
            raise ValueError(f"{path}: uuid {meta['uuid']!r} != {episode_id!r}")
        return meta

    def _serial(self, meta: dict, role: str) -> str:
        return str(meta[f"{role}_cam_serial"])

    def _svo(self, episode_id: str, serial: str) -> Path:
        return self.episodes[episode_id] / "recordings" / "SVO" / f"{serial}.svo"

    def _conf(self, serial: str) -> Path:
        return Path(self.config["calibration_dir"]) / f"SN{serial}.conf"

    def _step_map(self, episode_id: str, serial: str, traj: h5py.File) -> tuple[np.ndarray, np.ndarray]:
        """SVO frame timestamps and the trajectory step each frame belongs to (exact millisecond match with
        ``<serial>_frame_received``; frames without a step are an error)."""
        stamps = np.array([r.timestamp_ns for r in index_svo(self._svo(episode_id, serial)).records], np.int64)
        received = traj[f"observation/timestamp/cameras/{serial}_frame_received"][:]
        lookup = {int(v): i for i, v in enumerate(received)}
        steps = np.array([lookup.get(int(t // 1_000_000), -1) for t in stamps])
        if (steps < 0).any():
            raise ValueError(f"{serial}: {(steps < 0).sum()} SVO frames match no trajectory step")
        if np.any(np.diff(steps) <= 0):
            raise ValueError(f"{serial}: SVO frames map to non-increasing trajectory steps")
        return stamps, steps

    # ------------------------------------------------------------ adapter interface

    def list_episodes(self) -> list[str]:
        return sorted(self.episodes)

    def read_meta(self, episode_id: str) -> EpisodeMeta:
        meta = self._metadata(episode_id)
        streams, stereo, stamps_by_stream, notes, assumptions = [], [], {}, {}, []
        with h5py.File(self.episodes[episode_id] / "trajectory.h5", "r") as traj:
            for role in self.cameras:
                serial = self._serial(meta, role)
                cal = zed_stereo_calibration(self._conf(serial).read_text(), serial, "HD")
                stamps, steps = self._step_map(episode_id, serial, traj)
                for eye, e in (("left", cal.left), ("right", cal.right)):
                    sid = f"{role}/{eye}"
                    streams.append(CameraStream(
                        stream_id=sid, rig_id=role, eye=eye, modality="rgb", channels=3, channel_order="RGB",
                        dtype="uint8",
                        intrinsics=Intrinsics(cal.width, cal.height, e.fx, e.fy, e.cx, e.cy, "integer_centers",
                                              e.distortion_model, e.distortion,
                                              source=f"zed_factory_calibration SN{serial}.conf [{eye.upper()}_CAM_HD]"),
                        rectified=False, device_serial=serial,
                        sensor_name="ZED Mini" if role == "wrist" else "ZED 2",
                        timestamp_source="ZED SDK image timestamp recorded in the SVO (host clock)",
                        notes="raw side-by-side SVO half, decoded with the svo_decode worker"))
                    stamps_by_stream[sid] = stamps
                stereo.append(StereoGeometry(rig_id=role, left_stream=f"{role}/left", right_stream=f"{role}/right",
                                             status="metric", T_right_left=cal.T_right_left,
                                             baseline_m=cal.baseline_m,
                                             source=f"zed_factory_calibration SN{serial}.conf [STEREO]; {cal.reading}"))
                notes[role] = {"serial": serial, "trajectory_step_of_frame": steps.tolist()}
                if self.config.get("droid3d_intrinsics"):
                    rec = _jsonl_record(self.config["droid3d_intrinsics"], episode_id)
                    if rec is not None:
                        sdk = rec["cameras"][role]
                        notes[role]["sdk_baseline_m"] = sdk["stereo_baseline_m"]
                        if str(sdk["serial"]) != serial:
                            raise ValueError(f"{role}: DROID-3D serial {sdk['serial']} != metadata serial {serial}")
                        if abs(sdk["stereo_baseline_m"] - cal.baseline_m) > 1e-4:
                            raise ValueError(f"{role}: factory baseline {cal.baseline_m} != SDK {sdk['stereo_baseline_m']}")
            rs = "observation/robot_state"
            ts = ((traj["observation/timestamp/robot_state/read_start"][:].astype(np.int64)
                   + traj["observation/timestamp/robot_state/read_end"][:].astype(np.int64)) * 500_000)
            cart = traj[f"{rs}/cartesian_position"][:].astype(np.float64)
            rc = self.config.get("robot", {})
            ee = None
            if rc.get("flange_frame"):  # DROID cartesian_position = flange pose (checked against URDF FK, see docs)
                from scipy.spatial.transform import Rotation

                ee = np.tile(np.eye(4), (len(cart), 1, 1))
                ee[:, :3, :3] = Rotation.from_euler("xyz", cart[:, 3:]).as_matrix()
                ee[:, :3, 3] = cart[:, :3]
            trajectory = RobotTrajectory(
                timestamps_ns=ts, joint_positions=traj[f"{rs}/joint_positions"][:].astype(np.float64),
                gripper_state=traj[f"{rs}/gripper_position"][:].astype(np.float64),
                ee_pose_base=ee, ee_frame=rc.get("flange_frame") if ee is not None else None,
                extra={"cartesian_position": cart},
                extra_units={"cartesian_position": "m (xyz) + rad (extrinsic xyz Euler), DROID controller EE frame"})
        assumptions += [
            f"robot {meta.get('robot_serial')} recorded as franka_panda (same 7-DoF kinematic chain); "
            "robot-grounded calibration is not used for DROID",
            "gripper: Robotiq 2F-85, DROID gripper_position = normalized closure (0 open, 1 closed)",
            "robot timestamps = midpoint of the host-clock robot_state read_start/read_end (ms); camera timestamps "
            "= SVO record timestamps (host clock, equal to <serial>_frame_received in ms)",
            "ZED factory calibration reading (rotation, centre sign, DISTO model) validated against the SDK-rectified "
            "DROID MP4 left views; see docs/realworld_droid_validation.md",
        ]
        rc = self.config.get("robot", {})
        if rc.get("urdf_path"):  # composed arm + gripper URDF (scripts/realworld_compose_robot_urdf.py)
            manifest = json.loads(Path(rc["urdf_path"]).with_suffix(".manifest.json").read_text())
            robot = RobotDescription(
                arm_model="franka_panda", gripper_model=rc.get("gripper_model", "robotiq_2f_85"),
                urdf_path=str(rc["urdf_path"]), mesh_root=None, base_frame=rc["base_frame"],
                flange_frame=rc["flange_frame"], gripper_base_frame=manifest["mount"]["gripper_root"],
                T_flange_gripper=tuple(map(tuple, manifest["mount"]["T_parent_gripper"])),
                mount_source=manifest["mount"]["source"], joint_names=tuple(rc["joint_names"]),
                gripper_state_semantics="normalized_closure", gripper_joint=rc["gripper_joint"],
                asset_sha256={"urdf": manifest["urdf_sha256"]})
            assumptions.append(f"robot assets: {manifest['provenance']}; mount: {manifest['mount']['source']}")
        else:
            robot = RobotDescription(arm_model="franka_panda", gripper_model="robotiq_2f_85", urdf_path=None,
                                     mesh_root=None, mount_source="DROID standard mount (not measured here)")
        prov = Provenance(adapter=self.name, adapter_version=self.version,
                          source_root=str(self.episodes[episode_id]),
                          source_files=tuple(SourceFile(str(p), "", p.stat().st_size)
                                             for p in self.source_files(episode_id)),
                          created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                          notes=json.dumps({"uuid": meta["uuid"], "task": meta.get("current_task"),
                                            "cameras": notes}))
        return EpisodeMeta(episode_id=episode_id, streams=tuple(streams), stereo=tuple(stereo), robot=robot,
                           trajectory=trajectory, stream_timestamps_ns=stamps_by_stream, provenance=prov,
                           clock="DROID host clock (laptop): ZED image timestamps and robot_state read times",
                           assumptions=tuple(assumptions))

    def _decoded(self, episode_id: str, serial: str) -> np.ndarray:
        key = (episode_id, serial)
        if key in self._frames:
            return self._frames[key]
        cache = self.cache_dir / episode_id / f"{serial}.npy"
        record = cache.with_suffix(".json")
        svo = self._svo(episode_id, serial)
        if not (cache.is_file() and record.is_file()
                and json.loads(record.read_text()).get("svo_bytes") == svo.stat().st_size):
            from s4d.realworld.workers import WorkerSpec, call_worker

            dec = self.config["decoder"]
            spec = WorkerSpec("svo_decode", dec["python"], timeout_s=float(dec.get("timeout_s", 1800)))
            work = self.cache_dir / "_work" / episode_id / serial
            resp, out = call_worker(spec, "decode_svo", {"svo_path": str(svo)}, {}, work)
            cache.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache.with_suffix(".tmp.npy")
            np.save(tmp, out["frames"])
            tmp.replace(cache)
            record.write_text(json.dumps({"svo": str(svo), "svo_bytes": svo.stat().st_size,
                                          "decoder": resp["model"], "meta": resp["meta"],
                                          "timestamps_ns": out["timestamps_ns"].tolist()}, indent=1))
            for name in ("inputs.npz", "outputs.npz"):
                (work / name).unlink(missing_ok=True)
        self._frames[key] = np.load(cache, mmap_mode="r")
        return self._frames[key]

    def iter_frames(self, episode_id: str, stream_id: str, start: int = 0, stop: int | None = None):
        role, eye = stream_id.split("/")
        serial = self._serial(self._metadata(episode_id), role)
        frames = self._decoded(episode_id, serial)
        stamps = json.loads((self.cache_dir / episode_id / f"{serial}.json").read_text())["timestamps_ns"]
        width = frames.shape[2] // 2
        half = slice(0, width) if eye == "left" else slice(width, 2 * width)
        for i in range(start, len(frames) if stop is None else stop):
            yield i, int(stamps[i]), np.ascontiguousarray(frames[i][:, half])

    def source_files(self, episode_id: str) -> list[Path]:
        meta = self._metadata(episode_id)
        root = self.episodes[episode_id]
        files = [root / f"metadata_{episode_id}.json", root / "trajectory.h5"]
        for role in self.cameras:
            serial = self._serial(meta, role)
            files += [self._svo(episode_id, serial), self._conf(serial)]
        return files

    def provided_extrinsics(self, episode_id: str) -> dict:
        """DROID-3D world->camera view matrices (= T_cam_base) of the fixed cameras, with their source."""
        path = self.config.get("droid3d_extrinsics")
        if not path:
            return {}
        rec = _jsonl_record(path, episode_id)
        if rec is None:
            return {}
        out = {}
        for role in self.cameras:
            if role in VIEW_KEYS:
                out[f"{role}/left"] = {"T_cam_base": np.asarray(rec[VIEW_KEYS[role]], float),
                                       "source": {"kind": "droid3d_extrinsics", "path": str(path),
                                                  "field": VIEW_KEYS[role], "convention": "world->camera view matrix "
                                                  "of the SDK-rectified left camera (README 'Conventions' 1)",
                                                  "meta": rec.get("_meta", {})}}
        return out
