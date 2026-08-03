from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

import h5py
import numpy as np


COLLECTION_VERSION = "metaworld-hdf5-v3"


@dataclass
class DemoMeta:
    env_id: str
    env_name: str
    seed: int
    policy_type: str
    policy_name: str
    camera_names: list[str]
    model_file: str = "unknown"


class HDF5DemoWriter:
    """Synchronously write aligned, LZF-compressed Meta-World demonstrations.

    Overwrite-mode collection is transactional: data is written to
    ``<output>.incomplete`` and published with an atomic rename only after every
    demo passes temporal-length validation. Compression is performed in the
    same thread as simulation, so the environment cannot advance while a frame
    is still being saved.
    """

    def __init__(self, path: str, mode: str, compression: Optional[str]) -> None:
        if compression not in (None, "lzf", "gzip"):
            raise ValueError("output.compression must be null|lzf|gzip")
        self.path = Path(path).expanduser().resolve()
        self.mode = str(mode)
        self.compression = compression
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._publish_on_close = self.mode == "overwrite"
        self._closed = False
        self._active_demo = False

        if self.mode == "overwrite":
            self.write_path = self.path.with_name(f"{self.path.name}.incomplete")
            if self.write_path.exists():
                self.write_path.unlink()
            self.f = h5py.File(self.write_path, "w")
        elif self.mode == "resume":
            self.write_path = self.path
            self.f = h5py.File(self.path, "a")
        else:
            raise ValueError("output.mode must be overwrite|resume")

        self.f.attrs.update(
            {
                "collection_version": COLLECTION_VERSION,
                "collection_complete": False,
                "compression": "none" if compression is None else compression,
                "synchronous_frame_writes": True,
            }
        )
        self.data_grp = self.f.require_group("data")
        self.data_grp.attrs.setdefault("total", 0)
        self.data_grp.attrs.setdefault("num_demos", 0)

    def next_demo_index(self) -> int:
        indices = []
        for key in self.data_grp:
            if not key.startswith("demo"):
                continue
            try:
                indices.append(int(key.removeprefix("demo")))
            except ValueError:
                continue
        return max(indices, default=0) + 1

    def _compression_kwargs(self, shape: tuple[int, ...]) -> dict:
        if self.compression is None or shape == () or not all(shape):
            return {}
        return {"compression": self.compression}

    def _create_fixed(
        self,
        group: h5py.Group,
        name: str,
        data: np.ndarray,
        *,
        dtype=None,
    ) -> h5py.Dataset:
        array = np.asarray(data)
        return group.create_dataset(
            name,
            data=array,
            dtype=dtype,
            **self._compression_kwargs(array.shape),
        )

    def _make_extendable(
        self,
        group: h5py.Group,
        name: str,
        shape_tail: tuple[int, ...],
        dtype,
    ) -> h5py.Dataset:
        elements_per_frame = int(np.prod(shape_tail, dtype=np.int64))
        chunk_frames = 1 if elements_per_frame >= 4096 else 256
        return group.create_dataset(
            name,
            shape=(0, *shape_tail),
            maxshape=(None, *shape_tail),
            dtype=dtype,
            chunks=(chunk_frames, *shape_tail),
            compression=self.compression,
        )

    def begin_demo(
        self,
        demo_name: str,
        meta: DemoMeta,
        *,
        H: int,
        W: int,
        camera_intrinsics: Dict[str, np.ndarray],
        camera_extrinsics: Dict[str, np.ndarray],
        extra_attrs: Optional[dict] = None,
        save_depth: bool = False,
        segmentation_objects: Optional[Sequence[dict]] = None,
    ) -> None:
        if self._active_demo:
            raise RuntimeError("The previous demo must be ended before beginning another.")
        if demo_name in self.data_grp:
            del self.data_grp[demo_name]
            self.f.flush()

        self.camera_names = list(meta.camera_names)
        self.demo_grp = self.data_grp.create_group(demo_name)
        self.demo_grp.attrs.update(
            {
                "env_id": meta.env_id,
                "env_name": meta.env_name,
                "seed": int(meta.seed),
                "policy_type": meta.policy_type,
                "policy_name": meta.policy_name,
                "model_file": meta.model_file,
                "num_samples": 0,
                "camera_names": json.dumps(self.camera_names),
                "collection_complete": False,
            }
        )
        if extra_attrs:
            for key, value in extra_attrs.items():
                self.demo_grp.attrs[str(key)] = value

        camera_group = self.demo_grp.create_group("camera_params")
        self._create_fixed(
            camera_group,
            "intrinsics",
            np.stack([camera_intrinsics[camera] for camera in self.camera_names]),
        )
        self._create_fixed(
            camera_group,
            "extrinsics_world_T_cam",
            np.stack([camera_extrinsics[camera] for camera in self.camera_names]),
        )
        if segmentation_objects:
            self._write_segmentation_metadata(segmentation_objects)

        self.obs_grp = self.demo_grp.create_group("obs")
        self.env_obs_grp = self.demo_grp.create_group("obs_env")
        self.ds_rgb = {
            camera: self._make_extendable(
                self.obs_grp, f"{camera}_rgb", (H, W, 3), np.uint8
            )
            for camera in self.camera_names
        }
        self.ds_depth = (
            {
                camera: self._make_extendable(
                    self.obs_grp, f"{camera}_depth", (H, W), np.float32
                )
                for camera in self.camera_names
            }
            if save_depth
            else None
        )
        self.ds_seg = {
            camera: self._make_extendable(
                self.obs_grp, f"{camera}_seg", (H, W), np.int32
            )
            for camera in self.camera_names
        }
        self.ds_seg_type: dict[str, h5py.Dataset] = {}
        self.ds_states = None
        self.ds_actions = None
        self.ds_rewards = None
        self.ds_dones = None
        self.ds_success = None
        self.ds_obs = None
        self._frame_count = 0
        self._active_demo = True

    def _write_segmentation_metadata(self, objects: Sequence[dict]) -> None:
        group = self.demo_grp.create_group("segmentation")
        string_dtype = h5py.string_dtype(encoding="utf-8")
        fields = {
            "ids": (np.asarray([int(obj.get("id", -1)) for obj in objects], np.int32), None),
            "types": (np.asarray([int(obj.get("type", -1)) for obj in objects], np.int32), None),
            "names": (np.asarray([str(obj.get("name", "")) for obj in objects], object), string_dtype),
            "type_names": (np.asarray([str(obj.get("type_name", "")) for obj in objects], object), string_dtype),
            "body_ids": (np.asarray([int(obj.get("body_id", -1)) for obj in objects], np.int32), None),
            "body_names": (np.asarray([str(obj.get("body_name", "")) for obj in objects], object), string_dtype),
            "body_paths": (np.asarray([str(obj.get("body_path", "")) for obj in objects], object), string_dtype),
        }
        for name, (values, dtype) in fields.items():
            self._create_fixed(group, name, values, dtype=dtype)

    @staticmethod
    def _append_row(dataset: h5py.Dataset, row: np.ndarray) -> None:
        index = dataset.shape[0]
        dataset.resize((index + 1, *dataset.shape[1:]))
        dataset[index] = row

    def _validate_step_inputs(
        self,
        rgb_by_cam: Dict[str, np.ndarray],
        seg_id_by_cam: Dict[str, np.ndarray],
        depth_by_cam: Optional[Dict[str, np.ndarray]],
        seg_type_by_cam: Optional[Dict[str, np.ndarray]],
    ) -> None:
        for camera in self.camera_names:
            if camera not in rgb_by_cam or camera not in seg_id_by_cam:
                raise ValueError(f"Missing RGB or segmentation frame for {camera}.")
            if self.ds_depth is not None and (
                depth_by_cam is None or camera not in depth_by_cam
            ):
                raise ValueError(f"Depth saving is enabled but {camera} depth is missing.")
            if seg_type_by_cam is not None and camera not in seg_type_by_cam:
                raise ValueError(f"Segmentation types are missing for {camera}.")

    def append_step(
        self,
        *,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
        success: bool,
        obs_vec: np.ndarray,
        rgb_by_cam: Dict[str, np.ndarray],
        seg_id_by_cam: Dict[str, np.ndarray],
        depth_by_cam: Optional[Dict[str, np.ndarray]] = None,
        seg_type_by_cam: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        if not self._active_demo:
            raise RuntimeError("begin_demo must be called before append_step.")
        self._validate_step_inputs(
            rgb_by_cam, seg_id_by_cam, depth_by_cam, seg_type_by_cam
        )

        if self.ds_states is None:
            self.ds_states = self._make_extendable(
                self.demo_grp, "states", (state.size,), np.float64
            )
            self.ds_actions = self._make_extendable(
                self.demo_grp, "actions", (action.size,), np.float32
            )
            self.ds_rewards = self._make_extendable(
                self.demo_grp, "rewards", (), np.float32
            )
            self.ds_dones = self._make_extendable(
                self.demo_grp, "dones", (), np.uint8
            )
            self.ds_success = self._make_extendable(
                self.demo_grp, "success", (), np.uint8
            )
            self.ds_obs = self._make_extendable(
                self.env_obs_grp, "obs", (obs_vec.size,), np.float32
            )

        self._append_row(self.ds_states, np.asarray(state, dtype=np.float64))
        self._append_row(self.ds_actions, np.asarray(action, dtype=np.float32))
        self._append_row(self.ds_rewards, np.asarray(reward, dtype=np.float32))
        self._append_row(self.ds_dones, np.asarray(int(done), dtype=np.uint8))
        self._append_row(self.ds_success, np.asarray(int(success), dtype=np.uint8))
        self._append_row(self.ds_obs, np.asarray(obs_vec, dtype=np.float32))
        for camera in self.camera_names:
            self._append_row(self.ds_rgb[camera], np.asarray(rgb_by_cam[camera], np.uint8))
            if self.ds_depth is not None:
                self._append_row(
                    self.ds_depth[camera], np.asarray(depth_by_cam[camera], np.float32)
                )
            self._append_row(
                self.ds_seg[camera], np.asarray(seg_id_by_cam[camera], np.int32)
            )

        if seg_type_by_cam is not None:
            for camera in self.camera_names:
                if camera not in self.ds_seg_type:
                    self.ds_seg_type[camera] = self._make_extendable(
                        self.obs_grp,
                        f"{camera}_seg_type",
                        self.ds_seg[camera].shape[1:],
                        np.int32,
                    )
                self._append_row(
                    self.ds_seg_type[camera],
                    np.asarray(seg_type_by_cam[camera], np.int32),
                )

        self._frame_count += 1
        self.demo_grp.attrs["num_samples"] = self._frame_count
        self.data_grp.attrs["total"] = int(self.data_grp.attrs["total"]) + 1

    def _temporal_datasets(self) -> list[h5py.Dataset]:
        datasets = [
            self.ds_states,
            self.ds_actions,
            self.ds_rewards,
            self.ds_dones,
            self.ds_success,
            self.ds_obs,
            *self.ds_rgb.values(),
            *self.ds_seg.values(),
            *self.ds_seg_type.values(),
        ]
        if self.ds_depth is not None:
            datasets.extend(self.ds_depth.values())
        return [dataset for dataset in datasets if dataset is not None]

    def end_demo(self, expected_frames: Optional[int] = None) -> None:
        if not self._active_demo:
            raise RuntimeError("No active demo to end.")
        lengths = {dataset.name: int(dataset.shape[0]) for dataset in self._temporal_datasets()}
        unique_lengths = set(lengths.values())
        if unique_lengths != {self._frame_count} or self._frame_count <= 0:
            raise RuntimeError(
                f"Temporal datasets are misaligned in {self.demo_grp.name}: {lengths}"
            )
        if expected_frames is not None and self._frame_count != int(expected_frames):
            raise RuntimeError(
                f"Expected {expected_frames} frames, saved {self._frame_count} "
                f"in {self.demo_grp.name}."
            )
        self.demo_grp.attrs["num_samples"] = self._frame_count
        self.demo_grp.attrs["collection_complete"] = True
        self.data_grp.attrs["num_demos"] = sum(
            bool(group.attrs.get("collection_complete", False))
            for group in self.data_grp.values()
        )
        self.f.flush()
        self._active_demo = False

    def close(self, expected_demos: Optional[int] = None) -> None:
        if self._closed:
            return
        if self._active_demo:
            raise RuntimeError("Cannot publish while a demo is still active.")
        demos = [group for group in self.data_grp.values() if isinstance(group, h5py.Group)]
        incomplete = [group.name for group in demos if not bool(group.attrs.get("collection_complete", False))]
        if incomplete:
            raise RuntimeError(f"Cannot publish with incomplete demos: {incomplete}")
        if expected_demos is not None and len(demos) != int(expected_demos):
            raise RuntimeError(f"Expected {expected_demos} demos, found {len(demos)}.")
        self.data_grp.attrs["num_demos"] = len(demos)
        self.f.attrs["collection_complete"] = True
        self.f.flush()
        self.f.close()
        self._closed = True

        if self._publish_on_close:
            with self.write_path.open("rb") as handle:
                os.fsync(handle.fileno())
            os.replace(self.write_path, self.path)
            directory_fd = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)

    def abort(self) -> None:
        if not self._closed:
            self.f.flush()
            self.f.close()
            self._closed = True
