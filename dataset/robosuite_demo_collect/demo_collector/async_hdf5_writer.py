from __future__ import annotations

import json
import queue
import threading
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import h5py
import numpy as np

from .dino_dense import DINOv2DenseExtractor, DinoOutputSpec


@dataclass
class StepPacket:
    state: np.ndarray
    action: np.ndarray
    rgb_by_cam: Dict[str, np.ndarray]
    seg_by_cam: Dict[str, np.ndarray]
    t: Optional[float] = None


class AsyncHDF5Writer:
    """
    Owns the HDF5 file handle in a background thread.

    Important detail:
    - states/actions datasets MUST be created with correct second-dimension size
      at creation time (h5py maxshape cannot be changed later).
    - So we create them lazily on the first received StepPacket.
    """

    def __init__(
        self,
        output_path: str,
        camera_names: List[str],
        compression: str = "lzf",
        max_queue: int = 256,
        dino_device: str = "cuda",
        dino_source: str = "torchhub",
        dino_mode: str = "patch",
    ):
        self.output_path = output_path
        self.camera_names = camera_names
        self.compression = compression

        self.q: "queue.Queue[Tuple[str, Any]]" = queue.Queue(maxsize=max_queue)
        self.thread = threading.Thread(target=self._run, daemon=True)

        self._dino_kwargs = dict(
            device=dino_device,
            model_source=dino_source,
            mode=dino_mode,
        )

        self._started = False
        self.dropped_steps = 0

    def start(self, root_attrs: Dict[str, Any]) -> None:
        if self._started:
            return
        self._started = True
        self.q.put(("__init_file__", root_attrs))
        self.thread.start()

    def begin_demo(
        self,
        demo_name: str,
        model_xml: str,
        camera_intrinsics: Dict[str, np.ndarray],
        camera_extrinsics: Dict[str, np.ndarray],
        image_hw: Tuple[int, int],
        extra_demo_attrs: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload = dict(
            demo_name=demo_name,
            model_xml=model_xml,
            camera_intrinsics=camera_intrinsics,
            camera_extrinsics=camera_extrinsics,
            image_hw=image_hw,
            extra_demo_attrs=extra_demo_attrs or {},
        )
        self.q.put(("begin_demo", payload))

    def enqueue_step(self, pkt: StepPacket) -> None:
        try:
            self.q.put_nowait(("step", pkt))
        except queue.Full:
            self.dropped_steps += 1

    def end_demo(self) -> None:
        self.q.put(("end_demo", None))

    def close(self) -> None:
        self.q.put(("close", None))
        self.thread.join()

    # ----------------------- internal writer thread -----------------------

    def _make_extendable(self, group: h5py.Group, name: str, shape_tail, dtype, chunks=True):
        ds = group.create_dataset(
            name,
            shape=(0, *shape_tail),
            maxshape=(None, *shape_tail),
            dtype=dtype,
            chunks=chunks,
            compression=self.compression,
        )
        return ds

    def _append_row(self, ds: h5py.Dataset, row: np.ndarray) -> int:
        n = ds.shape[0]
        ds.resize((n + 1, *ds.shape[1:]))
        ds[n] = row
        return n

    def _run(self):
        f: Optional[h5py.File] = None
        data_grp: Optional[h5py.Group] = None

        demo_grp: Optional[h5py.Group] = None
        obs_grp: Optional[h5py.Group] = None

        ds_states: Optional[h5py.Dataset] = None
        ds_actions: Optional[h5py.Dataset] = None
        ds_rgb: Dict[str, h5py.Dataset] = {}
        ds_seg: Dict[str, h5py.Dataset] = {}
        ds_dino: Dict[str, h5py.Dataset] = {}

        dino_extractor: Optional[DINOv2DenseExtractor] = None
        dino_spec: Optional[DinoOutputSpec] = None

        while True:
            cmd, payload = self.q.get()

            if cmd == "__init_file__":
                root_attrs = payload
                f = h5py.File(self.output_path, "w")
                data_grp = f.create_group("data")
                for k, v in root_attrs.items():
                    data_grp.attrs[k] = v
                data_grp.attrs["total"] = 0
                self.q.task_done()
                continue

            if cmd == "begin_demo":
                assert f is not None and data_grp is not None

                if dino_extractor is None:
                    dino_extractor = DINOv2DenseExtractor(**self._dino_kwargs)

                demo_name = payload["demo_name"]
                model_xml = payload["model_xml"]
                H, W = payload["image_hw"]
                camera_intrinsics = payload["camera_intrinsics"]
                camera_extrinsics = payload["camera_extrinsics"]
                extra_demo_attrs = payload["extra_demo_attrs"]

                demo_grp = data_grp.create_group(demo_name)
                demo_grp.attrs["model_file"] = model_xml
                demo_grp.attrs["num_samples"] = 0
                demo_grp.attrs["camera_names"] = json.dumps(self.camera_names)

                cam_grp = demo_grp.create_group("camera_params")
                K_stack = np.stack([camera_intrinsics[c] for c in self.camera_names], axis=0)
                E_stack = np.stack([camera_extrinsics[c] for c in self.camera_names], axis=0)
                cam_grp.create_dataset("intrinsics", data=K_stack)
                cam_grp.create_dataset("extrinsics_world_T_cam", data=E_stack)

                for k, v in extra_demo_attrs.items():
                    demo_grp.attrs[k] = v

                # Lazily created on first step (dims unknown here)
                ds_states = None
                ds_actions = None

                obs_grp = demo_grp.create_group("obs")

                ds_rgb = {}
                ds_seg = {}
                ds_dino = {}
                dino_spec = None

                for cam in self.camera_names:
                    ds_rgb[cam] = self._make_extendable(obs_grp, f"{cam}_rgb", shape_tail=(H, W, 3), dtype=np.uint8)
                    ds_seg[cam] = self._make_extendable(obs_grp, f"{cam}_seg", shape_tail=(H, W), dtype=np.int32)

                self.q.task_done()
                continue

            if cmd == "step":
                assert f is not None and data_grp is not None
                assert demo_grp is not None and obs_grp is not None
                assert dino_extractor is not None

                pkt: StepPacket = payload

                # Create states/actions datasets with correct dims on first packet
                if ds_states is None:
                    state_dim = int(pkt.state.shape[0])
                    ds_states = demo_grp.create_dataset(
                        "states",
                        shape=(0, state_dim),
                        maxshape=(None, state_dim),
                        dtype=np.float64,
                        chunks=True,
                        compression=self.compression,
                    )

                if ds_actions is None:
                    action_dim = int(pkt.action.shape[0])
                    ds_actions = demo_grp.create_dataset(
                        "actions",
                        shape=(0, action_dim),
                        maxshape=(None, action_dim),
                        dtype=np.float32,
                        chunks=True,
                        compression=self.compression,
                    )

                self._append_row(ds_states, pkt.state.astype(np.float64))
                self._append_row(ds_actions, pkt.action.astype(np.float32))

                # RGB/SEG
                for cam in self.camera_names:
                    self._append_row(ds_rgb[cam], pkt.rgb_by_cam[cam].astype(np.uint8))
                    self._append_row(ds_seg[cam], pkt.seg_by_cam[cam].astype(np.int32))

                # DINO (batch 6 cams)
                imgs = np.stack([pkt.rgb_by_cam[cam] for cam in self.camera_names], axis=0)
                feats, spec = dino_extractor.extract(imgs)

                if dino_spec is None:
                    dino_spec = spec
                    for cam_i, cam in enumerate(self.camera_names):
                        tail = (spec.grid_h, spec.grid_w, spec.feat_dim)
                        ds_dino[cam] = self._make_extendable(obs_grp, f"{cam}_dinov2", shape_tail=tail, dtype=np.float16)
                    obs_grp.attrs["dinov2_patch_size"] = spec.patch_size
                    obs_grp.attrs["dinov2_feat_dim"] = spec.feat_dim
                    obs_grp.attrs["dinov2_mode"] = spec.mode

                for cam_i, cam in enumerate(self.camera_names):
                    self._append_row(ds_dino[cam], feats[cam_i].astype(np.float16))

                demo_grp.attrs["num_samples"] = int(ds_states.shape[0])
                data_grp.attrs["total"] = int(data_grp.attrs["total"]) + 1

                self.q.task_done()
                continue

            if cmd == "end_demo":
                if f is not None:
                    f.flush()
                self.q.task_done()
                continue

            if cmd == "close":
                if f is not None:
                    f.flush()
                    f.close()
                self.q.task_done()
                break
