from __future__ import annotations

import json
import queue
import threading
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import h5py
import numpy as np


@dataclass
class StepPacket:
    state: np.ndarray
    action: np.ndarray
    rgb_by_cam: Dict[str, np.ndarray]
    seg_by_cam: Dict[str, np.ndarray]
    obs: Dict[str, np.ndarray]
    t: Optional[float] = None


class AsyncHDF5Writer:
    """
    Owns the HDF5 file handle in a background thread.

    Required behavior (per your request):
      - No dropped steps: enqueue_step() BLOCKS when queue is full.
      - end_demo() waits until all queued work is finished and flushed.
    """

    def __init__(
        self,
        output_path: str,
        camera_names: List[str],
        compression: str = "lzf",
        max_queue: int = 256,
        block_on_full: bool = True,   # NEW: guarantee no drops if True
    ):
        self.output_path = output_path
        self.camera_names = camera_names
        self.compression = compression
        self.block_on_full = bool(block_on_full)

        self.q: "queue.Queue[Tuple[str, Any]]" = queue.Queue(maxsize=max_queue)
        self.thread = threading.Thread(target=self._run, daemon=True)

        self._started = False

        # With block_on_full=True, this should remain 0.
        # Kept only for debugging / sanity checks.
        self.dropped_steps = 0

    def start(self, root_attrs: Dict[str, Any]) -> None:
        if self._started:
            return
        self._started = True
        self.q.put(("__init_file__", root_attrs))  # blocking put is fine here
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
        self.q.put(("begin_demo", payload))  # blocking

    def enqueue_step(self, pkt: StepPacket) -> None:
        """
        Enqueue one step for writing.
        """
        if self.block_on_full:
            # BLOCK until there is space: guarantees no dropped steps.
            self.q.put(("step", pkt))
            return

        # Legacy "drop" mode 
        try:
            self.q.put_nowait(("step", pkt))
        except queue.Full:
            self.dropped_steps += 1

    def end_demo(self, wait: bool = True) -> None:
        """
        End current demo.

        Required behavior:
          - When wait=True, do not return until:
              1) all prior 'step' items are written, and
              2) the 'end_demo' flush is executed.

        Implementation:
          - Put an 'end_demo' marker into the queue.
          - Call q.join() to wait until queue is fully drained (task_done called).
        """
        self.q.put(("end_demo", None))  # blocking
        if wait:
            # Wait until ALL queued commands (including this end_demo) are processed.
            self.q.join()

    def flush(self) -> None:
        """
        Optional helper: wait until the queue is empty (all tasks done).
        Useful if you want explicit sync points.
        """
        self.q.join()

    def close(self) -> None:
        """
        Close writer thread and file.

        We first ensure everything is written, then send 'close' command.
        """
        self.q.join()               # wait pending tasks
        self.q.put(("close", None)) # request close
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
        # -----------------------------
        # Writer thread state (owned ONLY by this thread)
        # -----------------------------
        # HDF5 file handle and the top-level group (/data)
        f: Optional[h5py.File] = None
        data_grp: Optional[h5py.Group] = None

        # Per-demo groups: /data/demoX, and its subgroups
        demo_grp: Optional[h5py.Group] = None   # current demo group
        obs_grp: Optional[h5py.Group] = None    # current demo's "obs" group (images)
        env_obs_grp: Optional[h5py.Group] = None  # current demo's "obs_env" group (low-dim dict)

        # Lazily created datasets
        ds_states: Optional[h5py.Dataset] = None
        ds_actions: Optional[h5py.Dataset] = None

        # Per-camera image datasets for the current demo
        ds_rgb: Dict[str, h5py.Dataset] = {}
        ds_seg: Dict[str, h5py.Dataset] = {}

        # Per-key env observation datasets for the current demo
        ds_obs_dict: Dict[str, h5py.Dataset] = {}

        # -----------------------------
        # Main loop: consume commands from the queue forever
        # -----------------------------
        while True:
            # Block until there is a command to process.
            # Each queued item is (cmd, payload).
            cmd, payload = self.q.get()

            # =========================================================
            # (1) "__init_file__": create the HDF5 file and /data group
            # =========================================================
            if cmd == "__init_file__":
                # Payload is root attributes dict
                root_attrs = payload

                # Create new HDF5 file
                f = h5py.File(self.output_path, "w")

                # Create /data root group
                data_grp = f.create_group("data")

                # Attach user-provided root metadata as attributes on /data
                for k, v in root_attrs.items():
                    data_grp.attrs[k] = v

                # Track total number of steps written across all demos
                data_grp.attrs["total"] = 0

                # Mark this queue item complete
                self.q.task_done()
                continue

            # =========================================================
            # (2) "begin_demo": allocate per-demo groups and fixed-shape datasets
            # =========================================================
            if cmd == "begin_demo":
                # File and /data must already exist
                assert f is not None and data_grp is not None

                # Unpack demo metadata
                demo_name = payload["demo_name"]                   # e.g., "demo1"
                model_xml = payload["model_xml"]                   # MJCF XML snapshot
                H, W = payload["image_hw"]                         # image resolution
                camera_intrinsics = payload["camera_intrinsics"]   # dict cam -> K
                camera_extrinsics = payload["camera_extrinsics"]   # dict cam -> world_T_cam
                extra_demo_attrs = payload["extra_demo_attrs"]     # any additional per-demo attrs

                # Create demo group: /data/demoX
                demo_grp = data_grp.create_group(demo_name)

                # Save demo-level metadata
                demo_grp.attrs["model_file"] = model_xml
                demo_grp.attrs["num_samples"] = 0  # will be updated as steps are appended
                demo_grp.attrs["camera_names"] = json.dumps(self.camera_names)

                # Save camera parameters in a consistent camera order
                cam_grp = demo_grp.create_group("camera_params")
                K_stack = np.stack([camera_intrinsics[c] for c in self.camera_names], axis=0)
                E_stack = np.stack([camera_extrinsics[c] for c in self.camera_names], axis=0)
                cam_grp.create_dataset("intrinsics", data=K_stack)
                cam_grp.create_dataset("extrinsics_world_T_cam", data=E_stack)

                # Store additional per-demo attributes (control freq, device, notes, etc.)
                for k, v in extra_demo_attrs.items():
                    demo_grp.attrs[k] = v

                # Reset per-demo state/action datasets.
                # These are created lazily on the first "step" because we don’t know
                # state_dim and action_dim until we see actual arrays.
                ds_states = None
                ds_actions = None

                # Create image observation group: /data/demoX/obs
                obs_grp = demo_grp.create_group("obs")

                # Create per-camera extendable datasets for RGB and segmentation.
                # Shape is (T, H, W, C) for RGB and (T, H, W) for seg.
                ds_rgb = {}
                ds_seg = {}
                for cam in self.camera_names:
                    ds_rgb[cam] = self._make_extendable(
                        obs_grp, f"{cam}_rgb", shape_tail=(H, W, 3), dtype=np.uint8
                    )
                    ds_seg[cam] = self._make_extendable(
                        obs_grp, f"{cam}_seg", shape_tail=(H, W), dtype=np.int32
                    )

                # Create env observation group: /data/demoX/obs_env
                # This stores the raw robosuite obs dict (one dataset per key).
                env_obs_grp = demo_grp.create_group("obs_env")
                ds_obs_dict = {}

                # Mark this queue item complete
                self.q.task_done()
                continue

            # =========================================================
            # (3) "step": append one timestep of data to all datasets
            # =========================================================
            if cmd == "step":
                # Must already be inside an active demo
                assert f is not None and data_grp is not None
                assert demo_grp is not None and obs_grp is not None

                pkt: StepPacket = payload

                # ---- Create states/actions datasets lazily (only once per demo) ----
                # This is required because HDF5 datasets must know their second dimension
                # at creation time (state_dim/action_dim are unknown until first packet).
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

                # ---- Append state/action for this timestep ----
                self._append_row(ds_states, pkt.state.astype(np.float64))
                self._append_row(ds_actions, pkt.action.astype(np.float32))

                # ---- Append low-dimensional env observations (obs dict) ----
                # Create one extendable dataset per obs key on first step.
                # After creation, append one row per key each step.
                if env_obs_grp is not None:
                    obs_val = pkt.obs

                    # Create datasets for each observation key on the first step
                    if not ds_obs_dict:
                        for k, v in obs_val.items():
                            arr = np.asarray(v)
                            ds_obs_dict[str(k)] = self._make_extendable(
                                env_obs_grp,
                                name=str(k),
                                shape_tail=arr.shape,
                                dtype=arr.dtype,
                                chunks=True,
                            )

                    # Append per-key observation arrays for this timestep
                    for k, v in obs_val.items():
                        arr = np.asarray(v)
                        self._append_row(ds_obs_dict[str(k)], arr)

                # ---- Append per-camera images (RGB and SEG) ----
                # Each camera gets one frame appended for this timestep.
                for cam in self.camera_names:
                    self._append_row(ds_rgb[cam], pkt.rgb_by_cam[cam].astype(np.uint8))
                    self._append_row(ds_seg[cam], pkt.seg_by_cam[cam].astype(np.int32))

                # ---- Update counters / metadata ----
                # num_samples reflects number of timesteps in this demo (same as states length)
                demo_grp.attrs["num_samples"] = int(ds_states.shape[0])

                # total increments for every step written across all demos
                data_grp.attrs["total"] = int(data_grp.attrs["total"]) + 1

                # Mark this queue item complete
                self.q.task_done()
                continue

            # =========================================================
            # (4) "end_demo": flush HDF5 buffers (sync point for end_demo(wait=True))
            # =========================================================
            if cmd == "end_demo":
                # Flush file buffers so that all writes are persisted to disk.
                # When the main thread enqueues "end_demo" and then calls q.join(),
                # this flush is guaranteed to have happened once q.join() returns.
                if f is not None:
                    f.flush()

                # Mark this queue item complete
                self.q.task_done()
                continue

            # =========================================================
            # (5) "close": final flush + close file, then exit writer thread
            # =========================================================
            if cmd == "close":
                # Ensure everything is flushed and then close the file handle.
                # After this, the writer thread will terminate.
                if f is not None:
                    f.flush()
                    f.close()

                # Mark this queue item complete and break out of loop to stop thread
                self.q.task_done()
                break
