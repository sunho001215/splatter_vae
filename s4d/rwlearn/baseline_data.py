"""Real-world data adapters for the representation baselines, over the abstract per-camera ``FrameStore``
(``s4d.rwlearn.store``). Each class yields exactly the samples of that baseline's Meta-World adapter (same keys, shapes,
value ranges and camera conventions), so the trainers, models and losses stay unchanged. Real-world specifics:

* frames are stored once per camera; multi-view time steps come from the loader-side ``SyncIndex`` (reference camera
  frames with the nearest frame of every other camera within ``max_sync_ns``), single-camera DynaMo videos use each
  camera's own native frames;
* cameras are per EPISODE (own accepted calibration) and are attached per sample in the convention each baseline
  expects (SinCro: integer pixel centres + OpenGL c2w; SNeRL: rig K + normalised OpenGL poses; 3D-MVP: continuous K +
  OpenCV c2w);
* temporal windows never cross a recording pause or a dropped / unsynchronised frame (``run`` ids);
* SNeRL's semantic targets are the SAM 3 classes of the store; its DINO targets stay ONLINE in the trainer;
* the rig has three cameras, all training cameras.

SinCro's reference ``forward_sincro_batch`` builds every sample's rays from the FIRST sample's cameras (valid for one
fixed rig). ``sincro_forward_per_sample_cameras`` is that exact function with only the ray block replaced, so each
sample's rays come from its own episode's cameras (bit-identical when the cameras agree; tested).
"""

from __future__ import annotations

import inspect
import json
import os
import textwrap
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from s4d.baselines.data import GL_FROM_CV
from s4d.baselines.dynamo.data import official_slices
from s4d.baselines.snerl.data import SEMANTIC_CLASSES, normalise_c2w, official_K
from s4d.rwlearn.store import FrameStore, SyncIndex, camera_runs, open_store, sync_index

SHORT_RUN_STEPS = 10_000


# ----------------------------------------------------------------------------------------------- gate evidence
def check_gate(store: FrameStore, gate_report: str | Path | dict | None, *, steps: int | None,
               split: dict | None = None, target: str | None = None, caps: dict | None = None,
               consumer: str | None = None, smoke_without_calibration: bool = False) -> dict:
    """Pretraining on real-world data needs the user's full-data gate (``s4d.rwlearn.gate``). Beyond
    ``SHORT_RUN_STEPS`` (smoke / plumbing runs may go without it, flagged ``smoke``) all of these must hold:

    * a passed report whose validated episode set equals, per task, the store's episodes; for a GEOMETRY-DEPENDENT
      ``consumer`` (``gate.CONSUMERS``; an unknown or missing consumer counts as one) also the released calibration
      hold, a store built from each episode's validated calibration and only validated cameras
      (``gate.check_store_calibration``);
    * the store's task set equals the requested ``target`` (``tasks.TARGETS``: task-specific
      encoders only, user decision 2026-11-06);
    * the run's ``split`` is a disjoint train / validation partition of exactly the store's episodes;
    * no data cap truncates anything (``caps``: ``max_episodes`` / ``num_episodes`` None, ``max_frames_per_demo``
      None or at least the longest camera stream).

    A geometry-dependent consumer's short run also needs the validated calibration unless ``smoke_without_calibration``
    is set explicitly (synthetic plumbing only; ``gate.short_run_allowed``)."""
    report = gate_report
    if isinstance(report, (str, Path)):
        report = json.loads(Path(report).read_text())
    from s4d.rwlearn import gate as G

    store_tasks = {store.task(ep) for ep in store.episodes()}
    smoke = not (report and all(G.task_passed(report, t, consumer) for t in store_tasks))
    if smoke:
        if not G.short_run_allowed(consumer, steps, SHORT_RUN_STEPS, smoke_without_calibration):
            held = " (geometry-dependent consumer: the calibration hold must be released too)" \
                if G.consumer_info(consumer).geometry else ""
            raise RuntimeError(f"no passed full-data gate report for this store{held}; refusing a real-world run "
                               f"of {steps} steps (only <= {SHORT_RUN_STEPS}-step smoke runs may skip the gate)")
        return {"smoke": True, "consumer": consumer, "geometry_dependent": G.consumer_info(consumer).geometry}
    by_task: dict[str, list[str]] = {}
    for ep in store.episodes():
        by_task.setdefault(store.task(ep), []).append(ep)
    from s4d.rwlearn.identity import store_identities

    report = G.bind_store_episodes(report, by_task, store_identities(store))
    for task, eps in by_task.items():
        gate = report["tasks"].get(task)
        if gate is None or not gate.get("passed"):
            raise RuntimeError(f"{task}: not passed in the gate report")
        if sorted(gate["store_episodes"]) != sorted(eps):
            raise RuntimeError(f"{task}: store episodes differ from the gate's validated + preprocessed set")
    if G.consumer_info(consumer).geometry:
        G.check_store_calibration(report, by_task, getattr(store, "upstream", None), store.cameras)
    if target is not None:
        from s4d.rwlearn.configs import TARGET_TASKS

        if target not in TARGET_TASKS:
            raise RuntimeError(f"unknown target {target!r} (task-specific targets only: {sorted(TARGET_TASKS)})")
        if sorted(by_task) != sorted(TARGET_TASKS[target]):
            raise RuntimeError(f"store tasks {sorted(by_task)} != target {target} tasks {TARGET_TASKS[target]}")
    if split is not None:
        train, val = set(split.get("train", [])), set(split.get("validation", []))
        if (train & val or (train | val) != set(store.episodes()) or len(split["train"]) != len(train)
                or len(split.get("validation", [])) != len(val)):
            raise RuntimeError("the split is not a disjoint train / validation partition of the store's episodes")
    full_run = steps is None or steps > SHORT_RUN_STEPS
    for name, value in (caps or {}).items():  # only full runs must see all data; a cap that keeps all is fine
        if value is None or not full_run:
            continue
        if name == "max_frames_per_demo":
            limit = max(len(store.timestamps(ep, c)) for ep in store.episodes() for c in store.cameras(ep))
        else:  # episode caps
            limit = len(store.episodes())
        if int(value) < limit:
            raise RuntimeError(f"data cap {name}={value} would truncate the full-data set")
    return {"smoke": False, "tasks": sorted(by_task), "consumer": consumer,
            "geometry_dependent": G.consumer_info(consumer).geometry}


def _episodes(store: FrameStore, episodes, max_episodes):
    eps = list(episodes)[:max_episodes] if max_episodes is not None else list(episodes)
    missing = sorted(set(eps) - set(store.episodes()))
    if missing:
        raise KeyError(f"episodes not in the store: {missing[:5]}")
    return eps


class _SyncedEpisodes(Dataset):
    """Shared state: store (re-opened per worker from its locator when pickled), per-episode SyncIndex, cameras."""

    def __init__(self, locator, episodes, num_views: int, max_episodes=None, max_steps_per_demo=None,
                 max_sync_ns: int = 30_000_000):
        self.locator = locator
        self.store, self._pid = open_store(locator), None
        self.episodes = _episodes(self.store, episodes, max_episodes)
        self.sync: dict[str, SyncIndex] = {}
        for ep in self.episodes:
            s = sync_index(self.store, ep, max_sync_ns)
            if max_steps_per_demo is not None:
                cut = int(max_steps_per_demo)
                s = SyncIndex(s.cameras, s.frames[:cut], s.times[:cut], s.latest_times[:cut], s.segment[:cut],
                              s.run[:cut], s.max_sync_error_ns)
            if len(s.cameras) < num_views:
                raise ValueError(f"{ep}: {len(s.cameras)} cameras < num_views {num_views}")
            self.sync[ep] = s
        self.num_views = int(num_views)
        self.num_train_cams = min(len(s.cameras) for s in self.sync.values())
        self.lengths = {ep: len(s.times) for ep, s in self.sync.items()}
        self.K_cv = {ep: np.stack([self.store.intrinsics(ep, c) for c in self.sync[ep].cameras[:num_views]])
                     for ep in self.episodes}
        self.c2w_cv = {ep: np.stack([np.linalg.inv(self.store.w2c(ep, c)) for c in self.sync[ep].cameras[:num_views]])
                       for ep in self.episodes}
        self._release()

    def _release(self) -> None:
        """Drop a store opened from a locator, so DataLoader workers (forked or spawned) never share its file
        handles; every process reopens it lazily (``_store``), like the Meta-World adapters' lazy HDF5 handles."""
        if not isinstance(self.locator, FrameStore):
            self.store, self._pid = None, None

    def __getstate__(self):
        state = dict(self.__dict__)
        if not isinstance(self.locator, FrameStore):
            state["store"], state["_pid"] = None, None
        return state

    def _store(self) -> FrameStore:
        if self.store is None or getattr(self, "_pid", None) not in (None, os.getpid()):
            self.store, self._pid = open_store(self.locator), os.getpid()
        return self.store

    def rgb_step(self, ep: str, steps, views=None) -> np.ndarray:
        """(len(steps), V, H, W, 3) uint8 of synchronised steps."""
        s = self.sync[ep]
        views = range(self.num_views) if views is None else views
        steps = np.atleast_1d(steps)
        store = self._store()
        return np.stack([store.rgb(ep, s.cameras[v], s.frames[steps, v]) for v in views], 1)

    def close(self) -> None:  # SNeRL trainer API: release handles before DataLoader workers fork
        self._release()


class RealworldSinCroWindows(_SyncedEpisodes):
    """Reference ``MetaWorldSinCroSequenceDataset`` samples: ``images`` (T, H, V, W, 3) in [0, 1], ``K`` (integer pixel
    centres), ``c2w`` (OpenGL) of the window's episode; windows every ``temporal_stride`` steps, inside one run."""

    def __init__(self, path, episodes, sequence_length: int = 3, temporal_stride: int = 3, frame_spacing: int = 1,
                 num_views: int = 3, max_episodes: int | None = None, max_frames_per_demo: int | None = None):
        super().__init__(path, episodes, num_views, max_episodes, max_frames_per_demo)
        self.sequence_length, self.frame_spacing = int(sequence_length), int(frame_spacing)
        self.cams = {}
        for ep in self.episodes:
            K = self.K_cv[ep].copy()
            K[:, 0, 2] -= 0.5
            K[:, 1, 2] -= 0.5
            self.cams[ep] = (torch.from_numpy(K.astype(np.float32)),
                             torch.from_numpy((self.c2w_cv[ep] @ GL_FROM_CV).astype(np.float32)))
        span = (self.sequence_length - 1) * self.frame_spacing
        self.indices = [(ep, s) for ep in self.episodes for s in range(0, self.lengths[ep] - span, temporal_stride)
                        if self.sync[ep].run[s] == self.sync[ep].run[s + span]]
        if not self.indices:
            raise ValueError("no SinCro windows inside one recording run")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index: int) -> dict:
        ep, start = self.indices[index]
        steps = [start + k * self.frame_spacing for k in range(self.sequence_length)]
        images = torch.from_numpy(self.rgb_step(ep, steps)).float().div(255.0).permute(0, 2, 1, 3, 4)
        K, c2w = self.cams[ep]
        return {"images": images.contiguous(), "K": K.clone(), "c2w": c2w.clone(), "demo_key": ep,
                "hdf5_path": str(self.locator)}


class RealworldReViWoStates(_SyncedEpisodes):
    """Reference ``MetaWorldMultiViewAllCamerasHDF5Dataset`` samples: every synchronised step, all cameras, [-1, 1]."""

    def __init__(self, path, episodes, num_views: int = 3, max_episodes: int | None = None,
                 max_frames_per_demo: int | None = None):
        super().__init__(path, episodes, num_views, max_episodes, max_frames_per_demo)
        self.indices = [(ep, t) for ep in self.episodes for t in range(self.lengths[ep])]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index: int) -> dict:
        ep, t = self.indices[index]
        rgb = torch.from_numpy(self.rgb_step(ep, [t])[0])
        images = rgb.float().div(255.0).mul(2.0).sub(1.0).permute(0, 3, 1, 2)
        return {"images": images.contiguous(), "demo_key": ep, "t": t, "hdf5_path": str(self.locator)}


class RealworldDynaMoWindows(Dataset):
    """Official ``TrajectorySlicerDataset`` windows over (episode, camera, run) single-camera videos of each camera's
    NATIVE frames (no cross-camera matching); a pause or dropped frame ends a trajectory like an episode end."""

    def __init__(self, path, episodes, window: int = 5, frame_step: int = 1, max_episodes: int | None = None):
        self.locator, self.store = path, open_store(path)
        self.episodes = _episodes(self.store, episodes, max_episodes)
        self.window, self.frame_step = int(window), int(frame_step)
        self.trajectories = []
        for ep in self.episodes:
            for cam in self.store.cameras(ep):
                runs = camera_runs(self.store, ep, cam)
                for r in np.unique(runs):
                    idx = np.flatnonzero(runs == r)
                    self.trajectories.append((ep, cam, int(idx[0]), len(idx)))
        self.slices = official_slices([n for *_, n in self.trajectories], self.window, self.frame_step)
        if not self.slices:
            raise ValueError("no DynaMo windows: every trajectory is too short")
        self._release()

    _release = _SyncedEpisodes._release
    __getstate__ = _SyncedEpisodes.__getstate__
    _store = _SyncedEpisodes._store

    def __len__(self):
        return len(self.slices)

    def window_frames(self, index: int) -> tuple[str, str, list[int]]:
        i, start, end = self.slices[index]
        ep, cam, offset, _ = self.trajectories[i]
        return ep, cam, list(range(offset + start, offset + end, self.frame_step))

    def __getitem__(self, index: int) -> torch.Tensor:
        ep, cam, frames = self.window_frames(index)
        rgb = self._store().rgb(ep, cam, frames)
        return torch.from_numpy(np.ascontiguousarray(rgb)).permute(0, 3, 1, 2).unsqueeze(1).contiguous()


class RealworldMVPTimeSteps(_SyncedEpisodes):
    """``MVPTimeSteps`` samples (all cameras fused per synchronised step) with the episode's K / c2w."""

    def __init__(self, path, episodes, num_views: int = 3, max_episodes: int | None = None,
                 max_frames_per_demo: int | None = None):
        super().__init__(path, episodes, num_views, max_episodes, max_frames_per_demo)
        if num_views != self.num_train_cams:
            raise ValueError(f"num_views={num_views}: pretraining fuses all {self.num_train_cams} cameras")
        self.indices = [(ep, t) for ep in self.episodes for t in range(self.lengths[ep])]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item) -> dict:
        index = item[0] if isinstance(item, tuple) else item
        ep, t = self.indices[int(index)]
        s = self.sync[ep]
        rgb = self.rgb_step(ep, [t])[0]
        depth = np.stack([self._store().depth(ep, s.cameras[v], s.frames[[t], v])[0] for v in range(self.num_views)])
        return {"rgb": torch.from_numpy(np.ascontiguousarray(rgb)),
                "depth": torch.from_numpy(np.ascontiguousarray(depth.astype(np.float32))),
                "view_mask": torch.ones(self.num_views, dtype=torch.bool),
                "K": torch.from_numpy(self.K_cv[ep].astype(np.float32)),
                "c2w": torch.from_numpy(self.c2w_cv[ep].astype(np.float32)), "episode": ep, "t": t}


class RealworldSNeRLScenes(_SyncedEpisodes):
    """``SNeRLScenes`` (official draw order: uniform step over all steps, ``num_inputs`` distinct input cameras, target
    uniform over all cameras; draws from ``default_rng([seed, i])``) with per-episode cameras and SAM 3 targets."""

    def __init__(self, path, episodes, seed: int, num_inputs: int = 3, length: int | None = None,
                 lookat=(0.0, 0.0, 0.0), scale: float = 4.0):
        store = open_store(path)
        num_views = min(len(store.cameras(ep)) for ep in episodes)
        super().__init__(path, episodes, num_views)
        if not self._store().has_semantic:
            raise RuntimeError("SNeRL needs semantic masks in the store")
        self.seed, self.num_inputs = int(seed), int(num_inputs)
        self.lookat, self.scale = tuple(float(x) for x in lookat), float(scale)
        self.num_cams = self.num_views
        if self.num_inputs > self.num_cams:
            raise ValueError(f"{self.num_inputs} input views exceed the {self.num_cams} cameras")
        self.c2w_ep = {ep: normalise_c2w(torch.from_numpy(c), self.lookat, self.scale).float()
                       for ep, c in self.c2w_cv.items()}
        lengths = np.array([self.lengths[ep] for ep in self.episodes])
        self.offsets = np.concatenate([[0], np.cumsum(lengths)])
        self.length = int(length) if length is not None else int(self.offsets[-1])
        self.task = ",".join(sorted({self._store().task(ep) for ep in self.episodes}))
        self._release()
        self.semantic_classes = SEMANTIC_CLASSES

    def __len__(self):
        return self.length

    def draw(self, index: int) -> tuple[str, int, list[int], int]:
        rng = np.random.default_rng([self.seed, int(index)])
        frame = int(rng.integers(self.offsets[-1]))
        e = int(np.searchsorted(self.offsets, frame, side="right") - 1)
        inputs = rng.choice(self.num_cams, self.num_inputs, replace=False).tolist()
        target = int(rng.integers(self.num_cams))
        return self.episodes[e], frame - int(self.offsets[e]), inputs, target

    def camera(self, ep: str, view: int) -> tuple[np.ndarray, np.ndarray]:
        return self.K_cv[ep][view], self.c2w_cv[ep][view]

    def depth_unit(self) -> float:
        return 1.0  # the store returns metres

    def scene(self, ep: str, t: int, inputs: list[int], target: int | None, with_depth: bool = False) -> dict:
        s = self.sync[ep]
        store = self._store()
        c2w = self.c2w_ep[ep]
        out = {"obs_rgb": self.rgb_step(ep, [t], inputs)[0], "obs_c2w": c2w[inputs].clone(),
               "inputs": torch.as_tensor(inputs), "episode": ep, "t": int(t)}
        if target is not None:
            cam, f = s.cameras[target], s.frames[[t], target]
            out.update(target_rgb=store.rgb(ep, cam, f)[0], target_semantic=store.semantic(ep, cam, f)[0].astype(np.int64),
                       target_c2w=c2w[target].clone(), target_K=torch.from_numpy(official_K(self.K_cv[ep][target])),
                       target=int(target))
            if with_depth:
                out["target_depth"] = store.depth(ep, cam, f)[0].astype(np.float32)
        return out

    def __getitem__(self, index: int) -> dict:
        ep, t, inputs, target = self.draw(index)
        return self.scene(ep, t, inputs, target)


# ---------------------------------------------------------------------------------------------- SinCro rays
_REFERENCE_RAY_BLOCK = """\
    K_single = K_mats[0]   # [V, 3, 3]
    c2w_single = c2w[0]    # [V, 4, 4]

    rays_per_view = []
    for v in range(V):
        rays_o, rays_d = get_rays(H, W, K_single[v], c2w_single[v, :3, :4], device=device)
        rays_per_view.append(torch.stack([rays_o, rays_d], dim=0))
    rays_all = torch.stack(rays_per_view, dim=0)  # [V, 2, H, W, 3]

    tiled_rays = rays_all.unsqueeze(0).expand(B, -1, -1, -1, -1, -1)
    tiled_rays = tiled_rays.reshape(B * V, 2, H, W, 3)
"""
_PER_SAMPLE_RAY_BLOCK = """\
    K_single = K_mats[0]   # [V, 3, 3] (render() below only uses it for NDC rays)
    tiled_rays = torch.stack([torch.stack(get_rays(H, W, K_mats[b, v], c2w[b, v, :3, :4], device=device), dim=0)
                              for b in range(B) for v in range(V)], dim=0)  # [B*V, 2, H, W, 3]: own cameras
"""


def _derive_per_sample_forward():
    from s4d.baselines.sincro import training

    src = textwrap.dedent(inspect.getsource(training.forward_sincro_batch))
    if src.count(_REFERENCE_RAY_BLOCK) != 1:
        raise RuntimeError("the reference forward_sincro_batch ray block changed; re-derive the per-sample variant")
    src = src.replace(_REFERENCE_RAY_BLOCK, _PER_SAMPLE_RAY_BLOCK).replace(
        "def forward_sincro_batch(", "def sincro_forward_per_sample_cameras(", 1)
    namespace = dict(vars(training))
    exec(compile(src, f"<derived from {training.__file__}:forward_sincro_batch>", "exec"), namespace)  # noqa: S102
    fn = namespace["sincro_forward_per_sample_cameras"]
    fn.__doc__ = ("Reference ``forward_sincro_batch`` with each sample's rays from its own cameras (per-episode "
                  "calibrations); every other line is the reference source, derived at import time.")
    return fn


sincro_forward_per_sample_cameras = _derive_per_sample_forward()


def realworld_classes(interface: dict, steps: int | None, locator, caps: dict | None = None, *,
                      consumer: str) -> dict:
    """Trainer hook for ``data_interface.source: realworld`` (Meta-World runs never import this module): checks the
    full-data gate evidence (``data_interface.gate_report``) against the store, the run's split manifest, its target
    and its data caps (``check_gate``) AT THE CALLING TRAINER's consumer level, and returns the dataset classes +
    SinCro's forward. ``consumer`` is the trainer's fixed identity (``scripts/train_<b>.py``); a configuration that
    declares another ``data_interface.consumer`` is refused, so a config can never downgrade a geometric trainer."""
    if interface.get("source") != "realworld":
        raise ValueError(f"not a real-world data interface: {interface.get('source')!r}")
    declared = interface.get("consumer")
    if declared is not None and declared != consumer:
        raise ValueError(f"data_interface.consumer {declared!r} conflicts with the trainer's consumer {consumer!r}")
    split = json.loads(Path(interface["split_manifest"]).read_text()) if interface.get("split_manifest") else None
    store = open_store(locator)
    if getattr(store, "depth_source", None) == "native_realsense" and consumer != "mvp3d":
        raise ValueError("native_realsense pretraining is only supported for mvp3d")
    check_gate(store, interface.get("gate_report"), steps=steps, split=split,
               target=interface.get("target"), caps=caps, consumer=consumer,
               smoke_without_calibration=interface.get("smoke_without_calibration") is True)  # literal True only
    return {"sincro": RealworldSinCroWindows, "reviwo": RealworldReViWoStates, "dynamo": RealworldDynaMoWindows,
            "mvp3d": RealworldMVPTimeSteps, "snerl": RealworldSNeRLScenes,
            "sincro_forward": sincro_forward_per_sample_cameras}
