"""Our method (Original Splatter4D) pretrained on the REAL-WORLD compact training store (user directive 2026-11-11).

The configuration is the RESOLVED simulation "Original" (the Arm4-recommended method: ``configs/metaworld/base.yaml``
+ a task file + method-frozen-v1's four registered overrides + ``train.steps=200000 train.seed=0``, exactly the
``runs/setup/tools/arm4_common.py`` reference stack) with an explicit list of overrides, each with its reason
(``OVERRIDES``). Architecture, masking, losses, optimiser, schedule and budget are untouched. The generated files
``configs/realworld/ours/<target>.yaml`` are tested to equal ``render(target)`` and to differ from the simulation
Original in exactly the listed keys (tests/test_rwlearn_ours.py). Deviations and their justification:
``/home/ws/ws/realworld-ops/ours/DEVIATIONS.md``; design: ``.../ours/DESIGN.md``.

Data path (``data.regime: realworld``, ``scripts/train.py build_data``): the store descriptor (``data.store``, kind
compact; any per-episode calibration -- e.g. covering's per-period rig documents -- is the episode's own accepted
``w2c`` in its store file), the full-data gate at the GEOMETRY level of consumer ``ours`` (``baseline_data.check_gate``:
100-episode / amended gate, QA families singleview+multiview+geometry, calibration hold, store built from the validated
calibration, split = partition of the store), the canonical seeded episode split ``split_seed0.json`` (validation
episodes are never trained on), and ``s4d.realworld.train_windows.RealworldWindowDataset`` (our method's window loader:
M5 cross-camera sets, TrackEverything moving tracks + gripper FK points as 3D track targets, FoundationStereo depth).
Meta-World runs never import this module.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import yaml

from s4d.rwlearn.tasks import TARGETS

REPO = Path(__file__).resolve().parents[2]
REMOTE_REALWORLD_ROOT = "/home/compu/kaist/sunho/realworld"
CONSUMER = "ours"
DELETE = object()  # an override that removes a Meta-World-only key
# The simulation Original (method-frozen-v1 rows s3-m3d-seed0-<task>-200k; arm4_common.reference_stack("original")).
# Its eight per-task configurations differ only in data.task and model.anchor_stats, which every real-world target
# replaces; drawer-open is the base (as configs/realworld/baselines/<b>/<target>.yaml use the Meta-World drawer-open).
SIM_FILES = ("configs/metaworld/base.yaml", "configs/metaworld/tasks/drawer-open.yaml")
SIM_SET = ("train.steps=200000", "train.seed=0", "aug.synth_views=2", "aug.synth_render=true",
           "loss.depth_valid_far=true", "loss.motion_space=gaussian")
STRIDES = [1, 2, 3]  # camera frames; one frame = one real-world policy step (Meta-World [2, 4, 6] = 1/2/3 RL steps)
VAL_STRIDES = [1, 3]  # Meta-World [2, 6] = 1 and 3 RL steps
# The three-camera rig's measured scene parameters (docs/realworld_baselines.md, s4d/rwlearn/configs.py
# SCENE_PARAMETERS; least-squares optical-axis intersection, median camera distance 1.055 m), used ONLY for the
# evaluator's orbit video (a diagnostic panel). None = not measured for the target: the orbit video is skipped.
ORBIT_DISTANCE_M = 1.055
# Targets whose rig was measured on their own store (covering 2026-11-12, scene-stats on the 98-episode validated
# store: median camera distance 1.070 m, configs.COVERING_POSE_SCALE = 4 / 1.070); the other targets use the value above.
ORBIT_DISTANCES_M = {"covering": 1.070, "pick-and-place": 1.0695}  # pick-and-place: scene-stats 2026-11-13, 97 episodes


def sim_original() -> dict:
    """The resolved simulation Original configuration (no run section)."""
    from s4d.config import load_config

    return load_config([REPO / f for f in SIM_FILES], list(SIM_SET))


def run_inputs_dir(target: str) -> str:
    return f"{REMOTE_REALWORLD_ROOT}/learning_inputs/{target}"


def overrides(target: str) -> list[tuple[str, object, str]]:
    """(dotted key, value or DELETE, reason) applied to the simulation Original."""
    if target not in TARGETS:
        raise ValueError(f"unknown real-world target {target!r} (task-specific targets only: {TARGETS})")
    from s4d.rwlearn.configs import SCENE_PARAMETERS

    lookat = SCENE_PARAMETERS[target]["lookat"]
    return [
        ("data.regime", "realworld",
         "real-world compact training store instead of the simulator HDF5 (s4d/rwlearn/ours.py build_realworld)"),
        ("data.task", target, "the real-world target task (task-specific encoders, user decision 2026-11-06)"),
        ("data.root", DELETE, "Meta-World data root; the store descriptor data.store replaces it"),
        ("data.train_ratio", DELETE,
         "Meta-World 96/4 episode split; replaced by the canonical seeded per-task 90/10 EPISODE split shared by every "
         "real-world method (data.split_manifest, s4d/rwlearn/store.py::split_episodes, seed 0)"),
        ("data.episodes", DELETE, "Meta-World overfit-gate episode list; not defined for the real-world store"),
        ("data.store", f"{run_inputs_dir(target)}/store.yaml",
         "FrameStore descriptor (kind compact: per-episode files with 128 px centre-crop RGB, FoundationStereo depth, "
         "TrackEverything moving tracks, robot state, M5 sync sets and the episode's accepted calibration); the plan "
         "row sets the calibration release's copy"),
        ("data.split_manifest", f"{run_inputs_dir(target)}/split_seed0.json",
         "the canonical split of every real-world method; validation episodes are never trained on"),
        ("data.gate_report", f"{run_inputs_dir(target)}/gate_report.json",
         "full-data gate evidence; runs > 10k steps refuse without a passed report at the geometry level of consumer "
         "'ours' (calibration hold released, store built from the validated calibration)"),
        ("data.consumer", CONSUMER, "GEOMETRY-DEPENDENT consumer 'ours' (s4d/rwlearn/gate.py CONSUMERS)"),
        ("data.target", target, "task(s) in the store"),
        ("data.strides", list(STRIDES),
         "camera frames between window frames: one frame = one real-world policy step (66.7 ms), so 1/2/3 frames are "
         "the simulation's 1/2/3 RL steps (Meta-World [2, 4, 6] simulator steps at action repeat 2); the FQL "
         "observation history (3 consecutive policy steps) lies inside the pretraining distribution"),
        ("data.val_strides", list(VAL_STRIDES),
         "the policy spacing and the largest pretraining stride (Meta-World [2, 6] = 1 and 3 RL steps)"),
        ("data.max_set_motion_m", 0.005,
         "M5 cross-camera set criterion (coordinator decision 2026-11-06), the loaders' default, recorded explicitly"),
        ("data.max_tracks", None, "every stored moving track of the window's query chunk (no subsampling)"),
        ("model.anchor_stats", None,
         "per-task real-world workspace statistics, computed by the plan's stats row with the Meta-World procedure "
         "(scripts/compute_workspace_stats.py --regime realworld over the TRAINING episodes) and set by the row; "
         "null refuses to start (s4d/train/loop.py anchor_statistics_path)"),
        ("aug.synth_camera_rig", "training_cameras",
         "the two synthetic views are jittered around the sample's OWN calibrated cameras with the Meta-World offsets "
         "(augment.jitter_training_cameras); the default rig is the Meta-World camera rig in the Meta-World world "
         "frame and is refused outside Meta-World (augment.synth_camera_rig)"),
        ("eval.orbit_center", lookat,
         "diagnostic orbit video only: the rig's measured optical-axis intersection in the robot base frame "
         "(SCENE_PARAMETERS; null = not measured for this target, the orbit video is skipped)"),
        ("eval.orbit_distance", ORBIT_DISTANCES_M.get(target, ORBIT_DISTANCE_M) if lookat is not None else None,
         "diagnostic orbit video only: the measured median camera distance"),
        ("wandb.project", "splatter4d-realworld", "real-world runs are logged apart from the Meta-World project"),
    ]


def _set(cfg: dict, dotted: str, value) -> None:
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    if value is DELETE:
        del node[keys[-1]]
    else:
        node[keys[-1]] = copy.deepcopy(value)


def derive(target: str) -> tuple[dict, list]:
    cfg = sim_original()
    ov = overrides(target)
    for key, value, _ in ov:
        _set(cfg, key, value)
    return cfg, ov


def render(target: str) -> str:
    cfg, ov = derive(target)
    lines = [f"# Our method (Original Splatter4D, Arm4-recommended) pretraining on the REAL-WORLD {target} store",
             "# (generated by s4d/rwlearn/ours.py; do not edit by hand). Base: the RESOLVED simulation Original",
             f"# ({' + '.join(SIM_FILES)} + {' '.join(SIM_SET)}); every other key",
             "# is the simulation value. Overrides and reasons (deviations: realworld-ops/ours/DEVIATIONS.md):"]
    for key, value, why in ov:
        shown = "<deleted>" if value is DELETE else value
        lines.append(f"#   {key} = {shown}: {why}")
    lines.append("# Plan rows only through the registrar (full-data gate, stats row, calibration release).")
    return "\n".join(lines) + "\n" + yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False, width=120)


def config_path(target: str) -> Path:
    return REPO / "configs" / "realworld" / "ours" / f"{target}.yaml"


def derive_all() -> dict[Path, str]:
    return {config_path(t): render(t) for t in TARGETS}


def write_all() -> list[Path]:
    out = []
    for path, text in derive_all().items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        out.append(path)
    return out


# ----------------------------------------------------------------------------------------------------- data
def _windows_class():
    from s4d.realworld.train_windows import RealworldWindowDataset

    class OursRealworldWindows(RealworldWindowDataset):
        """``RealworldWindowDataset`` windows with the simulation's sampling rule and evaluator-facing additions:

        * stride uniformity (Meta-World ``MetaworldWindowDataset`` rule): with several strides, a start frame is kept
          only where EVERY stride yields a window, so the stride of a sample is exactly uniform (the base loader
          enumerates each stride independently: longer strides lose more start frames);
        * ``spread_episodes`` (validation loaders only): windows ordered round-robin across episodes and evenly over
          each episode's time, so the evaluator's first ``eval.max_batches`` batches cover every validation episode
          (the base order is episode by episode: 8 x 16 windows would come from the first episode alone);
        * ``meta["dt_seconds"]``: the reference camera's median frame period of the episode, so the probes' velocity
          targets are in m/s (Meta-World windows carry the simulator dt);
        * ``probe_state`` reordered from the store's [flange xyz, flange euler xyz, gripper] to [flange xyz, gripper,
          flange euler xyz], the slots ``s4d.diag.probes.probe_targets`` reads (0:3 hand, 4:7 second target): on real
          data the second probe target ("obj_pos") is the flange ORIENTATION; no object state is recorded."""

        PROBE_ORDER = (0, 1, 2, 6, 3, 4, 5)

        def __init__(self, *args, spread_episodes: bool = False, **kwargs):
            super().__init__(*args, **kwargs)
            self._dt = {}
            table = self.table
            if len(self.strides) > 1:
                _, inverse, counts = np.unique(table[:, :2], axis=0, return_inverse=True, return_counts=True)
                table = table[counts[inverse.reshape(-1)] == len(self.strides)]
            if spread_episodes:
                episode = table[:, 0]
                order = np.lexsort((table[:, 2], table[:, 1], episode))  # by episode, time, stride
                table = table[order]
                episode = table[:, 0]
                rank = np.zeros(len(table))
                for e in np.unique(episode):
                    rows = np.flatnonzero(episode == e)
                    rank[rows] = np.arange(len(rows)) / len(rows)
                table = table[np.lexsort((episode, rank))]  # round-robin over episodes, evenly over time
            if not len(table):
                raise ValueError("no window has every stride")
            self.table = table

        def _dt_seconds(self, ep: str) -> float:
            if ep not in self._dt:
                ef = self.store.episode(ep)
                stamps = np.asarray(ef.timestamps(ef.reference_camera), np.int64)
                self._dt[ep] = float(np.median(np.diff(stamps))) * 1e-9
            return self._dt[ep]

        def __getitem__(self, index):
            sample = super().__getitem__(index)
            i = int(index[0]) if isinstance(index, tuple) else int(index)
            ep = self.episodes[int(self.table[i, 0])]
            sample["probe_state"] = sample["probe_state"][:, list(self.PROBE_ORDER)].contiguous()
            sample["meta"]["dt_seconds"] = self._dt_seconds(ep)
            return sample

    return OursRealworldWindows


def open_compact_store(locator):
    """The compact store of a descriptor; derived stores (``depth_source``: the native-depth overlay of 3D-MVP) are
    refused: our method trains on the FoundationStereo depth of the compact store."""
    from s4d.realworld.train_store import CompactStore
    from s4d.rwlearn.store import open_store

    spec = locator
    if not isinstance(spec, dict) and not isinstance(spec, CompactStore):
        text = Path(locator).read_text()
        spec = json.loads(text) if str(locator).endswith(".json") else yaml.safe_load(text)
    if isinstance(spec, dict) and ("depth_source" in spec or spec.get("kind") != "compact"):
        raise ValueError("our method trains on the compact store's FoundationStereo depth (kind compact, no "
                         f"depth_source overlay); got {sorted(spec)}")
    store = open_store(spec)
    if not isinstance(store, CompactStore):
        raise ValueError(f"expected a CompactStore, got {type(store).__name__}")
    return store


def load_split(path) -> dict:
    split = json.loads(Path(path).read_text())
    for key in ("train", "validation"):
        if not isinstance(split.get(key), list) or not all(isinstance(e, str) for e in split[key]):
            raise ValueError(f"split manifest {path}: '{key}' must be a list of store episode keys")
    if not split["train"] or not split["validation"]:
        raise ValueError(f"split manifest {path}: needs training and validation episodes")
    return split


def check_data(cfg: dict, store, split: dict) -> dict:
    """The full-data gate at the geometry level of consumer 'ours' (``baseline_data.check_gate``); a run of at most
    10k steps without a passed report needs the explicit ``data.smoke_without_calibration: true`` (plumbing only)."""
    from s4d.config import get
    from s4d.rwlearn.baseline_data import check_gate

    if get(cfg, "data.consumer", CONSUMER) != CONSUMER:
        raise ValueError(f"data.consumer {get(cfg, 'data.consumer')!r} conflicts with the trainer's consumer "
                         f"{CONSUMER!r}")
    train, val = split["train"], split["validation"]
    if (len(set(train)) != len(train) or len(set(val)) != len(val) or set(train) & set(val)
            or sorted(set(train) | set(val)) != store.episodes()):  # also for smoke runs (check_gate skips it there)
        raise RuntimeError("the split is not a disjoint train / validation partition of the store's episodes")
    steps = int(get(cfg, "train.steps"))
    stop = get(cfg, "train.stop_step", None)
    trained = min(steps, int(stop)) if stop else steps
    return check_gate(store, get(cfg, "data.gate_report"), steps=trained, split=split, target=get(cfg, "data.target"),
                      caps=None, consumer=CONSUMER,
                      smoke_without_calibration=get(cfg, "data.smoke_without_calibration") is True)


def window_kwargs(cfg: dict) -> dict:
    from s4d.config import get
    from s4d.realworld.train_store import MAX_SET_MOTION_M

    tracks = get(cfg, "data.max_tracks", None)
    return {"max_tracks": None if tracks is None else int(tracks),
            "max_set_motion_m": float(get(cfg, "data.max_set_motion_m", MAX_SET_MOTION_M)),
            "sync_criterion": "M5"}


def build_realworld(cfg: dict, loader) -> tuple:
    """(train_dataset, val_loaders, probe_loaders, n_train_cams) for ``scripts/train.py build_data``: training windows
    of the training episodes at every ``data.strides``; validation (validation episodes) and probe (training
    episodes) loaders per ``data.val_strides``, as the Meta-World branch."""
    from s4d.config import get

    store = open_compact_store(get(cfg, "data.store"))
    split = load_split(get(cfg, "data.split_manifest"))
    gate = check_data(cfg, store, split)
    windows = _windows_class()
    kw = window_kwargs(cfg)
    seed = int(get(cfg, "train.seed", 0))
    strides = tuple(int(s) for s in get(cfg, "data.strides"))
    val_strides = tuple(int(s) for s in get(cfg, "data.val_strides"))
    train_ds = windows(store, episodes=split["train"], split="all", strides=strides, seed=seed, **kw)
    train_ds.gate = gate
    val_loaders = {s: loader(windows(store, episodes=split["validation"], split="all", strides=(s,), seed=seed,
                                     spread_episodes=True, **kw), False) for s in val_strides}
    probe_loaders = {s: loader(windows(store, episodes=split["train"], split="all", strides=(s,), seed=seed, **kw),
                               True) for s in val_strides}
    cams = {len(c) for c in train_ds.cams}
    if len(cams) != 1:
        raise ValueError(f"every episode must provide the same number of cameras, got {sorted(cams)}")
    return train_ds, val_loaders, probe_loaders, cams.pop()
