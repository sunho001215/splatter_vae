"""Real-world baseline pretraining configs, DERIVED from the accepted Meta-World configs (``configs/baselines/<b>/
drawer-open.yaml``) by an explicit list of overrides, each with its reason (docs/realworld_baselines.md). Every other
key - architecture, optimiser, schedule, step counts, loss weights - stays the documented protocol value.

``python -I scripts/realworld_learning.py configs`` writes ``configs/realworld/baselines/<b>/<target>.yaml``; a test
asserts the committed files equal ``derive_all()``.
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

from s4d.rwlearn.tasks import TARGET_TASKS as TARGET_TASKS  # the authoritative target registry (re-exported)
from s4d.rwlearn.tasks import TARGETS as TARGETS

REPO = Path(__file__).resolve().parents[2]
BASELINES = ("snerl", "mvp3d", "sincro", "reviwo", "dynamo")
REMOTE_REALWORLD_ROOT = "/home/compu/kaist/sunho/realworld"  # remote campaign root of the real-world data (role host)
NUM_CAMERAS = 3

# Scene geometry measured on the processed episodes (s4d/rwlearn/store.py::scene_statistics, drawer-open-000000 and
# mug-hanging-test-000001; identical fixed rig): least-squares intersection of the optical axes and camera distance.
SNERL_LOOKAT = [0.334, -0.031, 0.353]  # metres, robot base frame
SNERL_POSE_SCALE = 3.79  # 4 / median camera distance (1.055 m): cameras 4 official units from the origin
MVP3D_BOUNDS = [-0.15, -0.7, 0.15, 1.05, 0.6, 1.0]  # >= 99 % of robot/object/fixture depth points of both tasks

# covering, measured 2026-11-12 on its full calibration-validated store (98 episodes, scene-stats; agent COV,
# docs/realworld_baselines.md): optical-axis lookat [0.3222, -0.0306, 0.346] m, median camera distance 1.070 m
# (0.996-1.107) -> pose_scale 4 / 1.070 = 3.74; MVP3D_BOUNDS hold (robot 99.65 %, object 100 %, fixture 100 % inside);
# SNeRL 2/6-unit window 67.6 % of valid depth, SinCro 0.1/2.5 m 98.3 % (as the measured tasks).
COVERING_LOOKAT = [0.322, -0.031, 0.346]
COVERING_POSE_SCALE = 3.74
# pick-and-place, measured 2026-11-13 on its full calibration-validated geometry store (97 episodes,
# learning_inputs/calv2-4368f22/pick-and-place/store.yaml, scene-stats with released 5ea3dd1 code; agent CALSYNC,
# rwx/results/scene/pick-and-place-scene-stats.json sha 71095e5c): optical-axis lookat [0.322, -0.0304, 0.346] m, median
# camera distance 1.0695 m (0.9965-1.1080) -> pose_scale 4 / 1.0695 = 3.74; MVP3D_BOUNDS hold (robot 99.61 %, object
# 99.98 %, fixture 100 % inside); SNeRL 2/6-unit window 66.4 % of valid depth, SinCro 0.1/2.5 m 98.4 %.
PICK_AND_PLACE_LOOKAT = [0.322, -0.030, 0.346]
PICK_AND_PLACE_POSE_SCALE = 3.74

# Per-target scene geometry. drawer-open / mug-hanging map to exactly the values above. A new task starts at
# None = NOT MEASURED. Raw flange poses fit inside MVP3D_BOUNDS, but they cannot establish the
# validated camera poses (lookat, pose_scale) or the object/fixture depth extent (bounds). Measure each new task on its
# calibration-validated processed store (``scripts/realworld_learning.py scene-stats --store <store.yaml>`` =
# store.scene_statistics: optical-axis least-squares lookat, 4 / median camera distance, >= 99 % robot/object/fixture
# containment, foreground near/far coverage; docs/realworld_baselines.md), set the measured values here, regenerate
# the configs and release. Until then the planner refuses the scene-dependent consumers (``pending_scene_parameters``).
SCENE_PARAMETERS = {
    "drawer-open": {"lookat": SNERL_LOOKAT, "pose_scale": SNERL_POSE_SCALE, "bounds": MVP3D_BOUNDS},
    "mug-hanging": {"lookat": SNERL_LOOKAT, "pose_scale": SNERL_POSE_SCALE, "bounds": MVP3D_BOUNDS},
    "covering": {"lookat": COVERING_LOOKAT, "pose_scale": COVERING_POSE_SCALE, "bounds": MVP3D_BOUNDS},
    "pick-and-place": {"lookat": PICK_AND_PLACE_LOOKAT, "pose_scale": PICK_AND_PLACE_POSE_SCALE, "bounds": MVP3D_BOUNDS},
}
SCENE_KEYS = {"snerl": ("lookat", "pose_scale"), "mvp3d": ("bounds",)}  # the baselines that read scene geometry
POSE_SCALE_NOTE = ("cameras 4 official units from the origin, as official / Meta-World (median distance 1.055 m); "
                   "official near/far 2/6 -> 0.53-1.58 m camera depth, which holds the workspace (68.5 % of all valid "
                   "depth of the centre-cropped views; the rest is distant room background)")
POSE_SCALE_NOTES = {"covering": "cameras 4 official units from the origin, as official / Meta-World (median distance "
                                "1.070 m, covering store); official near/far 2/6 -> 0.53-1.60 m camera depth (67.6 % of "
                                "all valid depth of the centre-cropped views; the rest is distant room background)",
                    "pick-and-place": "cameras 4 official units from the origin, as official / Meta-World (median "
                                      "distance 1.0695 m, pick-and-place store); official near/far 2/6 -> 0.53-1.60 m "
                                      "camera depth (66.4 % of all valid depth of the centre-cropped views; the rest is "
                                      "distant room background)"}
PENDING_SCENE = ("NOT MEASURED for this task (TODO): measure with store.scene_statistics on its validated processed "
                 "episodes (docs/realworld_baselines.md); null on purpose, the planner refuses this baseline until "
                 "then")


def pending_scene_parameters(baseline: str, target: str) -> list[str]:
    """The scene parameters ``baseline`` reads that are not measured yet for ``target`` (empty: ready)."""
    return [k for k in SCENE_KEYS.get(baseline, ()) if SCENE_PARAMETERS[target][k] is None]


def run_inputs_dir(target: str) -> str:
    """Remote directory the launch tooling (s4d/rwlearn/launch.py ``prepare``) fills for one target: the store
    descriptor (``store.yaml``: kind + location of the compact training store + episode ids), the gate report and the
    split manifest. Only written after the full-data gate passed."""
    return f"{REMOTE_REALWORLD_ROOT}/learning_inputs/{target}"


def _geometry(baseline: str) -> bool:
    from s4d.rwlearn.gate import consumer_info

    return consumer_info(baseline).geometry


def overrides(baseline: str, target: str) -> list[tuple[str, object, str]]:
    """(dotted key, value, reason) applied to the Meta-World drawer-open config."""
    scene = SCENE_PARAMETERS[target]
    common = [
        ("dataset.hdf5_path", f"{run_inputs_dir(target)}/store.yaml",
         "for source realworld this key holds the FrameStore descriptor (s4d/rwlearn/store.py; compact per-camera "
         "training store, centre crop 128 px), not an HDF5 file; the trainers pass it through unchanged"),
        ("data_interface.split_manifest", f"{run_inputs_dir(target)}/split_seed0.json",
         "seeded per-task 90/10 EPISODE-level split, fixed and recorded (s4d/rwlearn/store.py::split_episodes, "
         "seed 0); validation episodes are never trained on"),
        ("data_interface.gate_report", f"{run_inputs_dir(target)}/gate_report.json",
         "full-data gate evidence (s4d/rwlearn/gate.py); runs > 10k steps refuse without a passed report whose "
         "episode sets equal the store's"),
        ("data_interface.consumer", baseline,
         ("GEOMETRY-DEPENDENT consumer (s4d/rwlearn/gate.py CONSUMERS): also needs the calibration hold released "
          "(calibration_validated per episode) and a store built from the validated calibration")
         if _geometry(baseline) else
         "non-geometric consumer (s4d/rwlearn/gate.py CONSUMERS): RGB / timestamps only, not held by calibration"),
        ("data_interface.source", "realworld",
         "selects the real-world dataset classes (per-episode cameras, loader-side sync, run-bounded windows)"),
        ("data_interface.target", target, "task(s) in the store"),
    ]
    spec = {
        "snerl": [
            ("data_interface.lookat", scene["lookat"],
             "normalised scene origin: least-squares intersection of the three optical axes (Meta-World: rig look-at)"),
            ("data_interface.pose_scale", scene["pose_scale"], POSE_SCALE_NOTES.get(target, POSE_SCALE_NOTE)),
            ("dataset.num_inputs", 3, "3 input cameras (user decision 2026-11-06): the rig has exactly 3, so every "
                                      "training target view is also an input view (Meta-World: 3 of 6). "
                                      "Validation (separate episodes) reports input-view reconstruction and "
                                      "leave-one-camera-out novel-view reconstruction separately "
                                      "(s4d/rwlearn/snerl_eval.py)"),
        ],
        "sincro": [
            ("dataset.num_views", NUM_CAMERAS, "all three cameras (Meta-World: 6 training cameras)"),
            ("data_interface.sincro_encoder_num_view", 6,
             "the trainer builds the encoder with this num_view: upstream MaskedViTEncoder makes num_view // 3 + 1 "
             "decoder image tokens, which must equal 1 primary + 2 reference views = 3; 6 keeps the Meta-World encoder "
             "architecture exactly (with num_view 3 upstream asserts); the three views enter through the data"),
            ("data_interface.frame_spacing", 1,
             "one camera frame = one real-world policy step (Meta-World: 2 sim steps = one RL step)"),
        ],
        "reviwo": [
            ("dataset.camera_num", NUM_CAMERAS, "all three cameras"),
            ("train.camera_num", NUM_CAMERAS, "all three cameras"),
        ],
        "dynamo": [
            ("data_interface.frame_spacing", 1, "one camera frame = one real-world policy step (upstream frame_step 1)"),
            ("projector.output_dim", 14,
             "chosen per the README rule 'approximately the underlying state dimension': 7 arm joints + gripper + "
             "6-DoF manipulated object (Meta-World: its 39-d observation)"),
        ],
        "mvp3d": [
            ("dataset.num_views", NUM_CAMERAS, "fuse all three cameras per time step (the original fused setting)"),
            ("render.bounds", scene["bounds"],
             "real-world scene bounds in the robot base frame (measured, see docs/realworld_baselines.md)"),
        ],
    }[baseline]
    spec = [(key, value, PENDING_SCENE if value is None else why) for key, value, why in spec]
    return common + spec


def _set(cfg: dict, dotted: str, value) -> None:
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = copy.deepcopy(value)


def derive(baseline: str, target: str) -> tuple[dict, list]:
    src = REPO / "configs" / "baselines" / baseline / "drawer-open.yaml"
    cfg = yaml.safe_load(src.read_text())
    ov = overrides(baseline, target)
    for key, value, _ in ov:
        _set(cfg, key, value)
    return cfg, ov


def render(baseline: str, target: str) -> str:
    cfg, ov = derive(baseline, target)
    lines = [f"# {baseline} pretraining on the REAL-WORLD {target} store (generated by s4d/rwlearn/configs.py; do not",
             f"# edit by hand). Base: configs/baselines/{baseline}/drawer-open.yaml (accepted Meta-World protocol);",
             "# every other key is the documented protocol value. Overrides and reasons:"]
    for key, value, why in ov:
        lines.append(f"#   {key} = {value}: {why}")
    lines.append("# Plan rows only through scripts/realworld_learning.py plan (full-data gate); see "
                 "docs/realworld_baselines.md.")
    return "\n".join(lines) + "\n" + yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False, width=120)


def derive_all() -> dict[Path, str]:
    return {REPO / "configs" / "realworld" / "baselines" / b / f"{t}.yaml": render(b, t)
            for b in BASELINES for t in TARGETS}


def write_all() -> list[Path]:
    out = []
    for path, text in derive_all().items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        out.append(path)
    return out
