"""Camera suite v2 (user directive 2026-10-29): the original protocol is untouched, the new groups are deterministic
and separate, the stronger paths have the registered amplitudes and are continuous, the multi-view replacement rule
holds for the new groups, and v2 records go to their own file."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from s4d.data.metaworld.cameras import EVAL_CAMERAS, LOOKAT, TRAIN_CAMERAS
from s4d.rl import camera_suites as CS
from s4d.rl.env import (
    HELDOUT_PATHS,
    HELDOUT_V2_PATHS,
    TRAIN_PATHS,
    FreeCamera,
    _camera_at,
    _forward,
    camera_c2w,
    heldout_v2_paths,
    lateral_offsets,
    orbit_camera,
    trajectory_path,
)
from s4d.rl.evaluate import (
    episode_seeds,
    evaluation_groups,
    evaluation_groups_v2,
    multiview_groups,
    nearest_training_camera,
    v2_episode_seeds,
)

REPO = Path(__file__).resolve().parents[1]
CFG = {"episodes_per_train_camera": 20, "episodes_per_heldout_camera": 20, "episodes_per_trajectory": 20}


def _original_trajectory_path(kind, base="train1", num_frames=72, lateral_amplitude=0.12, circular_azimuth_deg=10.0,
                              circular_elevation_deg=6.0):
    """Verbatim copy of ``trajectory_path`` before camera suite v2 (commit 795bec3)."""
    theta, phi = TRAIN_CAMERAS[int(base.removeprefix("train"))]
    camera = orbit_camera(theta, phi)
    if kind == "lateral":
        forward = _forward(camera)
        position = np.asarray(LOOKAT) - camera.distance * forward
        right = np.cross(forward, [0.0, 0.0, 1.0])
        right /= np.linalg.norm(right)
        return [_camera_at(position + offset * right) for offset in lateral_offsets(num_frames, lateral_amplitude)]
    if kind == "circular":
        angles = 2.0 * math.pi * np.arange(num_frames) / num_frames
        return [
            orbit_camera(theta + circular_elevation_deg * math.sin(a), phi + circular_azimuth_deg * math.cos(a))
            for a in angles
        ]
    raise ValueError(kind)


def _poses(path) -> np.ndarray:
    return np.asarray([[c.azimuth, c.elevation, c.distance] for c in path], dtype=np.float64)


def test_original_groups_paths_and_seeds_are_unchanged():
    groups = evaluation_groups(CFG)
    assert [(n, c) for n, (_, c) in groups.items()] == (
        [(f"train{i}", 20) for i in range(6)] + [(f"eval{i}", 20) for i in range(4)]
        + [("traj_lateral", 20), ("traj_circular", 20)])
    for i, (theta, phi) in enumerate(TRAIN_CAMERAS):
        assert groups[f"train{i}"][0] == [FreeCamera(phi, -theta)]
    for i, (theta, phi) in enumerate(EVAL_CAMERAS):
        assert groups[f"eval{i}"][0] == [FreeCamera(phi, -theta)] == HELDOUT_PATHS[f"eval{i}"]
    for kind in ("lateral", "circular"):  # bit-identical to the pre-v2 implementation
        assert np.array_equal(_poses(groups[f"traj_{kind}"][0]), _poses(_original_trajectory_path(kind)))
        assert trajectory_path(kind) == _original_trajectory_path(kind)
    # v1 episode seeds: hard-coded values of the unchanged stream (run seed, evaluation index)
    assert episode_seeds(2000, 100, 5) == [90844117, 732692417, 918469152, 365632154, 1457286247]
    assert episode_seeds(1000, 96, 3) == [579079079, 2132347755, 1766313095]
    digest = hashlib.sha256(json.dumps(episode_seeds(2001, 98, 240)).encode()).hexdigest()
    assert digest == "7e7d606ad458a001d3a168c48802424b699bb71716008253613d0723f57836d0"
    multi = evaluation_groups(CFG, views=3)  # multi-view v1 groups unchanged by the ref_ handling
    assert list(multi) == list(groups)
    for j in range(4):
        n = nearest_training_camera(HELDOUT_PATHS[f"eval{j}"])
        assert multi[f"eval{j}"][0] == (HELDOUT_PATHS[f"eval{j}"], *multi[f"train{n}"][0][1:])


def _checked(path: Path, key: str) -> dict:
    payload = json.loads(path.read_text())
    body = {k: v for k, v in payload.items() if k != key}
    assert hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest() == payload[key]
    return payload


def test_v2_suite_is_frozen_distinct_and_separated_from_training():
    assert len(CS.HELDOUT_V2_CAMERAS) == 4 and CS.HELDOUT_V2_NAMES == ("vis0", "vis1", "vis2", "vis3")
    assert not set(CS.HELDOUT_V2_NAMES) & set(HELDOUT_PATHS) and not set(CS.HELDOUT_V2_NAMES) & set(TRAIN_PATHS)
    train = [tuple(map(float, c)) for c in TRAIN_CAMERAS]
    # the registered rule and both amendments are unmodified and chained
    rule = _checked(REPO / CS.RULE_FILE, "rule_sha256")
    a1 = _checked(REPO / CS.RULE_AMENDMENT_FILE, "amendment_sha256")
    a2 = _checked(REPO / CS.RULE_AMENDMENT_A2_FILE, "amendment_sha256")
    assert a1["amends_rule_sha256"] == rule["rule_sha256"] and a2["amends_amendment_sha256"] == a1["amendment_sha256"]
    definition = json.loads((REPO / CS.DEFINITION_FILE).read_text())
    assert definition["rule_sha256"] == rule["rule_sha256"]
    assert definition["amendments"]["A2"]["sha256"] == a2["amendment_sha256"]
    frozen = {c["name"]: (c["tilt_deg"], c["azimuth_deg"]) for c in definition["heldout_v2"]["cameras"]}
    assert frozen == dict(zip(CS.HELDOUT_V2_NAMES, CS.HELDOUT_V2_CAMERAS))
    exceptions = {task: {vis: (e["tilt_deg"], e["azimuth_deg"]) for vis, e in per.items()}
                  for task, per in definition["heldout_v2"]["task_exceptions"].items()}
    assert exceptions == CS.HELDOUT_V2_TASK_EXCEPTIONS
    separation = rule["candidates"]["min_separation_from_training_deg"]
    pairwise = a2["selection"]["min_pairwise_separation_deg"]
    for i, cam in enumerate(CS.HELDOUT_V2_CAMERAS):
        assert CS.min_separation_deg(cam, train) >= separation
        assert CS.min_separation_deg(cam, CS.HELDOUT_V2_CAMERAS[:i]) >= pairwise
        path = HELDOUT_V2_PATHS[f"vis{i}"]
        assert len(path) == 1 and path[0] == orbit_camera(*cam) and path[0].distance == 1.0
    for task, per in CS.HELDOUT_V2_TASK_EXCEPTIONS.items():
        used = CS.heldout_v2_cameras(task)
        for vis, cam in per.items():
            assert used[vis] == cam and CS.min_separation_deg(cam, train) >= separation
            others = [c for name, c in used.items() if name != vis and name not in per]
            assert CS.min_separation_deg(cam, others) >= pairwise
        assert heldout_v2_paths(task)[next(iter(per))] == [orbit_camera(*next(iter(per.values())))]
    assert CS.heldout_v2_cameras("hammer") == dict(zip(CS.HELDOUT_V2_NAMES, CS.HELDOUT_V2_CAMERAS))
    levels = {k: {"kind": kind, **amps} for k, (kind, amps) in CS.STRONGER_TRAJECTORIES.items()}
    assert definition["moving_paths"]["levels"] == levels
    assert all(v["passed"] for v in definition["moving_paths"]["validation"].values())


def test_v2_groups_are_deterministic_with_their_own_paired_seeds():
    groups = evaluation_groups_v2(CFG)
    expected = ([f"ref_train{i}" for i in range(6)] + [f"vis{i}" for i in range(4)]
                + ["ref_traj_lateral", "ref_traj_circular"]
                + ["traj_lateral_2x", "traj_circular_2x", "traj_lateral_3x", "traj_circular_3x"])
    assert list(groups) == expected and all(n == 20 for _, n in groups.values())
    assert evaluation_groups_v2(CFG) == groups
    assert list(evaluation_groups_v2(CFG, reference="none")) == expected[6:10] + expected[12:]
    assert list(evaluation_groups_v2(CFG, reference="train")) == expected[:10] + expected[12:]
    assert not set(groups) & set(evaluation_groups(CFG)), "v2 groups never reuse a v1 group name"
    assert groups["ref_traj_lateral"][0] == trajectory_path("lateral")
    for task, per in CS.HELDOUT_V2_TASK_EXCEPTIONS.items():  # task exceptions keep the group names and counts
        with_task = evaluation_groups_v2(CFG, task=task)
        assert list(with_task) == expected and with_task != groups
        for vis in CS.HELDOUT_V2_NAMES:
            assert with_task[vis][0] == (heldout_v2_paths(task)[vis] if vis in per else groups[vis][0])
    assert groups["ref_train3"][0] == TRAIN_PATHS["train3"]
    with pytest.raises(ValueError):
        evaluation_groups_v2(CFG, reference="all")
    seeds = v2_episode_seeds(2000, 100, 20)
    assert seeds == v2_episode_seeds(2000, 100, 20) and seeds[:5] == v2_episode_seeds(2000, 100, 5)
    assert seeds != v2_episode_seeds(2000, 99, 20) and seeds != v2_episode_seeds(2001, 100, 20)
    assert not set(seeds) & set(episode_seeds(2000, 100, 240)), "separate stream from the v1 seeds"


def test_stronger_paths_have_the_registered_amplitudes_period_base_and_phase():
    base = TRAIN_PATHS["train1"][0]
    theta, phi = TRAIN_CAMERAS[1]
    centre = camera_c2w(base)[:3, 3]
    right = np.cross(_forward(base), [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    for name, (kind, amps) in CS.STRONGER_TRAJECTORIES.items():
        path = trajectory_path(name)
        assert len(path) == 72
        assert path == _original_trajectory_path(kind, **amps)
        if kind == "lateral":
            offsets = np.asarray([(camera_c2w(c)[:3, 3] - centre) @ right for c in path])
            assert offsets[0] == pytest.approx(0, abs=1e-9), "starts at the base camera, like the 1x path"
            assert offsets.max() == pytest.approx(amps["lateral_amplitude"], abs=0.01)
            assert offsets.min() == pytest.approx(-amps["lateral_amplitude"], abs=0.01)
            assert np.argmax(offsets) == 18 and np.argmin(offsets) == 53  # same phase as the reference path
        else:
            tilt = -_poses(path)[:, 1]
            az = _poses(path)[:, 0]
            assert az[0] == pytest.approx(phi + amps["circular_azimuth_deg"]) and tilt[0] == pytest.approx(theta)
            assert az.max() - az.min() == pytest.approx(2 * amps["circular_azimuth_deg"])
            assert tilt.max() - tilt.min() == pytest.approx(2 * amps["circular_elevation_deg"], rel=1e-3)
            assert tilt[18] == pytest.approx(theta + amps["circular_elevation_deg"])


@pytest.mark.parametrize("name", ["lateral", "circular", *CS.STRONGER_TRAJECTORIES])
def test_paths_are_continuous_across_the_loop_boundary(name):
    path = trajectory_path(name)
    c2w = np.stack([camera_c2w(p) for p in path])
    k = CS.path_kinematics(c2w, LOOKAT, action_repeat=2, sim_dt=0.0125, episode_agent_steps=125)
    assert k["loop_step_m"] <= k["max_step_within_period_m"] + 1e-9
    steps = np.linalg.norm(np.roll(c2w[:, :3, 3], -1, axis=0) - c2w[:, :3, 3], axis=1)
    assert steps.max() == pytest.approx(k["translation_m_per_agent_step"]["max"])
    assert k["period_agent_steps"] == 72 and k["period_sim_steps"] == 144
    # the env indexes a path by render count: render k (reset = 0) uses pose k mod 72, so a 125-step episode wraps once
    assert [(k_ % 72) for k_ in (0, 71, 72, 125)] == [0, 71, 0, 53]


def test_multiview_replacement_rule_for_v2_groups():
    single, multi = evaluation_groups_v2(CFG), evaluation_groups_v2(CFG, views=3)
    assert list(multi) == list(single) and all(multi[n][1] == single[n][1] for n in single)
    v1 = evaluation_groups(CFG, views=3)
    for i in range(6):
        assert multi[f"ref_train{i}"][0] == v1[f"train{i}"][0]
    for task in [None, *CS.HELDOUT_V2_TASK_EXCEPTIONS]:
        multi_task = evaluation_groups_v2(CFG, views=3, task=task)
        for name, path in heldout_v2_paths(task).items():
            n = nearest_training_camera(path)
            assert multi_task[name][0] == (path, *v1[f"train{n}"][0][1:]), "replaces its nearest training camera"
    for name in [f"traj_{k}" for k in CS.STRONGER_TRAJECTORIES] + ["ref_traj_lateral", "ref_traj_circular"]:
        assert multi[name][0] == (single[name][0], *v1["train1"][0][1:]), "trajectories replace train1"
    assert multiview_groups({"ref_traj_lateral": (trajectory_path("lateral"), 1)}, 3)["ref_traj_lateral"][0] == \
        v1["traj_lateral"][0]


def test_v2_writes_a_separate_output_and_reevaluates_final_and_last5():
    assert CS.SUITE_OUTPUTS == {"v1": "eval.jsonl", "v2": "eval_cameras_v2.jsonl"}
    assert CS.SUITE_PROVENANCE["v1"] == "eval_provenance" != CS.SUITE_PROVENANCE["v2"]
    steps = list(range(10_000, 1_000_001, 10_000))
    assert CS.reevaluation_steps(steps, 1_000_000, 10_000, "final") == [1_000_000]
    assert CS.reevaluation_steps(steps, 1_000_000, 10_000, "last5") == [960_000, 970_000, 980_000, 990_000, 1_000_000]
    with pytest.raises(ValueError, match="lacks snapshots"):
        CS.reevaluation_steps(steps[:-1], 1_000_000, 10_000, "final")


def test_eval_rl_v2_reevaluation_writes_only_its_own_file(tmp_path):
    """A tiny DrM + CNN run, then ``eval_rl.py --suites v2 --snapshots last5``: records keyed by v2 groups in
    eval_cameras_v2.jsonl; eval.jsonl is neither created nor touched; per-frame cameras of moving multi-view paths are
    checked in the env worker below."""
    tiny = ["train.num_seed_steps=20", "agent.num_expl_steps=20", "train.batch_size=16", "train.replay_size=500",
            "train.log_every_steps=10", "eval.every_steps=10", "train.checkpoint_every_steps=30",
            "eval.episodes_per_train_camera=1", "eval.episodes_per_heldout_camera=1", "eval.episodes_per_trajectory=1",
            "env.image_size=64", "env.max_episode_steps=20", "wandb.enabled=false"]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": os.environ["CUDA_VISIBLE_DEVICES"].split(",")[0], "MUJOCO_GL": "egl"}

    def run(script, *args):
        result = subprocess.run([sys.executable, "-I", str(REPO / script), *args], env=env, capture_output=True,
                                text=True, timeout=900)
        assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-4000:]

    run_dir = tmp_path / "v2_cli"
    run("scripts/train_rl.py", "--task", "hammer", "--encoder", "cnn", "--seed", "3", "--run-dir", str(run_dir),
        "--set", "train.num_train_steps=60", *tiny)
    run("scripts/eval_rl.py", "--run-dir", str(run_dir), "--suites", "v2", "--snapshots", "last5", "--workers", "1",
        "--envs-per-worker", "8")
    records = [json.loads(line) for line in (run_dir / "eval_cameras_v2.jsonl").read_text().splitlines()]
    assert [r["step"] for r in records] == [20, 30, 40, 50, 60] and not (run_dir / "eval.jsonl").exists()
    assert all(r["suite"] == CS.SUITE_V2 and r["episodes"] == 16 for r in records)
    assert all(f"{g}_success" in r for r in records for g in evaluation_groups_v2(CFG))
    assert not any(k.startswith(("eval0", "train0_", "traj_lateral_success")) for r in records for k in r)
    assert (run_dir / CS.SUITE_PROVENANCE["v2"] / "provenance.json").is_file()


def test_moving_path_cameras_are_supplied_per_frame():
    first_uuid = os.environ["CUDA_VISIBLE_DEVICES"].split(",")[0]
    result = subprocess.run([sys.executable, "-I", str(REPO / "tests/_camera_suites_worker.py")],
                            env={**os.environ, "CUDA_VISIBLE_DEVICES": first_uuid, "MUJOCO_GL": "egl"},
                            capture_output=True, text=True, timeout=900)
    assert result.returncode == 0, result.stderr[-3000:]
    out = json.loads(result.stdout.strip().splitlines()[-1])
    assert out["c2w_matches_path_per_frame"] and out["static_view_matches_rig"] and out["frames_checked"] >= 12
