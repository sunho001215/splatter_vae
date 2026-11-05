"""Controlled depth comparison: the multi-source consistency kernel and the five-variant driver's configs."""

import numpy as np
import pytest
import yaml


def _two_views():
    from s4d.realworld.synthetic import look_at_T_cam_base

    K = np.array([[80.0, 0, 40.0], [0, 80.0, 30.0], [0, 0, 1]])
    T = np.stack([look_at_T_cam_base(eye=[1.0, -0.15, 0.6], target=[0.0, 0.0, 0.0]),
                  look_at_T_cam_base(eye=[1.0, 0.15, 0.6], target=[0.0, 0.0, 0.0])])
    H, W = 60, 80
    from s4d.realworld.geometry import pixel_centers

    depth = []
    for v in range(2):  # ground plane z = 0 seen by each camera: exact depth by ray-plane intersection
        uv = pixel_centers(H, W).reshape(-1, 2)
        rays = np.c_[(uv[:, 0] - K[0, 2]) / K[0, 0], (uv[:, 1] - K[1, 2]) / K[1, 1], np.ones(len(uv))]
        R, t = T[v][:3, :3], T[v][:3, 3]
        c = -R.T @ t
        dirs = rays @ R
        lam = -c[2] / dirs[:, 2]
        depth.append(np.where(lam > 0, lam, 0).reshape(H, W))
    return K, T, np.stack(depth).astype(np.float32)


def test_multi_source_kernel_scores_sources_on_shared_points():
    from s4d.realworld.kernels import depth_compare_multi

    K, T, d = _two_views()
    S, V, H, W = 1, 2, *d.shape[1:]
    rng = np.random.default_rng(0)
    noisy = d + rng.normal(0, 0.01, d.shape).astype(np.float32) * (d > 0)
    holes = d.copy()
    holes[:, :, :20] = 0  # A has a hole that B fills
    arrays = {"images": np.zeros((S, V, H, W, 3), np.uint8), "K": np.stack([K, K]), "T_cam_base": T,
              "surface_base": np.zeros((S, V, 4, 3), np.float32) + 50.0, "gripper_points": np.zeros(4, bool),
              "calib_used": np.zeros((S, V), bool)}
    for name, dd in (("A", holes), ("B", noisy), ("C", d)):
        arrays[f"depth_{name}"] = dd[None]
        arrays[f"valid_{name}"] = (dd > 0)[None]
    out, meta = depth_compare_multi({"sources": ["A", "B", "C"], "pairs": [["A", "B"], ["A", "C"]],
                                     "reference_source": "A", "max_points": 3000}, arrays)
    pair = meta["sets"][0]["pairs"]["0->1"]
    ab = pair["A|B|scene"]
    assert ab["kept"] > 500 and ab["A"]["n"] == ab["B"]["n"] == ab["kept"]  # same points for both sources
    # the exact source is limited by nearest-pixel lookup on this coarse grazing plane; 1 cm noise adds clearly more
    assert ab["B"]["median_abs"] > ab["A"]["median_abs"] + 0.002
    assert 0 < pair["joint|scene"]["kept"] <= 3000  # supports are capped at max_points by random subsampling
    assert pair["filled|C"]["points"] > 0
    # exact depth in the hole region (image border, grazing plane): only quantisation-level disagreement remains
    assert pair["filled|C"]["vs_other_reference"]["median_abs"] < 0.03
    cov = meta["sets"][0]["views"][0]["coverage"]["scene"]
    assert cov["A"] < cov["C"] and cov["joint"] <= cov["A"]
    assert out["grid_A"].shape[:2] == (1, 2) and out["grid_static"].dtype == bool


def test_variant_configs_share_the_frozen_base_and_differ_only_in_depth():
    from ops.inference_server.run_depth_variants import VARIANTS, variant_config

    base = yaml.safe_load(open("configs/realworld/mug_hanging_ep1.yaml"))
    cfgs = {n: variant_config(base, n, __import__("pathlib").Path(f"/tmp/x/{n}")) for n in VARIANTS}
    assert {n: c["stages"]["lift3d"]["depth_stage"] for n, c in cfgs.items()} == {
        "B0": "sensor_depth", "B1": "foundation_stereo_depth", "E1": "mapanything_depth", "E2": "mapanything_depth",
        "E3": "mapanything_depth"}
    assert {n: c["stages"].get("mapanything_depth", {}).get("input_depth_stage") for n, c in cfgs.items()} == {
        "B0": None, "B1": None, "E1": "sensor_depth", "E2": "foundation_stereo_depth",
        "E3": "foundation_stereo_filtered"}
    ma = [c["stages"]["mapanything_depth"] for n, c in cfgs.items() if n.startswith("E")]
    strip = [{k: v for k, v in m.items() if k != "input_depth_stage"} for m in ma]
    assert strip[0] == strip[1] == strip[2]  # identical MapAnything settings
    assert all(c["workers"]["mapanything"] == base["workers"]["mapanything"] for c in cfgs.values())
    assert all(c["stages"]["calibrate"] == base["stages"]["calibrate"] for c in cfgs.values())
    assert all(c["adapter"] == base["adapter"] for c in cfgs.values())


def test_variant_roots_refuse_an_incomplete_base(tmp_path):
    from ops.inference_server.run_depth_variants import link_shared

    with pytest.raises(SystemExit, match="lacks complete shared stages"):
        link_shared(tmp_path / "base", tmp_path / "B0", "ep")


def test_trackeverything_roots_use_the_variant_depth_and_never_touch_the_base_tracks():
    from pathlib import Path

    from ops.inference_server.run_depth_variants import SHARED, shared_stages, variant_config
    from s4d.realworld.cli import default_stages

    base = yaml.safe_load(open("configs/realworld/mug_hanging_ep1.yaml"))
    assert "track_everything" not in default_stages(base)  # a default base run never replaces AllTracker tracks
    for name, depth in (("B0", "sensor_depth"), ("E2", "mapanything_depth")):
        cfg = variant_config(base, name, Path(f"/tmp/x/{name}-TE"), "trackeverything")
        te = cfg["stages"]["track_everything"]
        assert te["depth_stage"] == depth and te["grid_stride"] == base["stages"]["track2d"]["grid_stride"]
        assert te["window"] == base["stages"]["track2d"]["window"] and te["step"] == base["stages"]["track2d"]["step"]
    assert "track2d" not in shared_stages("trackeverything") and set(shared_stages("alltracker")) == set(SHARED)


@pytest.mark.parametrize("force", [False, True])
@pytest.mark.parametrize("complete", [False, True])
def test_shared_stage_symlink_refuses_build_even_under_force(tmp_path, force, complete):
    from s4d.realworld import io as rio

    base, shared = tmp_path / "base", tmp_path / "shared"
    opts = dict(stage="calibrate", version="test", config={}, inputs={}, build=lambda _: {})
    if complete:
        rio.run_stage(base, **opts)
    shared.symlink_to(base)
    with pytest.raises(RuntimeError, match="refusing to build shared stage calibrate.*symlink"):
        rio.run_stage(shared, force=force, **opts)
    assert shared.is_symlink() and base.exists() == complete


@pytest.mark.parametrize("tracker", ["alltracker", "trackeverything"])
def test_variant_cli_excludes_and_refuses_frozen_stages(tracker):
    from pathlib import Path

    from ops.inference_server.run_depth_variants import variant_config
    from s4d.realworld.cli import adapter_config, check_stage_scope, default_stages

    base = yaml.safe_load(open("configs/realworld/mug_hanging_ep1.yaml"))
    cfg = variant_config(base, "E2", Path(f"/tmp/x/E2-{tracker}"), tracker)
    frozen = cfg["depth_variant"]["shared_stages"] + ["ingest"]
    assert not set(frozen) & set(default_stages(cfg))
    check_stage_scope(cfg, default_stages(cfg))
    for stage in frozen:
        with pytest.raises(ValueError, match="shared/frozen.*base root"):
            check_stage_scope(cfg, [stage])
    assert adapter_config(cfg) == adapter_config(base)  # retain calibration settings needed by readers


def _record(root, stage, inputs=None, force=False):
    from s4d.realworld import io as rio

    return rio.run_stage(rio.stage_dir(root, "ep", stage), stage=stage, version="test", config={},
                         inputs=inputs or {}, build=lambda _: {}, force=force)


def test_comparison_rejects_stale_depth_even_when_calibration_matrices_match(tmp_path):
    from ops.inference_server.depth_compare_variants import verify_calibration_inputs

    base, variant = tmp_path / "base", tmp_path / "E1"
    cal = _record(base, "calibrate")["build_id"]
    depth = _record(variant, "mapanything_depth", {"calibrate": cal})["build_id"]
    _record(variant, "lift3d", {"calibrate": cal, "mapanything_depth": depth})
    assert verify_calibration_inputs(base, "ep", {"E1": variant}, {"E1": "mapanything_depth"}) == cal
    # An identical calibration config/summary still has a new build id after a forced rebuild.
    _record(base, "calibrate", force=True)
    with pytest.raises(SystemExit, match="stale mapanything_depth.*calibrate input"):
        verify_calibration_inputs(base, "ep", {"E1": variant}, {"E1": "mapanything_depth"})


def test_comparison_requires_each_variants_lift_calibration_input(tmp_path):
    from ops.inference_server.depth_compare_variants import verify_calibration_inputs

    base, variant = tmp_path / "base", tmp_path / "E2"
    cal = _record(base, "calibrate")["build_id"]
    _record(variant, "mapanything_depth", {"calibrate": cal})
    _record(variant, "lift3d")
    with pytest.raises(SystemExit, match="stale lift3d.*missing base calibrate"):
        verify_calibration_inputs(base, "ep", {"E2": variant}, {"E2": "mapanything_depth"})


def test_sensor_and_fs_depth_can_be_calibration_independent_but_lift_cannot(tmp_path):
    from ops.inference_server.depth_compare_variants import verify_calibration_inputs

    base = tmp_path / "base"
    cal = _record(base, "calibrate")["build_id"]
    for name, stage in (("B0", "sensor_depth"), ("B1", "foundation_stereo_depth")):
        _record(base, stage)
        _record(tmp_path / name, "lift3d", {"calibrate": cal})
        assert verify_calibration_inputs(base, "ep", {name: base}, {name: stage}) == cal
        _record(tmp_path / name, "lift3d", {"calibrate": "old"}, force=True)
        with pytest.raises(SystemExit, match="stale lift3d.*calibrate input"):
            verify_calibration_inputs(base, "ep", {name: base}, {name: stage})


def test_filtered_depth_checks_transitive_calibration_and_input_builds(tmp_path):
    from ops.inference_server.depth_compare_variants import verify_calibration_inputs

    base = tmp_path / "base"
    cal = _record(base, "calibrate")["build_id"]
    fs = _record(base, "foundation_stereo_depth", {"calibrate": "old"})["build_id"]
    _record(base, "foundation_stereo_filtered", {"foundation_stereo_depth": fs})
    with pytest.raises(SystemExit, match="stale foundation_stereo_depth.*calibrate input"):
        verify_calibration_inputs(base, "ep", {"B1f": base}, {"B1f": "foundation_stereo_filtered"})
    _record(base, "foundation_stereo_depth", {"calibrate": cal}, force=True)
    with pytest.raises(SystemExit, match="stale foundation_stereo_filtered.*input foundation_stereo_depth"):
        verify_calibration_inputs(base, "ep", {"B1f": base}, {"B1f": "foundation_stereo_filtered"})

