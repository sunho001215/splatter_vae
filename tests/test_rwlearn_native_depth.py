"""CPU-only native depth overlays: tiny recorded H5s, no GPU, released weights or network."""
from __future__ import annotations

import json
import os
import pickle
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch
import yaml

from s4d.realworld import train_store as TS
from s4d.realworld.calib.rig import FORMAT as RIG_FORMAT
from s4d.realworld.calib.rig import publish_rig
from s4d.rwlearn import actions as A
from s4d.rwlearn import fql_data as FD
from s4d.rwlearn.deploy import contracts as C
from s4d.rwlearn.deploy.history import HistoryWindow
from s4d.rwlearn.deploy.realsense import load_rig_session
from s4d.rwlearn.deploy.replay import RecordedEpisode
from s4d.rwlearn.native_depth import build_native_depth, sha256_file
from s4d.rwlearn.store import open_store
from tests.test_robot_pc import write_recording


def make_store(root: Path, count=1):
    root.mkdir(parents=True)
    paths = []
    for i in range(count):
        eid = f"episode_{i:06d}"
        raw = root / "drawer-open" / f"{eid}.h5"
        raw.parent.mkdir(exist_ok=True)
        write_recording(raw, frames=4, seed=i)
        rec = RecordedEpisode(raw)
        path = root / f"{eid}.compact.h5"
        cams = rec.cameras()
        with h5py.File(path, "w") as f:
            f.attrs.update(format=TS.FORMAT, schema_version=TS.SCHEMA_VERSION, episode_id=eid, task="drawer-open",
                           cameras=json.dumps(cams), reference_camera=cams[0], image_size=128, max_sync_ns=30_000_000,
                           raw_recording=str(raw), raw_sha256=sha256_file(raw),
                           depth_stage="foundation_stereo_depth", upstream=json.dumps({"calibrate": "fixture"}))
            for c in cams:
                g = f.create_group(f"cameras/{c}")
                g["timestamps_ns"] = rec.timestamps_ns(c)
                g["segment"] = np.zeros(4, np.int16)
                g["rgb"] = np.stack([rec.frame(c, i, depth=False).rgb for i in range(4)])
                g["depth"] = np.full((4, 128, 128), 1.234567, np.float32)  # FS sentinel, never native
                g["semantic"] = np.ones((4, 128, 128), np.uint8)
                g["K"] = np.eye(3)
                g["w2c"] = np.eye(4)
            f["sync/frames"] = np.tile(np.arange(4)[:, None], (1, len(cams)))
            f["sync/surface_disp_m"] = np.zeros(4)
        paths.append(str(path))
    rig = root / "rig.json"
    publish_rig(rig, {"format": RIG_FORMAT, "name": "native-fixture", "cameras": {
        f"{c}/color": {"status": "validated", "T_cam_base": np.eye(4).tolist(),
                       "depth_scale": {"sensor_depth": factor, "foundation_stereo_depth": 1.08}}
        for c, factor in zip(cams, (0.997, 1.003, 1.017))}}, {
            f"{c}__color__{name}": np.zeros((2, 2), dtype)
            for c in cams for name, dtype in (("grey", np.float32), ("dynamic", bool))})
    locator = root / "store.yaml"
    locator.write_text(yaml.safe_dump({"kind": "compact", "paths": paths, "fixture_note": "preserved"}))
    return SimpleNamespace(root=root, locator=locator, rig=rig, paths=paths,
                           out=root / "store-native.yaml", overlays=root / "overlays")


def build(fixture, **kwargs):
    return build_native_depth(fixture.locator, fixture.rig, fixture.overlays, fixture.out, **kwargs)


def dataset_attrs(store, condition="mvp3d_single"):
    rec = RecordedEpisode(store.raw_recording(store.episodes()[0]))
    return {"condition": condition, "calibration": json.dumps({ep: {"cameras": store.cameras(ep)}
                                                            for ep in store.episodes()}),
            "frame_stack": 3, "contract": json.dumps(A.contract_for(rec.raw).to_json()),
            "export_sha256": "", "image_hw": "[128, 128]"}


@pytest.mark.parametrize("condition", ["mvp3d_single", "mvp3d_multiview"])
def test_native_depth_matches_replay_history_and_pretraining(tmp_path, condition):
    fixture = make_store(tmp_path / "store")
    inputs = [fixture.locator, *map(Path, fixture.paths)]
    before = [p.read_bytes() for p in inputs]
    assert build(fixture)["built"] == 1
    native, ordinary = open_store(fixture.out), open_store(fixture.locator)
    assert native.depth_source == C.NATIVE_DEPTH and not hasattr(ordinary, "depth_source")
    ep, = native.episodes()
    rec = RecordedEpisode(native.raw_recording(ep))
    cams = native.cameras(ep)
    session = load_rig_session(fixture.rig.resolve(), cams, {c: rec.K_native(c) for c in cams})
    frames = {c: [rec.frame(c, i, depth=True, correction=session["native_depth_correction"][c],
                           alignment="rs_align") for i in range(4)] for c in cams}
    for c in cams:
        expected = np.stack([f.depth for f in frames[c]])
        actual = native.depth(ep, c, np.arange(4))
        assert actual.dtype == np.float32 and actual.tobytes() == expected.tobytes()
        assert (actual == 0).any() and not np.array_equal(actual, ordinary.depth(ep, c, np.arange(4)))
        assert native.depth(ep, c, [3, 0, 3]).tobytes() == expected[[3, 0, 3]].tobytes()
        assert native.depth(ep, c, []).shape == (0, 128, 128)
        for method in ("rgb", "semantic"):
            np.testing.assert_array_equal(getattr(native, method)(ep, c, [0, 2]),
                                          getattr(ordinary, method)(ep, c, [0, 2]))
        for method in ("intrinsics", "w2c", "timestamps", "segments"):
            np.testing.assert_array_equal(getattr(native, method)(ep, c), getattr(ordinary, method)(ep, c))
    attrs = {**dataset_attrs(native, condition), "training_depth": C.NATIVE_DEPTH,
             "native_depth_provenance": json.dumps(native.depth_provenance([ep]))}
    spec = C.InputSpec.from_meta({"condition": condition, "image_hw": [128, 128],
                                 "contract": json.loads(attrs["contract"])}, C.training_contract(attrs))
    selected = cams[:1] if spec.views == 1 else cams
    spec = spec.bind(session_id="native-equality", camera_id=selected[0] if spec.views == 1 else None,
                     native_hw={c: rec.native_hw(c) for c in selected},
                     calibration={c: session["calibration"][c] for c in selected},
                     native_depth_correction={c: session["native_depth_correction"][c] for c in selected})
    history = HistoryWindow(spec)
    for i in range(4):
        for c in selected:
            history.push(c, frames[c][i])
        window = history.assemble({c: frames[c][i].seq for c in selected}, pad_run_start=i < 2,
                                  surface_disp_m=[0.0] * 3 if spec.views > 1 else None)
        indices = [max(0, i - 2), max(0, i - 1), i]
        expected = np.stack([native.depth(ep, c, indices) for c in selected], 1)
        assert window.observation()["depth"].tobytes() == expected.tobytes()
    # The pretraining adapter reads the overlay through the existing FrameStore interface.
    from s4d.rwlearn.baseline_data import RealworldMVPTimeSteps

    data = RealworldMVPTimeSteps(fixture.out, [ep], num_views=3)
    np.testing.assert_array_equal(data[2]["depth"].numpy(), np.stack([native.depth(ep, c, [2])[0] for c in cams]))
    restored = pickle.loads(pickle.dumps(native))
    assert not restored._overlay_files
    assert restored.depth(ep, cams[0], [0]).tobytes() == native.depth(ep, cams[0], [0]).tobytes()
    restored.close()
    ordinary.close()
    native.close()
    assert [p.read_bytes() for p in inputs] == before


@pytest.mark.parametrize("case,message", [("raw_sha256", "raw_sha256"), ("count", "frame count"),
                                         ("timestamp", "timestamps"), ("missing", "missing native depth"),
                                         ("rig", "rig_document_sha256"), ("correction", "correction")])
def test_overlay_identity_refusals(tmp_path, case, message):
    fixture = make_store(tmp_path / "store")
    build(fixture)
    spec = yaml.safe_load(fixture.out.read_text())
    path = Path(spec["depth_overlays"][0])
    if case == "missing":
        path.unlink()
    else:
        with h5py.File(path, "r+") as f:
            cam = sorted(f["cameras"])[0]
            if case == "raw_sha256":
                f.attrs[case] = "0" * 64
            elif case == "rig":
                f.attrs["rig_document_sha256"] = "0" * 64
            elif case == "correction":
                corr = json.loads(f.attrs["native_depth_correction"])
                corr[cam]["evidence"]["rig_calibration"] = "0" * 64
                f.attrs["native_depth_correction"] = json.dumps(corr)
            elif case == "count":
                del f[f"cameras/{cam}/depth"]
                f[f"cameras/{cam}/depth"] = np.zeros((3, 128, 128), np.float32)
            else:
                f[f"cameras/{cam}/timestamps_ns"][1] += 1
    with pytest.raises(ValueError, match=message):
        store = open_store(fixture.out)
        try:
            ep = store.episodes()[0]
            store.depth(ep, store.cameras(ep)[0], [0])
        finally:
            store.close()
    if case == "missing":
        assert build(fixture)["built"] == 1
    else:
        with pytest.raises(ValueError, match=message):
            build(fixture)  # an existing invalid overlay never skips


@pytest.mark.parametrize("reader", ["depth", "depth_provenance"])
def test_overlay_replacement_is_reverified(tmp_path, reader):
    fixture = make_store(tmp_path / "store")
    build(fixture)
    store = open_store(fixture.out)
    ep, = store.episodes()
    try:
        store.depth(ep, store.cameras(ep)[0], [0])
        original = store._overlay_files[ep][1]
        path = Path(store.depth_overlays[ep])
        replacement = path.with_suffix(".replacement.h5")
        shutil.copyfile(path, replacement)
        with h5py.File(replacement, "r+") as f:
            f.attrs["raw_sha256"] = "0" * 64
        replacement.replace(path)
        with pytest.raises(ValueError, match="raw_sha256"):
            if reader == "depth":
                store.depth(ep, store.cameras(ep)[0], [0])
            else:
                store.depth_provenance([ep])
        assert not original.id.valid and ep not in store._overlay_files
    finally:
        store.close()


def test_overlay_cache_is_bounded(tmp_path):
    fixture = make_store(tmp_path / "store", count=3)
    build(fixture)
    store = open_store(fixture.out)
    store.max_open = 2
    eps = store.episodes()
    cam = store.cameras(eps[0])[0]
    expected = {}
    for ep in eps:
        with h5py.File(store.depth_overlays[ep], "r") as f:
            expected[ep] = f[f"cameras/{cam}/depth"][:][[3, 0, 3]].tobytes()
    handles = []
    try:
        for i in (0, 1, 0, 2, 1):
            ep = eps[i]
            assert store.depth(ep, cam, [3, 0, 3]).tobytes() == expected[ep]
            handles.append(store._overlay_files[ep][1])
            assert len(store._overlay_files) <= store.max_open
        assert handles[0] is handles[2]  # cache hits reuse the open handle and refresh its LRU position
        assert not handles[1].id.valid and not handles[0].id.valid
        assert handles[4] is not handles[1]
        assert list(store._overlay_files) == [eps[2], eps[1]]
        assert handles[3].id.valid and handles[4].id.valid
    finally:
        store.close()
    assert not store._overlay_files and all(not f.id.valid for f in handles)


def test_overlay_cache_pickle_pid_and_close(tmp_path, monkeypatch):
    from s4d.rwlearn import native_depth as ND

    fixture = make_store(tmp_path / "store")
    build(fixture)
    store = open_store(fixture.out)
    ep, = store.episodes()
    cam = store.cameras(ep)[0]
    inherited = []
    monkeypatch.setattr(ND, "_INHERITED_HANDLES", inherited)
    try:
        expected = store.depth(ep, cam, [3, 0, 3]).tobytes()
        original = store._overlay_files[ep][1]
        restored = pickle.loads(pickle.dumps(store))
        try:
            assert not restored._overlay_files and restored._overlay_pid is None
            for _ in range(2):
                assert restored.depth(ep, cam, [3, 0, 3]).tobytes() == expected
        finally:
            restored.close()
        assert store._overlay_files[ep][1] is original and original.id.valid
        # Simulate a worker inheriting a populated cache from a different PID.
        store._overlay_pid = -1
        assert store.depth(ep, cam, [3, 0, 3]).tobytes() == expected
        assert inherited == [original] and original.id.valid
        assert store._overlay_files[ep][1] is not original and store._overlay_pid == os.getpid()
    finally:
        store.close()
        for f in inherited:
            f.close()  # simulated inherited handles still belong to this test process
    assert not store._overlay_files and store._overlay_pid is None
    with h5py.File(store.depth_overlays[ep], "r+") as f:
        f.attrs["raw_sha256"] = "0" * 64
    try:
        with pytest.raises(ValueError, match="raw_sha256"):
            store.depth(ep, cam, [0])  # close() also invalidates verification
    finally:
        store.close()


def test_native_build_resume_subset_and_parallelism(tmp_path):
    environment = dict(os.environ)
    fixture = make_store(tmp_path / "store", count=2)
    input_spec = yaml.safe_load(fixture.locator.read_text())
    input_spec["paths"].reverse()  # order differs from episodes(), which sorts
    fixture.locator.write_text(yaml.safe_dump(input_spec))
    assert build(fixture, episodes=["drawer-open/episode_000001"]) == {
        "built": 1, "skipped": 0, "descriptor_written": False, "out_store": str(fixture.out)}
    assert not fixture.out.exists()
    existing, = fixture.overlays.glob("*.h5")
    before = (existing.read_bytes(), existing.stat().st_mtime_ns)
    assert build(fixture, workers=2)["built"] == 1
    assert (existing.read_bytes(), existing.stat().st_mtime_ns) == before
    assert build(fixture, workers=2)["skipped"] == 2
    spec = yaml.safe_load(fixture.out.read_text())
    assert spec == {**yaml.safe_load(fixture.locator.read_text()), "depth_source": C.NATIVE_DEPTH,
                    "depth_overlays": spec["depth_overlays"], "rig_document_sha256": sha256_file(fixture.rig)}
    assert not list(fixture.overlays.glob("*.partial"))
    # Overlay order follows the descriptor even though episodes() sorts its keys.
    native = open_store(fixture.out)
    for ep, path in native.depth_overlays.items():
        with h5py.File(path) as f:
            assert f.attrs["episode_id"] == ep
    native.close()
    assert dict(os.environ) == environment


@pytest.mark.parametrize("case", ["raw_hash", "raw_changed", "count", "timestamp"])
def test_builder_refuses_raw_or_store_identity_changes(tmp_path, case):
    fixture = make_store(tmp_path / "store")
    build(fixture)
    with h5py.File(fixture.paths[0], "r+") as f:
        if case == "raw_hash":
            f.attrs["raw_sha256"] = "0" * 64
        elif case == "raw_changed":
            with h5py.File(f.attrs["raw_recording"], "r+") as raw:
                raw.attrs["modified"] = True
        else:
            ds = f["cameras/cam_a/timestamps_ns"]
            if case == "timestamp":
                ds[1] += 1
            else:
                del f["cameras/cam_a/timestamps_ns"]
                f["cameras/cam_a/timestamps_ns"] = np.zeros(3, np.int64)
    with pytest.raises(ValueError, match="raw_sha256|frame count/timestamps"):
        build(fixture)  # recompute the raw hash even when an overlay already exists


def test_native_build_atomic_failure_and_input_protection(tmp_path, monkeypatch):
    from s4d.rwlearn import native_depth as ND

    fixture = make_store(tmp_path / "store")
    spec = yaml.safe_load(fixture.locator.read_text())
    overlay = fixture.overlays / "failure.h5"
    fixture.overlays.mkdir()
    original = RecordedEpisode.frame

    def fail(self, cam, index, **kwargs):
        if index == 1:
            raise RuntimeError("injected frame failure")
        return original(self, cam, index, **kwargs)

    monkeypatch.setattr(RecordedEpisode, "frame", fail)
    with pytest.raises(RuntimeError, match="injected"):
        ND._build_episode((spec["paths"][0], overlay, str(fixture.rig.resolve()), sha256_file(fixture.rig)))
    assert not overlay.exists() and not list(fixture.overlays.iterdir()) and not fixture.out.exists()
    for protected in (fixture.locator, fixture.rig, Path(fixture.paths[0])):
        before = protected.read_bytes()
        with pytest.raises(ValueError, match="overwrite"):
            build_native_depth(fixture.locator, fixture.rig, fixture.overlays, protected)
        assert protected.read_bytes() == before


def test_native_depth_cli_build(tmp_path):
    fixture = make_store(tmp_path / "store")
    script = Path(__file__).resolve().parents[1] / "scripts/realworld_native_depth.py"
    result = subprocess.run([sys.executable, "-I", str(script), "build", "--store", str(fixture.locator),
                             "--rig", str(fixture.rig), "--out-dir", str(fixture.overlays),
                             "--out-store", str(fixture.out), "--workers", "2", "--episodes",
                             "drawer-open/episode_000000"], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    payload = "\n".join(line for line in result.stdout.splitlines() if not line.startswith("[gpu_guard]"))
    assert result.stdout.startswith("[gpu_guard]")  # the scheduler-required guard ran first
    assert json.loads(payload) == {"built": 1, "skipped": 0, "descriptor_written": True,
                                        "out_store": str(fixture.out)}
    store = open_store(fixture.out)
    assert store.depth_source == C.NATIVE_DEPTH
    store.close()


def test_native_fql_provenance_and_default_attributes(tmp_path, monkeypatch):
    fixture = make_store(tmp_path / "store")
    build(fixture)
    ordinary, native = open_store(fixture.locator), open_store(fixture.out)
    ep = ordinary.episodes()[0]
    # Tiny adapter keeps the real depth-reading encode_timeline path, without a renderer or export dependency.
    class MeanDepth(torch.nn.Module):
        def forward(self, obs):
            return obs["depth"].mean((-1, -2))

    adapter = torch.nn.Module()
    adapter.backbone = MeanDepth()
    adapter.stack_feature = False
    adapter.observation = {"depth": True, "cameras": True}
    monkeypatch.setattr(FD, "encoder_adapter", lambda *args: adapter)
    export = fixture.root / "tiny.pt"
    export.write_bytes(b"tiny adapter fixture")
    for condition in ("mvp3d_single", "mvp3d_multiview", "e2e_cnn"):
        plain = fixture.root / f"{condition}-plain.h5"
        FD.build_fql_dataset(ordinary, condition, None if condition == "e2e_cnn" else export, plain,
                             episodes=[ep], assume_final_success=True)
        with h5py.File(plain) as f:
            assert "training_depth" not in f.attrs and "native_depth_provenance" not in f.attrs
        if condition == "e2e_cnn":
            continue
        out = fixture.root / f"{condition}-native.h5"
        FD.build_fql_dataset(native, condition, export, out, episodes=[ep], assume_final_success=True)
        with h5py.File(out) as f:
            assert f.attrs["training_depth"] == C.NATIVE_DEPTH
            provenance = json.loads(f.attrs["native_depth_provenance"])
            assert provenance == native.depth_provenance([ep])
            assert C.training_contract(dict(f.attrs))["native_depth_provenance"] == provenance
            meta = {"condition": condition, "export_sha256": f.attrs["export_sha256"],
                    "contract": json.loads(f.attrs["contract"]), "image_hw": [128, 128],
                    "training_depth": C.NATIVE_DEPTH}
        assert C.read_training_contract(meta, out)["training_depth"] == C.NATIVE_DEPTH
        with pytest.raises(ValueError, match="training_depth"):
            C.read_training_contract({**meta, "training_depth": "foundation_stereo_depth"}, out)
    for condition, c in FD.CONDITIONS.items():
        if c.depth:
            continue
        with pytest.raises(ValueError, match="only supported for 3D-MVP"):
            FD.build_fql_dataset(native, condition, None, fixture.root / "refused.h5", episodes=[ep])
    from s4d.rwlearn.baseline_data import realworld_classes

    for consumer in ("dynamo", "sincro", "reviwo", "snerl"):
        with pytest.raises(ValueError, match="only supported for mvp3d"):
            realworld_classes({"source": "realworld"}, 1, native, consumer=consumer)
    classes = realworld_classes({"source": "realworld", "smoke_without_calibration": True}, 1,
                                native, consumer="mvp3d")
    assert "mvp3d" in classes
    from s4d.rwlearn.baseline_data import check_gate

    assert check_gate(native, None, steps=1, consumer="mvp3d_single", smoke_without_calibration=True)["smoke"]
    ordinary.close()
    native.close()


def test_training_depth_contract_and_checkpoint_compatibility(tmp_path):
    from s4d.rwlearn.fql import FQLAgent
    from s4d.rwlearn.fql_data import ObsEncoder
    from s4d.rwlearn.inference import checkpoint_payload

    fixture = make_store(tmp_path / "store")
    store = open_store(fixture.locator)
    attrs = dataset_attrs(store)
    fs = C.training_contract(attrs)
    assert fs["training_depth"] == "foundation_stereo_depth"
    assert "FoundationStereo" in " ".join(C._differences("mvp3d_single", True, True))
    native = C.training_contract({**attrs, "training_depth": C.NATIVE_DEPTH})
    assert native["training_depth"] == C.NATIVE_DEPTH and "FoundationStereo" not in native["depth_source_evidence"]
    agent = FQLAgent.create(1, 18, 7, {"actor_hidden_dims": (8,), "value_hidden_dims": (8,)},
                            encoder=lambda: ObsEncoder(2, False, proj=8))
    data = SimpleNamespace(attrs={**attrs, "export_path": "", "reward": "ogbench_final_success"},
                           feat_dim=2, sequence=False, path="fixture.h5")
    payload = checkpoint_payload(agent, data, "mvp3d_single", 1)
    assert "training_depth" not in payload["meta"]
    assert C.read_training_contract(payload["meta"], fs) == fs
    data.attrs["training_depth"] = C.NATIVE_DEPTH
    meta = checkpoint_payload(agent, data, "mvp3d_single", 1)["meta"]
    assert meta["training_depth"] == C.NATIVE_DEPTH
    sidecar = fixture.root / "sidecar.json"
    sidecar.write_text(json.dumps(native))
    assert C.read_training_contract(meta, sidecar) == native
    explicit_fs = {**meta, "training_depth": "foundation_stereo_depth"}
    legacy = {k: v for k, v in fs.items() if k != "training_depth"}
    for checkpoint in (payload["meta"], explicit_fs, meta):
        for training in (legacy, fs, native):
            accepted = (training.get("training_depth") == checkpoint["training_depth"]
                        if "training_depth" in checkpoint else training.get("training_depth") != C.NATIVE_DEPTH)
            sidecar.write_text(json.dumps(training))
            # Both mismatch directions, including missing depth metadata, for both source paths and fixtures.
            for ckpt, source in ((checkpoint, training), (checkpoint, sidecar),
                                 ({**checkpoint, "fixture": True, "observation_contract": training}, None)):
                if accepted:
                    assert C.read_training_contract(ckpt, source) == training
                else:
                    with pytest.raises(ValueError, match="training_depth"):
                        C.read_training_contract(ckpt, source)
    spec = C.InputSpec.from_meta(meta, native)
    assert all("FoundationStereo" not in text for text in spec.train_deploy_differences)
    # Existing frame validation also refuses FS-labelled frames for a native training spec.
    from s4d.rwlearn.deploy.fixtures import session_choices, synthetic_frames
    from s4d.rwlearn.deploy.history import validate_frame

    choices = session_choices(SimpleNamespace(input_spec=spec), "native")
    spec = spec.bind(**choices)
    frame = synthetic_frames(spec)[spec.camera_ids[0]][0]
    validate_frame(frame, spec)
    wrong = replace(frame, metadata={**frame.metadata, "depth_source": "foundation_stereo_depth"})
    with pytest.raises(ValueError, match="deployment depth"):
        validate_frame(wrong, spec)
    store.close()


def test_native_depth_script_passes_the_scheduler_gpu_guard_check():
    """The scheduler refuses (raises in its tick) for a job script without a GPU guard call (2026-11-10 incident)."""
    from ops.jobs.scheduler import check_guarded

    assert check_guarded("scripts/realworld_native_depth.py").name == "realworld_native_depth.py"


# ---- raw-root-mapped build (local placement of the 3D-MVP chain, agent PNP 2026-11-13; s4d/rwlearn/native_depth_mapped)
def _overlay_arrays(out_store):
    native = open_store(out_store)
    try:
        out = {}
        for ep, path in native.depth_overlays.items():
            with h5py.File(path) as f:
                attrs = {k: v for k, v in f.attrs.items() if k != "raw_recording"}
                corr = json.loads(attrs.pop("native_depth_correction"))
                for c in corr.values():  # the evidence names the rig document's own path (separate fixture roots)
                    c["evidence"].pop("document", None)
                attrs["native_depth_correction"] = corr
                out[ep] = (attrs, {c: f[f"cameras/{c}/depth"][:] for c in f["cameras"]})
        return out
    finally:
        native.close()


def _moved_raw(fixture, tmp_path):
    """Copy the recordings to a separate local root and remove them from the recorded location."""
    local = tmp_path / "local-raw"
    for raw in sorted((fixture.root / "drawer-open").glob("*.h5")):
        target = local / "drawer-open" / raw.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(raw, target)
        raw.rename(raw.with_suffix(".moved"))
    return local


def test_mapped_build_equals_the_unmapped_build(tmp_path):
    from s4d.rwlearn import native_depth as ND
    from s4d.rwlearn.native_depth_mapped import build_native_depth_mapped

    environment = dict(os.environ)
    reference = make_store(tmp_path / "ref", count=2)
    build(reference, workers=2)
    fixture = make_store(tmp_path / "store", count=2)
    local = _moved_raw(fixture, tmp_path)
    with pytest.raises((OSError, ValueError)):  # the recorded raw paths are gone: the unmapped build cannot read them
        build(fixture)
    shutil.rmtree(fixture.overlays, ignore_errors=True)
    result = build_native_depth_mapped(fixture.locator, fixture.rig, fixture.overlays, fixture.out,
                                       raw_root=f"{fixture.root}={local}", workers=2)
    assert result == {"built": 2, "skipped": 0, "descriptor_written": True, "out_store": str(fixture.out)}
    got, want = _overlay_arrays(fixture.out), _overlay_arrays(reference.out)
    assert got.keys() == want.keys()
    for ep in got:
        assert got[ep][0] == want[ep][0]  # identical identity attrs (raw_sha256, rig, corrections, counts)
        for cam in want[ep][1]:
            assert np.array_equal(got[ep][1][cam], want[ep][1][cam])
    assert ND.CompactStore is TS.CompactStore and ND._build_episode.__module__ == "s4d.rwlearn.native_depth"
    assert dict(os.environ) == environment  # the mapping is installed only for the build
    # resume through the mapping verifies (and skips) the published overlays
    assert build_native_depth_mapped(fixture.locator, fixture.rig, fixture.overlays, fixture.out,
                                     raw_root=f"{fixture.root}={local}")["skipped"] == 2


@pytest.mark.parametrize("case", ["sha", "symlink", "outside", "relative", "root_link", "missing"])
def test_mapped_build_refuses_unverified_raw(tmp_path, case):
    from s4d.rwlearn import native_depth as ND
    from s4d.rwlearn.native_depth_mapped import build_native_depth_mapped

    fixture = make_store(tmp_path / "store")
    local = _moved_raw(fixture, tmp_path)
    raw, = (local / "drawer-open").glob("*.h5")
    mapping = f"{fixture.root}={local}"
    if case == "sha":
        with h5py.File(raw, "a") as f:
            f.attrs["tampered"] = 1
    elif case == "symlink":
        copy = tmp_path / "copy.h5"
        raw.rename(copy)
        raw.symlink_to(copy)
    elif case == "outside":
        mapping = f"/elsewhere={local}"
    elif case == "relative":
        mapping = f"{fixture.root}=relative/root"
    elif case == "root_link":
        (tmp_path / "root-link").symlink_to(local)
        mapping = f"{fixture.root}={tmp_path / 'root-link'}"
    else:
        raw.unlink()
    with pytest.raises(ValueError):
        build_native_depth_mapped(fixture.locator, fixture.rig, fixture.overlays, fixture.out, raw_root=mapping)
    assert not fixture.out.exists()
    assert ND.CompactStore is TS.CompactStore and "S4D_NATIVE_DEPTH_RAW_ROOT" not in os.environ


def test_native_depth_cli_build_with_raw_root(tmp_path):
    fixture = make_store(tmp_path / "store")
    local = _moved_raw(fixture, tmp_path)
    script = Path(__file__).resolve().parents[1] / "scripts/realworld_native_depth.py"
    result = subprocess.run([sys.executable, "-I", str(script), "build", "--store", str(fixture.locator),
                             "--rig", str(fixture.rig), "--out-dir", str(fixture.overlays),
                             "--out-store", str(fixture.out), "--workers", "2",
                             "--raw-root", f"{fixture.root}={local}"], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    payload = "\n".join(line for line in result.stdout.splitlines() if not line.startswith("[gpu_guard]"))
    assert json.loads(payload) == {"built": 1, "skipped": 0, "descriptor_written": True, "out_store": str(fixture.out)}
