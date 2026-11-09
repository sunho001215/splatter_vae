"""Runner-based dispatch shared by local and remote entry points; no transport or model calls."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from s4d.realworld import baseline_stages as B
from s4d.realworld import cli
from s4d.realworld import depth_stages as D
from s4d.realworld import export as E
from s4d.realworld import ir_stereo_depth as IR
from s4d.realworld import pipeline as P
from s4d.realworld import track_motion as TM
from s4d.realworld import trackeverything_stage as TE
from s4d.realworld import train_store as TS
from s4d.realworld.runners import LocalRunner

REPO = Path(__file__).resolve().parents[1]


def _standard():
    return yaml.safe_load((REPO / "configs/realworld/mug_hanging.yaml").read_text())


def test_standard_dispatch_order_runner_and_depth_routes(tmp_path, monkeypatch):
    from ops.inference_server.run_episode import STAGES

    assert STAGES == cli.STAGES + cli.BASELINE_STAGES
    runner = LocalRunner(tmp_path / "worker")
    calls = []
    monkeypatch.setattr(P, "load_meta", lambda *a: SimpleNamespace(
        episode_id="ep", views=lambda: [SimpleNamespace(stream_id="cam", kind="rgbd")]))

    def stub(name):
        def call(*args, **kwargs):
            calls.append((name, args, kwargs))
            if name == "track_everything":
                return {"track2d": {"summary": {}}, "lift3d": {"summary": {}}}
            return {"summary": {}}
        return call

    for module, names in ((P, ("ingest", "calibrate")), (D, ("sensor_depth",)), (IR, ("ir_stereo_depth",)),
                          (B, ("sam3_masks",)), (TE, ("track_everything",)), (TM, ("track_motion",)),
                          (E, ("export",)), (TS, ("write_episode",))):
        for name in names:
            monkeypatch.setattr(module, name, stub(name))
    cfg = {**_standard(), "output_root": str(tmp_path / "out")}
    wanted = cli.default_stages(cfg)
    cli.check_stage_scope(cfg, wanted)
    assert set(wanted) == {"ingest", "sensor_depth", "foundation_stereo_depth", "sam3_masks", "calibrate",
                           "track_everything", "track_motion", "train_data"}  # no window export, no DINO cache
    assert cli.stage_workers(wanted) == {"foundation_stereo", "trackeverything", "sam3"}

    def run(name, fn):
        if name in wanted:
            fn(False)

    cli.run_stages(cfg, None, "ep", runner, run, wanted, spec_factory=lambda name: name)
    names = [name for name, _, _ in calls]
    assert names == ["ingest", "sensor_depth", "ir_stereo_depth", "sam3_masks", "calibrate", "track_everything",
                     "track_motion", "write_episode"]
    by = {name: (args, kwargs) for name, args, kwargs in calls}
    assert by["sensor_depth"][0][2] is runner and by["ir_stereo_depth"][0][2:4] == ("foundation_stereo", runner)
    assert by["calibrate"][1]["runner"] is runner and by["calibrate"][1]["cfg"].evidence == "rgbd"
    assert by["calibrate"][1]["depth_stage"] == "sensor_depth" and by["calibrate"][0][2] is None
    assert by["track_everything"][0][2:4] == ("trackeverything", runner)
    assert by["track_everything"][1]["depth_stage"] == "foundation_stereo_depth"
    assert by["track_everything"][1]["model_hw"] == (384, 512)
    store_args, store_kw = by["write_episode"]
    assert store_args[2] == Path(cfg["output_root"]) / "train_data" / "ep"
    assert store_kw["task"] == "mug-hanging" and store_kw["depth_stage"] == "foundation_stereo_depth"
    assert store_kw["max_sync_ns"] == 30_000_000 and store_kw["semantic"] is True  # SNeRL declared: semantics stored
    assert by["sam3_masks"][1]["runner"] is runner
    # the former window export runs only when configured explicitly; the DINO cache stage no longer exists
    legacy = {**cfg, "stages": {**cfg["stages"], "export": {"depth_stage": "foundation_stereo_depth"}}}
    assert "export" in cli.default_stages(legacy)
    with pytest.raises(ValueError, match="not stages of this pipeline"):
        cli.check_stage_scope({**cfg, "stages": {**cfg["stages"], "dino_snerl": {}}}, ["ingest"])


def test_unconfigured_and_unknown_stages_are_refused():
    cfg = _standard()
    with pytest.raises(ValueError, match="not stages of this pipeline"):
        cli.check_stage_scope({**cfg, "stages": {**cfg["stages"], "mapanything_depth": {}}}, ["ingest"])
    provided = {**cfg, "stages": {**cfg["stages"], "calibrate": {"source": "adapter_provided"}}}
    with pytest.raises(ValueError, match="calibrate.source"):
        cli.check_stage_scope(provided, ["calibrate"])
    no_te = {**cfg, "stages": {k: v for k, v in cfg["stages"].items() if k != "track_everything"}}
    assert "track_everything" not in cli.default_stages(no_te)  # no default tracker
    with pytest.raises(ValueError, match="not configured"):
        cli.check_stage_scope(no_te, ["track_everything"])


def test_remote_worker_paths_override_host_paths_and_preserve_the_standard_config():
    from ops.inference_server.run_episode import remote_pipeline_config, spec

    server = yaml.safe_load((REPO / "configs/realworld/inference_server.yaml").read_text())
    original = _standard()
    resolved = remote_pipeline_config(original, server)
    assert resolved["adapter"] == original["adapter"]
    assert resolved["output_root"] == original["output_root"] and resolved["stages"] == original["stages"]
    assert spec(resolved, "s4d_kernels").python == "/opt/envs/s4d/bin/python"
    for worker in ("foundation_stereo", "trackeverything", "sam3", "dino_snerl"):  # dino_snerl: SNeRL contract check
        assert spec(resolved, worker) == spec(original, worker)
    fs = resolved["workers"]["foundation_stereo"]
    assert fs["python"] == "/opt/envs/foundation_stereo/bin/python"
    assert fs["checkpoint"] == "/ram/weights/foundation_stereo/23-51-11/model_best_bp2.pth"
    assert set(server["workers"]) == {"sam3", "foundation_stereo", "trackeverything", "dino_snerl", "s4d_kernels"}
    assert {w["worker"] for w in server["weights"]} <= set(server["workers"])


def test_local_run_validates_before_writing_the_root_config(tmp_path):
    cfg = {**_standard(), "output_root": str(tmp_path / "out")}
    path = tmp_path / "pipeline.yaml"
    path.write_text(yaml.safe_dump(cfg))
    for stages in ("unknown", "dino_snerl,mapanything_depth", "s2m2_depth"):
        with pytest.raises(SystemExit) as exc:
            cli.main(["run", "--config", str(path), "--episode", "mug-hanging-test-000001", "--stages", stages])
        assert exc.value.code == 2
    bad = {**cfg, "stages": {**cfg["stages"], "lift3d": {}}}
    path.write_text(yaml.safe_dump(bad))
    with pytest.raises(SystemExit):
        cli.main(["run", "--config", str(path), "--episode", "mug-hanging-test-000001"])
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("stage", ["unknown", "s2m2_depth"])
def test_remote_rejects_unknown_stages_before_reading_server_config(stage):
    from ops.inference_server.run_episode import main

    with pytest.raises(SystemExit) as exc:
        main(["--pipeline", "/unread/pipeline", "--server", "/unread/server", "--episode", "ep", "--stages", stage])
    assert exc.value.code == 2


def test_status_recognizes_retired_depth_stage_records(tmp_path, capsys):
    from s4d.realworld import io as rio

    rec = rio.run_stage(rio.stage_dir(tmp_path, "ep", "s2m2_depth"), stage="s2m2_depth", version="1",
                        config={}, inputs={}, build=lambda _: {})
    assert cli.main(["status", "--root", str(tmp_path), "--episode", "ep"]) == 0
    line = next(line for line in capsys.readouterr().out.splitlines() if line.startswith("s2m2_depth"))
    assert line == f"{'s2m2_depth':13s} complete {rec['fingerprint'][:12]}"
