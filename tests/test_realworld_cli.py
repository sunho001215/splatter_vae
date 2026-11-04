"""Runner-based dispatch shared by local and remote entry points; no transport or model calls."""

from types import SimpleNamespace

import pytest

from s4d.realworld import cli
from s4d.realworld import depth_stages as D
from s4d.realworld import pipeline as P
from s4d.realworld.runners import LocalRunner


@pytest.mark.parametrize("stereo", [False, True], ids=["rgbd-evidence", "stereo-provided"])
def test_stage_dispatch_preserves_depth_routes_and_runner(stereo, tmp_path, monkeypatch):
    from ops.inference_server.run_episode import STAGES

    assert STAGES == cli.STAGES + cli.BASELINE_STAGES
    runner = LocalRunner(tmp_path / "worker")
    calls = []
    provided = {"rig/left": {"T_cam_base": "provided"}}
    adapter = SimpleNamespace(provided_extrinsics=lambda ep: provided)
    monkeypatch.setattr(P, "load_meta", lambda *a: SimpleNamespace(stereo=("rig",) if stereo else ()))

    def stub(name):
        def call(*args, **kwargs):
            calls.append((name, args, kwargs))
            return {"summary": {}}
        return call

    for module, names in ((P, ("ingest", "stereo_rectify", "calibrate", "calibrate_provided", "lift3d")),
                           (D, ("sensor_depth", "mapanything_rig_depth", "mapanything_depth"))):
        for name in names:
            monkeypatch.setattr(module, name, stub(name))
    cfg = {"output_root": str(tmp_path / "out"), "workers": {"mapanything": {"python": "unused"}},
           "stages": {"calibrate": {"source": "adapter_provided"} if stereo else {"evidence": "rgbd"}}}
    wanted = list(cli.STAGES[:6]) + ["lift3d"]

    def run(name, fn):
        if name in wanted:
            fn(False)

    cli.run_stages(cfg, adapter, "ep", runner, run, wanted)
    names = [name for name, _, _ in calls]
    if stereo:
        assert names == ["ingest", "sensor_depth", "stereo_rectify", "mapanything_rig_depth",
                         "calibrate_provided", "mapanything_depth", "lift3d"]
        assert calls[4][1][2] is provided
    else:
        assert names == ["ingest", "sensor_depth", "calibrate", "mapanything_depth", "lift3d"]
        assert calls[2][2]["cfg"].evidence == "rgbd" and calls[2][1][2] is None
    for name, args, kwargs in calls:
        if name in ("stereo_rectify", "calibrate", "lift3d"):
            assert kwargs["runner"] is runner
        if name in ("mapanything_rig_depth", "mapanything_depth"):
            assert args[2].runner is runner
        if name == "sensor_depth":
            assert args[2] is runner


def test_remote_worker_paths_override_host_droid_paths_and_preserve_rgbd_config():
    from pathlib import Path

    import yaml

    from ops.inference_server.run_episode import remote_pipeline_config, spec

    repo = Path(__file__).resolve().parents[1]
    server = yaml.safe_load((repo / "configs/realworld/inference_server.yaml").read_text())
    for name in ("droid_raw_validation", "mug_hanging_rgbd"):
        original = yaml.safe_load((repo / f"configs/realworld/{name}.yaml").read_text())
        resolved = remote_pipeline_config(original, server)
        assert resolved["adapter"] == original["adapter"]
        assert resolved["output_root"] == original["output_root"] and resolved["stages"] == original["stages"]
        assert spec(resolved, "s4d_kernels").python == "/opt/envs/s4d/bin/python"
        ma = resolved["workers"]["mapanything"]
        assert ma["python"] == "/opt/envs/mapanything/bin/python"
        assert ma["repo_dir"] == "/opt/upstream/map-anything"
        assert ma["checkpoint"] == "/ram/weights/map-anything-a1d87e90/model.safetensors"
        assert ma["params"]["hf_dir"] == "/ram/weights/map-anything-a1d87e90"
        assert ma["params"]["model_hw"] == original["workers"]["mapanything"]["params"]["model_hw"]
        if name == "mug_hanging_rgbd":
            for worker in ("mapanything", "alltracker", "sam3", "dino_snerl"):
                assert spec(resolved, worker) == spec(original, worker)


def test_remote_rejects_unknown_stages_before_reading_server_config():
    from ops.inference_server.run_episode import main

    with pytest.raises(SystemExit) as exc:
        main(["--pipeline", "/unread/pipeline", "--server", "/unread/server", "--episode", "ep", "--stages", "unknown"])
    assert exc.value.code == 2
