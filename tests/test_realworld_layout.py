"""Real-world implementation packages retain the historical module objects and command lines."""

from __future__ import annotations

import ast
import importlib
import os
import pickle
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ALIASES = {
    **{f"s4d.realworld.{old}": f"s4d.realworld.calib.{new}" for old, new in (
        ("calibration", "core"), ("calibration_rig", "rig"), ("calibration_board", "board"),
        ("calibration_release", "release"))},
    **{f"s4d.realworld.rig_dw_{name}": f"s4d.realworld.calib.dw.{name}" for name in (
        "boardcv", "data", "extext", "fixedness", "frames", "geom", "matches", "publish", "robot", "run", "solve")},
    **{f"s4d.realworld.rig_loma_{name}": f"s4d.realworld.calib.loma.{name}" for name in (
        "corr", "data", "depth", "publish", "render", "run", "solve")},
    **{f"s4d.realworld.{old}": f"s4d.realworld.stages.{new}" for old, new in (
        ("calibration_stage", "calibrate"), ("depth_stages", "depth"), ("ir_stereo_depth", "ir_depth"),
        ("baseline_stages", "semantics"), ("trackeverything_stage", "trackeverything"), ("track_motion", "motion"))},
    **{f"s4d.realworld.{old}": f"s4d.realworld.robotics.{new}" for old, new in (
        ("robot", "model"), ("robot_visual", "visual"), ("mounts", "mounts"))},
}
CLI_ALIASES = {old: new for old, new in ALIASES.items() if old.rsplit(".", 1)[1] in (
    "calibration_release", "rig_dw_boardcv", "rig_dw_data", "rig_dw_fixedness", "rig_dw_publish", "rig_dw_run",
    "rig_loma_data", "rig_loma_depth", "rig_loma_publish", "rig_loma_run")}
ENV = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1", "COLUMNS": "80"}


@pytest.mark.parametrize("old,new", sorted(ALIASES.items()))
def test_old_module_is_the_implementation(old, new):
    module = importlib.import_module(old)  # exercise old-first imports too
    assert module is importlib.import_module(new) is sys.modules[old]
    parent, leaf = old.rsplit(".", 1)
    assert getattr(importlib.import_module(parent), leaf) is module
    shim = REPO / (old.replace(".", "/") + ".py")
    assert len(shim.read_text().splitlines()) < 25
    assert Path(module.__file__) == REPO / (new.replace(".", "/") + ".py")
    # Definitions pickle under their implementation path, and historical (old-path) lookups resolve to the same objects.
    tree = ast.parse(Path(module.__file__).read_text())
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            definition = getattr(module, node.name)
            assert definition.__module__ == new
            assert pickle.loads(pickle.dumps(definition)) is definition
            assert pickle.loads(pickle.dumps(definition, protocol=0).replace(new.encode(), old.encode())) is definition  # GLOBAL opcode


def isolated(code, *args):
    # -I omits PYTHONPATH and cwd. Bootstrap only this checkout, then use Python's -m runner below.
    return subprocess.run([sys.executable, "-I", "-c", f"import sys; sys.path.insert(0, {str(REPO)!r}); " + code, *args],
                          cwd=REPO, env=ENV, capture_output=True, text=True, timeout=120)


@pytest.mark.parametrize("old,new", sorted(CLI_ALIASES.items()))
def test_old_isolated_module_and_script_help_keep_argv(old, new):
    script = REPO / (old.replace(".", "/") + ".py")
    old_module = isolated("import runpy; runpy._run_module_as_main(sys.argv.pop(1))", old, "--help")
    old_script = subprocess.run([sys.executable, "-I", str(script), "--help"], cwd=REPO, env=ENV,
                                capture_output=True, text=True, timeout=120)
    # Run the implementation with exactly the historical argv, as the shim does. The usage basename must survive.
    implementation = isolated("import runpy; module = sys.argv[1]; sys.argv[:] = sys.argv[2:]; "
                              "runpy.run_module(module, run_name='__main__')", new, str(script), "--help")
    expected = (0, old_module.stdout, "")
    assert (old_module.returncode, old_module.stdout, old_module.stderr) == expected
    assert (old_script.returncode, old_script.stdout, old_script.stderr) == expected
    assert (implementation.returncode, implementation.stdout, implementation.stderr) == expected
    assert old_module.stdout.startswith(f"usage: {script.name}")
    # A native -m call at the new path uses its new argv[0] (and argparse wraps usage accordingly).
    new_module = isolated("import runpy; runpy._run_module_as_main(sys.argv.pop(1))", new, "--help")
    new_basename = new.rsplit(".", 1)[1] + ".py"
    assert new_module.returncode == 0 and new_module.stderr == ""
    assert new_module.stdout.startswith(f"usage: {new_basename}")


def test_release_keeps_file_relative_paths(monkeypatch):
    from s4d.realworld.calib import release

    assert release.INGEST_WORKER == REPO / "runs/setup/tools/realworld_ingest_20261106/ingest_worker.py"
    calls = []

    def capture(argv, **kwargs):
        from types import SimpleNamespace

        calls.append(argv)
        return SimpleNamespace(stdout="test-commit\n")

    monkeypatch.setattr(release.subprocess, "run", capture)
    assert release._code_commit() == "test-commit"
    assert calls == [["git", "-C", str(REPO / "s4d/realworld"), "rev-parse", "HEAD"]]


def test_dw_and_loma_reexport_the_shared_definitions():
    from s4d.realworld.calib.dw import common, publish, run
    from s4d.realworld.calib.loma import run as loma

    assert publish.CAMS is run.CAMS is loma.CAMS is common.CAMS
    for name in ("URDF", "episode_split", "heldout_extext", "matchset", "perturb_common"):
        assert getattr(run, name) is getattr(loma, name) is getattr(common, name)
