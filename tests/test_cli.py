"""CLI boundary and text-report tests. Values are synthetic, not experiment evidence."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(f"cli_{name}", REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_summarizer_reports_missing_nonfinite_strict_collapse_and_trends(tmp_path, monkeypatch, capsys):
    summary = script("summarize_run")
    records = [{"step": i, "loss/total": 1.0 + i / 10} for i in range(40)]
    records[-1]["metric/active_fraction_scene"] = 0.1
    records[0]["loss/total"] = float("nan")
    records.insert(1, {"step": 0, "loss/total": 1.0})
    (tmp_path / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in records))
    eval_dir = tmp_path / "eval" / "step_0000040"
    eval_dir.mkdir(parents=True)
    (eval_dir / "summary.json").write_text(
        json.dumps(
            {
                "metric/active_fraction_scene@s2": 0.1,
                "metric/active_fraction_scene@s6": 0.1,
                "metric/active_fraction_dynamic@s2": 0.11,
                "metric/active_fraction_dynamic@s6": 0.12,
                "metric/psnr@s2": 28,
                "metric/psnr@s6": 27,
                "metric/psnr_moving@s2": float("nan"),
                "metric/psnr_moving@s6": float("nan"),
            }
        )
    )
    monkeypatch.setattr(sys, "argv", ["summarize_run.py", str(tmp_path)])
    summary.main()
    output = capsys.readouterr().out

    def row(label):
        return next(line for line in output.splitlines() if label in line)

    assert "@s2" in output and "@s6" in output
    assert row("M1 active fraction scene").count("FAIL") == 2
    assert row("M1 active fraction dynamic").count("PASS") == 2
    assert "PASS" in row("M3 PSNR train") and "FAIL" in row("M3 PSNR train"), "strides are judged separately"
    assert row("M3 PSNR moving").count("unavailable") == 2
    assert row("M3 PSNR held-out").count("missing") == 2
    assert "collapse" in output and "non-finite" in output and "rising" in output
    assert summary.trend([], "loss/total") == "n/a"


def test_summarizer_empty_run(tmp_path, monkeypatch, capsys):
    summary = script("summarize_run")
    monkeypatch.setattr(sys, "argv", ["summarize_run.py", str(tmp_path)])
    summary.main()
    assert "no evaluation summary found" in capsys.readouterr().out


def test_evaluator_rejects_outside_output_before_renderer(monkeypatch):
    evaluator = script("evaluate")
    monkeypatch.setattr(sys, "argv", ["evaluate.py", "--run", "/home/ws/data/droid"])
    monkeypatch.setattr(evaluator, "require_prebuilt_renderer", lambda: pytest.fail("output boundary must run first"))
    with pytest.raises(ValueError, match="inside"):
        evaluator.main()


def test_evaluator_preflights_before_checkpoint_and_data_reads(tmp_path, monkeypatch):
    evaluator = script("evaluate")
    monkeypatch.setattr(sys, "argv", ["evaluate.py", "--run", str(tmp_path)])

    def blocked():
        raise RuntimeError("native renderer blocked")

    monkeypatch.setattr(evaluator, "require_prebuilt_renderer", blocked)
    monkeypatch.setattr(evaluator, "load_config", lambda *a: pytest.fail("renderer must preflight first"))
    with pytest.raises(RuntimeError, match="native renderer blocked"):
        evaluator.main()


def test_long_run_gate_requires_passing_current_source_evidence(tmp_path, monkeypatch):
    import _bootstrap as bootstrap

    monkeypatch.setattr(bootstrap, "REPO", tmp_path)
    monkeypatch.setattr(bootstrap, "source_fingerprints", lambda: {"code.py": "sha256"})
    with pytest.raises(RuntimeError, match="all tests"):
        bootstrap.require_passed_tests()
    docs = tmp_path / "docs"
    docs.mkdir()
    report = {"full_suite_passed": False, "exit_code": 1, "counts": {"passed": 100, "error": 4}}
    (docs / "tests.json").write_text(json.dumps(report))
    with pytest.raises(RuntimeError, match="all tests"):
        bootstrap.require_passed_tests()
    report.update(full_suite_passed=True, exit_code=0, counts={"passed": 100, "failed": 0, "error": 0, "skipped": 0})
    (docs / "tests.json").write_text(json.dumps(report))
    with pytest.raises(RuntimeError, match="stale"):
        bootstrap.require_passed_tests()
    report["source_sha256"] = {"code.py": "sha256"}
    (docs / "tests.json").write_text(json.dumps(report))
    bootstrap.require_passed_tests()


def test_training_gate_precedes_distributed_initialization_and_outputs(tmp_path, monkeypatch):
    trainer = script("train")
    monkeypatch.setattr(sys, "argv", ["train.py", "--config", "fixture.yaml", "--name", "test-only"])
    monkeypatch.setattr(trainer, "load_config", lambda *a: {})

    def blocked():
        raise RuntimeError("full suite has native errors")

    monkeypatch.setattr(trainer, "require_passed_tests", blocked)
    monkeypatch.setattr(trainer.ddp, "init_distributed", lambda: pytest.fail("test gate must run first"))
    with pytest.raises(RuntimeError, match="full suite"):
        trainer.main()
