"""s4d/remote_artifact_check.py: the standard-library-only, read-only inspection run on the remote host (2026-10-27)."""
import ast
import os
import sys
from pathlib import Path

import torch

from ops.remote import artifact_check as check

SOURCE = Path(check.__file__)


def test_checker_imports_only_the_standard_library():
    imported = set()
    for node in ast.walk(ast.parse(SOURCE.read_text())):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module.split(".")[0])
    assert imported <= set(sys.stdlib_module_names)


def test_reports_checksums_small_records_and_archive_structure(tmp_path):
    (tmp_path / "exit.json").write_text('{"code": 0}')
    (tmp_path / "checkpoints").mkdir()
    torch.save({"step": 5, "encoder": {"a": torch.zeros(4, 2), "b": torch.ones(3, dtype=torch.float16)},
                "rng": {"numpy": (1, 2)}}, tmp_path / "checkpoints" / "step_5.pt")
    report = check.inspect_run(str(tmp_path), ["exit.json", "checkpoints/step_5.pt"])
    assert report["problems"] == [] and report["payloads"] == {"exit.json": '{"code": 0}'}
    assert len(report["files"]["exit.json"]["sha256"]) == 64 and report["files"]["exit.json"]["bytes"] == 11
    archive = report["archives"]["checkpoints/step_5.pt"]
    assert archive["zip_ok"] and archive["step"] == 5 and archive["encoder_storage_problems"] == []
    assert archive["encoder"] == {"a": {"dtype": "FloatStorage", "shape": [4, 2]},
                                  "b": {"dtype": "HalfStorage", "shape": [3]}}


def test_missing_symlinked_outside_and_corrupt_files_are_reported(tmp_path):
    (tmp_path / "outside.json").write_text("{}")
    tmp_path = tmp_path / "run"
    tmp_path.mkdir()
    torch.save({"step": 1, "encoder": {"a": torch.zeros(1000)}}, tmp_path / "good.pt")
    data = bytearray((tmp_path / "good.pt").read_bytes())
    data[len(data) // 2] ^= 0xFF  # flip one byte inside the tensor data: the member CRC fails
    (tmp_path / "corrupt.pt").write_bytes(bytes(data))
    (tmp_path / "truncated.pt").write_bytes(bytes(data[: len(data) // 3]))
    os.symlink(tmp_path / "good.pt", tmp_path / "link.pt")
    names = ["missing.json", "link.pt", "../outside.json", "corrupt.pt", "truncated.pt"]
    report = check.inspect_run(str(tmp_path), names)
    assert [p["name"] for p in report["problems"]] == ["missing.json", "link.pt", "../outside.json"]
    assert report["archives"]["corrupt.pt"]["zip_ok"] is False and report["archives"]["truncated.pt"]["zip_ok"] is False


def test_no_pickled_global_is_resolved(tmp_path, monkeypatch):
    class Evil:
        def __reduce__(self):
            return (os.system, ("echo unsafe",))
    torch.save({"step": 1, "evil": Evil()}, tmp_path / "evil.pt")
    called = []
    monkeypatch.setattr(os, "system", lambda *args: called.append(args))
    archive = check.inspect_run(str(tmp_path), ["evil.pt"])["archives"]["evil.pt"]
    assert archive["zip_ok"] and "posix.system" in archive["globals"] and not called


def test_non_contiguous_or_out_of_bounds_tensors_are_storage_problems(tmp_path):
    base = torch.arange(12.0)
    torch.save({"step": 1, "encoder": {"view": base.view(3, 4).t(), "slice": base[4:8], "ok": torch.zeros(2, 2)}},
               tmp_path / "views.pt")
    archive = check.inspect_run(str(tmp_path), ["views.pt"])["archives"]["views.pt"]
    assert archive["encoder_storage_problems"] == ["view"]  # an offset slice is valid, a transposed view is not
