"""Pure registration and verified local-pool input tests. Never imports a scheduler or campaign driver."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from ops.jobs import host_roles
from ops.jobs import realworld_local as rw
from s4d.train.continuation import file_sha

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("ours_ext_reg", ROOT / "runs/setup/tools/register_ours_ext_20261113.py")
reg = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reg)


def queue():
    rows = json.loads((ROOT / "tests/fixtures/ours_ext_templates.json").read_text())
    return {"jobs": rows + [dict(id="other", priority=0), dict(id="other-fql", priority=1, role="realworld_fql")]}


def make_plan():
    cfg = yaml.safe_load((ROOT / reg.BASE).read_text())
    release = dict(commit="a" * 40, image="sha256:" + "b" * 64, evidence_manifest="acceptance.json")
    return reg.build_rows(queue(), release, "c" * 64, file_sha(ROOT / reg.BASE), cfg)


def test_dry_run_rows_priorities_exact_fql_and_independent_branches(tmp_path, capsys):
    current = queue()
    rows, priorities = make_plan()
    assert len(rows) == len({r["id"] for r in rows}) == 15
    assert priorities["pretraining"] == -1 and priorities["ours_fql"] == 0
    ids = {r["id"]: r for r in rows}
    sources = {r["id"]: r for r in current["jobs"]}
    pre_a, pre_b = reg.PREFIX + "pre300k", reg.PREFIX + "pre400k"
    assert ids[pre_a]["deps"] == [reg.OLD, reg.STATS]
    assert ids[pre_b]["deps"] == [pre_a, reg.STATS]
    assert all(ids[job]["priority"] < min(r.get("priority", 5) for r in current["jobs"]) for job in (pre_a, pre_b))
    for label in ("300k", "400k"):
        dataset = f"ourx-20261113-fqldata{label}-ours-drawer-open"
        assert ids[dataset]["deps"] == [reg.PREFIX + "export" + label]
        for seed in range(3):
            job = f"ourx-20261113-fql{label}-ours-drawer-open-s{seed}"
            source = sources[ids[job]["derived_from"]]
            expected = [
                a.replace(source["id"], job).replace("rwcal-20261111-fqldata-ours-drawer-open", dataset)
                for a in source["args"]
            ]
            assert ids[job]["args"] == expected
            assert ids[job]["fql_budget"] == source["fql_budget"]
            assert ids[job]["deps"] == [dataset]
            assert ids[job]["priority"] < min(
                r.get("priority", 5) for r in current["jobs"] if r.get("role") == "realworld_fql"
            )
        assert pre_b not in ids["ourx-20261113-fql300k-ours-drawer-open-s0"]["deps"]
    for row in rows:
        assert "pin_host" not in row and row["gpu"] == "any"
        assert row["gpu_profile"] == "1g.24gb" and row["cpu_threads"] <= 16
        assert row["mem_gb"] == row["gpu_mem_gb"] < 24
        assert "config.yaml" not in row["required_results"]  # container wrapper reserves this result name
        assert rw.has_resolver(row)
        assert host_roles.job_class(row, {**sources, **ids}).cls in ("realworld_pretrain", "realworld_fql")
        needed = rw.needed_inputs(row)
        if row["script"] in (rw.OURS_CONTINUATION, rw.OURS_EVAL):
            # Every remote/run path hidden in the baseline is a consumed, plain, checked argument.
            checked = {rw.REMOTE_PREFIX + "/" + path for path in needed["mirror"]} | {
                f"runs/{dep}/{path}" for dep, path in needed["dependency_files"]
            }
            assert rw._resolved_paths(row, ROOT) <= checked
    before = yaml.safe_dump(current).encode()
    candidate = reg.queue_candidate(before, rows)
    assert yaml.safe_load(candidate)["jobs"] == current["jobs"] + rows
    (tmp_path / "experiments").mkdir()
    (tmp_path / "experiments/queue.yaml").write_bytes(before)
    (tmp_path / "experiments/registry.jsonl").write_bytes(b"")
    original = tmp_path / "runs" / reg.OLD
    (original / "checkpoints").mkdir(parents=True)
    (original / "config.yaml").write_bytes((ROOT / reg.BASE).read_bytes())
    (original / "checkpoints/step_0200000.pt").write_bytes(b"CPU fixture checkpoint")
    from s4d.config import load_config

    for seed in range(3):
        job = f"rwnq-20261111-fql-ours-drawer-open-s{seed}"
        args = sources[job]["args"]
        saved = load_config([ROOT / args[args.index("--config") + 1]], args[args.index("--set") + 1 :])
        path = tmp_path / "runs" / job / "config.yaml"
        path.parent.mkdir(parents=True)
        path.write_text(yaml.safe_dump(saved))
    reg.main(["--repo", str(tmp_path), "--commit", "a" * 40, "--image", "sha256:" + "b" * 64])
    report = json.loads(capsys.readouterr().out)
    assert report["registered"] is False and report["queue_diff"]["added_rows"] == 15
    assert report["queue_diff"]["modified_rows"] == 0
    assert (tmp_path / "experiments/queue.yaml").read_bytes() == before
    assert not (tmp_path / "runs/setup").exists()
    # Rerun against a changed priority field computes a strictly higher new priority.
    current["jobs"].append(dict(id="higher-fql", role="realworld_fql", priority=-10))
    cfg = yaml.safe_load((ROOT / reg.BASE).read_text())
    _, priority = reg.build_rows(current, report["release"], "c" * 64, "d" * 64, cfg)
    assert priority["ours_fql"] == -11 and priority["pretraining"] == -12


def test_execute_refuses_worktree_and_unverified_inputs(tmp_path):
    with pytest.raises(RuntimeError, match="coordinator"):
        reg.execute(ROOT, {})
    rows, _ = make_plan()
    bad = copy.deepcopy(rows[0])
    bad["args"][bad["args"].index("--source-checkpoint") + 1] = "runs/undeclared/checkpoints/step_0200000.pt"
    with pytest.raises(rw.LocalInputError, match="dependency"):
        rw.needed_inputs(bad)
    bad = copy.deepcopy(rows[0])
    bad["args"].append("loss.chamfer=1")
    with pytest.raises(rw.LocalInputError, match="four verified"):
        rw.needed_inputs(bad)


def test_pure_mig_profile_and_completion_predicates():
    from ops.jobs.continuation import continuation_endpoint, profile_matches

    assert profile_matches({"gpu_profile": "1g.24gb"}, "MIG-one", {"MIG-one"})
    assert not profile_matches({"gpu_profile": "1g.24gb"}, "GPU-physical", {"MIG-one"})
    assert profile_matches({}, "GPU-physical", set())
    with pytest.raises(ValueError):
        profile_matches({"gpu_profile": "7g.192gb"}, "MIG-one", {"MIG-one"})
    completion = dict(status="completed", steps=300000, schedule_steps=400000)
    assert continuation_endpoint(["--stop-step", "300000"], completion) == 300000
    with pytest.raises(ValueError):
        continuation_endpoint(["--stop-step", "300000"], dict(completion, steps=400000))
    with pytest.raises(ValueError):
        continuation_endpoint(["--stop-step", "400000"], completion)


def test_container_output_paths_for_new_jobs():
    from ops.containers.launch import validate_output_arguments

    for script in ("scripts/train_ours_continuation.py", "scripts/evaluate_ours_length.py"):
        validate_output_arguments(script, ["--name", "own", "--output-root", "runs"], "own")
        with pytest.raises(ValueError, match="artifacts"):
            validate_output_arguments(script, ["--name", "other", "--output-root", "runs"], "own")


def test_actual_legacy_completion_endpoint_is_validated_without_driver_imports():
    from ops.jobs.continuation import pretraining_endpoint

    assert pretraining_endpoint({"steps": 10, "stop_step": 5}) == 5
    assert pretraining_endpoint({"steps": 10, "stop_step": 10**309}) == 10
    assert pretraining_endpoint({"steps": 10, "stop_step": 0}) == 10
    assert pretraining_endpoint({"steps": 10, "stop_step": float("nan")}) is None
    assert pretraining_endpoint({"steps": 10, "stop_step": True}) is None
    assert pretraining_endpoint({"steps": 10, "stop_step": "5"}) is None


def test_explicit_step_overrides_do_not_hide_early_stop():
    from ops.jobs.continuation import resolved_pretraining_endpoint

    assert resolved_pretraining_endpoint(["train.steps=400000", "train.stop_step=300000"], {}) == 300000
    assert resolved_pretraining_endpoint(["train.steps=10"], {"train": {"steps": 5, "stop_step": 3}}) == 3
    assert resolved_pretraining_endpoint([], {"train": {"steps": 200000}}) == 200000
    assert resolved_pretraining_endpoint(["train.steps=10"], {}) == 10
