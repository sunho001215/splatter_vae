"""Pure task/variant construction and input closure; no campaign imports or publication."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from ops.jobs import realworld_local as rw
from s4d.config import load_config
from s4d.train.schedule import lr_function

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("ours400k", ROOT / "runs/setup/tools/register_ours400k_20261113.py")
reg = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reg)


def make_rows(task, source=True, variant="S"):
    spec = json.loads((ROOT / f"configs/realworld/ours_ext/{task}-S.json").read_text())
    m = spec["data_manifest"]
    m["baseline_config"] = f"configs/realworld/ours_ext/baseline200k-{task}-{variant}.yaml"
    current = dict(jobs=json.loads((ROOT / "tests/fixtures/ours400k_templates.json").read_text()))
    before = copy.deepcopy((current, m))
    rows = reg.build_rows(
        task,
        variant,
        m,
        spec["base200k_source"] if source else None,
        current=current,
        release=dict(
            commit="a" * 40, image="sha256:" + "b" * 64, source="local", evidence_manifest="docs/tests.json (local evidence)"
        ),
    )
    assert (current, m) == before
    return rows, current, spec


@pytest.mark.parametrize("task,source,count", [("covering", True, 7), ("pick-and-place", False, 7)])
def test_exact_400k_graph_and_template_science(task, source, count):
    rows, current, spec = make_rows(task, source)
    assert len(rows) == len({r["id"] for r in rows}) == count
    ids = {r["id"]: r for r in rows}
    templates = {r["id"]: r for r in current["jobs"]}
    suffix = f"ours-{task}-S"
    pre = f"rwv2-20261113-pre400k-{suffix}"
    dataset = f"rwv2-20261113-fqldata400k-{suffix}"
    assert "300k" not in str(rows)
    assert all(r["priority"] == (-3 if r["script"] in ("scripts/train.py", rw.OURS_CONTINUATION) else -1) for r in rows)
    assert all(r["realworld_target"] == task and r["gpu_profile"] == "1g.24gb" and r["gpu"] == "any" for r in rows)
    assert all(r["local_placement_pause"] == reg.LOCAL_PAUSE and "pin_host" not in r for r in rows)
    assert ids[pre]["args"][ids[pre]["args"].index("--start-step") + 1] == "200000"
    assert ids[pre]["args"][ids[pre]["args"].index("--stop-step") + 1] == "400000"
    assert spec["data_manifest"]["stats_id"] in ids[pre]["deps"]
    if not source:
        first = ids[f"rwv2-20261113-pre-{suffix}"]
        assert first["deps"] == ["rwcal-20261111-ours-pick-and-place-stats"]
        assert first["id"] in ids[pre]["deps"]
        assert "--source-sha256" not in ids[pre]["args"]
    for seed in range(3):
        row = ids[f"rwv2-20261113-fql400k-{suffix}-s{seed}"]
        old = templates[row["derived_from"]]
        expected = old["args"].copy()
        expected[expected.index("--name") + 1] = row["id"]
        changes = {
            "data.dataset": f"runs/{dataset}/dataset.h5",
            "data.split_manifest": spec["data_manifest"]["split_manifest"],
            "data.gate_report": spec["data_manifest"]["gate_report"],
        }
        expected = [
            f"{x.partition('=')[0]}={changes[x.partition('=')[0]]}" if x.partition("=")[0] in changes else x
            for x in expected
        ]
        assert row["args"] == expected
        assert row["fql_budget"] == old["fql_budget"]
        assert row["deps"] == [dataset]
    for row in rows:
        needed = rw.needed_inputs(row)
        checked = {rw.REMOTE_PREFIX + "/" + p for p in needed["mirror"]} | {
            f"runs/{dep}/{name}" for dep, name in needed["dependency_files"]
        }
        assert rw._resolved_paths(row, ROOT) <= checked


def test_sf_and_new_task_identity_are_independent():
    rows, _, _ = make_rows("covering", False, "SF")
    assert all("-covering-SF" in r["id"] for r in rows)
    assert len(rows) == 8
    rows2 = copy.deepcopy(rows)
    rows2[0]["args"].append("unrelated")
    assert rows2 != rows


def test_single_cycle_keeps_original_horizon_and_no_300k_restart():
    cfg = load_config([ROOT / "configs/realworld/ours_ext/baseline200k-covering-S.yaml", ROOT / reg.SCHEDULE])
    old = copy.deepcopy(cfg)
    old["train"].pop("lr_schedule")
    old["train"].pop("restarts")
    old["train"]["steps"] = 200000
    continued, original = lr_function(cfg), lr_function(old)
    for step in range(200000):
        assert continued(step) == original(step)
    assert cfg["train"]["restarts"] == [dict(start=200000, end=400000, warmup=2500, peak_lr=0.00025)]
    lr = lambda s: cfg["train"]["lr"] * continued(s)
    assert lr(200000) == pytest.approx(1e-5)
    assert lr(202500) == pytest.approx(0.00025)
    assert lr(300000) > 1e-4
    assert lr(400000) == pytest.approx(1e-5)


@pytest.mark.parametrize("mutation", ["variant", "hash", "stats", "baseline", "source"])
def test_bad_manifest_fails_closed(mutation):
    _, current, spec = make_rows("covering")
    m = spec["data_manifest"]
    variant = "S"
    source = spec["base200k_source"]
    if mutation == "variant":
        variant = "success"
    if mutation == "hash":
        m["gate_sha256"] = ""
    if mutation == "stats":
        m["stats_id"] = "wrong"
    if mutation == "baseline":
        m["baseline_config"] = "../config.yaml"
    if mutation == "source":
        source.pop("checkpoint_sha256")
    with pytest.raises(RuntimeError):
        reg.build_rows("covering", variant, m, source, current=current, release={})


def test_local_hold_exact_new_ids(tmp_path):
    rows, _, _ = make_rows("covering")
    path = tmp_path / "runs/setup/admission-pauses.json"
    path.parent.mkdir(parents=True)
    record = dict(rows=rows, local_pause=reg.LOCAL_PAUSE)
    hold = reg.reviewed.local_hold(rows, reg.LOCAL_PAUSE)
    path.write_text(json.dumps(dict(pauses=[hold])))
    reg.reviewed.require_local_hold(tmp_path, record)
    hold["ids"].append("unrelated")
    path.write_text(json.dumps(dict(pauses=[hold])))
    with pytest.raises(RuntimeError):
        reg.reviewed.require_local_hold(tmp_path, record)
