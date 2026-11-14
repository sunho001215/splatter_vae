"""Offline coordinator transaction tests; never consult or mutate live state."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "ours_task_eval_reg", ROOT / "runs/setup/tools/register_ours_task_eval_20261115.py"
)
reg = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reg)

COMMIT = "a" * 40
IMAGE = "sha256:" + "b" * 64
RELEASE = dict(commit=COMMIT, image=IMAGE, source="local", evidence_manifest=reg.R.LOCAL_EVIDENCE)
PICK = "rwv2-20261113-eval400k-ours-pick-and-place-S"


def old_eval(task):
    spec = json.loads((ROOT / f"configs/realworld/ours_ext/{task}-S.json").read_text())
    spec["data_manifest"]["include_eval"] = True
    current = dict(jobs=json.loads((ROOT / "tests/fixtures/ours400k_templates.json").read_text()))
    rows = reg.reg.build_rows(task, "S", spec["data_manifest"], spec["base200k_source"],
                              current=current, release=RELEASE)
    row = rows[-1]
    assert row["script"] == reg.NEW_SCRIPT
    row.update(script=reg.OLD_SCRIPT, host="local", pin_host="local", placement="historical placement")
    return row


@pytest.fixture
def campaign(tmp_path):
    (tmp_path / "experiments").mkdir()
    (tmp_path / "runs/setup").mkdir(parents=True)
    covering, pick = old_eval("covering"), old_eval("pick-and-place")
    drawer = copy.deepcopy(covering)
    drawer.update(id="ourx-20261113-ours-drawer-open-eval400k", realworld_target="drawer-open")
    drawer["args"][drawer["args"].index("--name") + 1] = drawer["id"]
    drawer["args"][drawer["args"].index("--config") + 1] = "configs/realworld/ours_ext/baseline200k.yaml"
    unrelated = dict(id="unrelated", script="scripts/other.py", host="local", args=["untouched"],
                     arbitrary={"boolean": True, "integer": 1})
    document = dict(hosts={"local": {}, "remote": {}}, policy={"retain": [False, 0]},
                    jobs=[covering, pick, drawer, unrelated])
    (tmp_path / "experiments/queue.yaml").write_text(yaml.safe_dump(document, sort_keys=False))
    events = [dict(id=reg.COVERING, event="failed", time=1)]
    for dep, name in reg.rw.needed_inputs(covering)["dependency_files"]:
        path = tmp_path / "runs" / dep / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"original checkpoint or stats fixture")
        events.append(dict(id=dep, event="exited", code=0, attempt=1, time=0))
    (tmp_path / "experiments/registry.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    pauses = dict(version=1, pauses=[dict(name="unrelated", ids=["other"], prefixes=[], active=True),
                                   reg.R.local_hold([dict(id="other-held-row")], reg.reg.LOCAL_PAUSE)])
    (tmp_path / reg.PAUSES).write_text(json.dumps(pauses))
    return tmp_path, document, pauses


def prepare(root):
    return reg.prepare(root, ROOT, COMMIT, IMAGE)


def snapshot(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def allow_fixture_publication(root, record, monkeypatch):
    """Replace only adopted-main/provenance boundaries; keep the real IO and both locks."""
    from ops.jobs import migration

    monkeypatch.setattr(reg, "MAIN", root)
    monkeypatch.setattr(reg, "__file__", str(root / reg.TOOL))
    for name in reg.TOOLS:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / name).read_bytes())
    record["source_root"] = str(root)
    monkeypatch.setattr(reg.R, "local_release_pin", lambda *args: RELEASE)
    monkeypatch.setattr(reg.rw, "placement_scope", lambda *args, **kwargs: {})
    monkeypatch.setattr(migration, "commit_reader", lambda *args: (lambda name: (ROOT / name).read_bytes(), []))


def test_dry_run_replacements_keep_exact_science_resources_and_all_existing_rows(campaign, tmp_path):
    root, before, _ = campaign
    original = snapshot(root)
    record = prepare(root)
    assert snapshot(root) == original
    assert record["status"] == "dry-run" and record["registered"] is False
    assert record["queue_diff"] == dict(added_rows=2, modified_rows=0, removed_rows=0,
                                        added_ids=[reg.COVERING + "-v2", PICK + "-v2"])
    assert record["superseded_ids"] == [PICK]
    for source, row in zip(before["jobs"], record["rows"]):
        expected_args = copy.deepcopy(source["args"])
        expected_args[expected_args.index("--name") + 1] += "-v2"
        assert row["args"] == expected_args
        for key in ("deps", "inputs", "required_results", "mem_gb", "ram_gb", "disk_gb", "cpu_threads",
                    "gpu_mem_gb", "max_restarts", "realworld_gate_sha256", "realworld_manifest_sha256"):
            assert row[key] == source[key]
        assert row["script"] == reg.NEW_SCRIPT and row["gpu_profile"] == "1g.24gb" and row["priority"] == -1
        assert row["host"] == "remote" and row["release_source"] == "local"
        assert "placement" not in row and "pin_host" not in row
    candidate = reg.queue_candidate((root / "experiments/queue.yaml").read_bytes(), record["rows"])
    parsed = reg.R.queue_document(candidate)
    assert parsed == {**before, "jobs": before["jobs"] + record["rows"]}
    assert parsed["jobs"][0] == before["jobs"][0]  # Failed covering history untouched.
    assert len(record["input_sha256"]) >= len(reg.TOOLS) + 4


def test_cli_defaults_to_dry_run(campaign, capsys):
    root, _, _ = campaign
    before = snapshot(root)
    assert reg.main(["--repo", str(root), "--source-root", str(ROOT), "--commit", COMMIT, "--image", IMAGE]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "dry-run"
    assert snapshot(root) == before


def test_execute_requires_released_main_copy(campaign):
    root, _, _ = campaign
    before = snapshot(root)
    with pytest.raises(RuntimeError, match="coordinator released main-checkout"):
        reg.execute(root, prepare(root))
    assert snapshot(root) == before


def test_execute_appends_only_and_holds_old_rows_before_publication(campaign, monkeypatch):
    root, before, original_pauses = campaign
    record = prepare(root)
    allow_fixture_publication(root, record, monkeypatch)
    registry = (root / "experiments/registry.jsonl").read_bytes()
    atomic_write = reg.R.atomic_write

    def checked_write(path, data):
        if path == root / reg.PAUSES:
            assert reg.R.queue_document((root / "experiments/queue.yaml").read_bytes()) == before
        return atomic_write(path, data)

    monkeypatch.setattr(reg.R, "atomic_write", checked_write)
    result = reg.execute(root, record)
    assert result["registered"] is True
    parsed = reg.scheduler.load_queue(root / "experiments/queue.yaml")
    assert parsed["jobs"] == before["jobs"] + record["rows"]
    assert (root / "experiments/registry.jsonl").read_bytes() == registry
    assert reg.scheduler.read_registry(root / "experiments/registry.jsonl")[reg.COVERING]["status"] == "failed"
    pauses = json.loads((root / reg.PAUSES).read_bytes())
    assert pauses["pauses"][0] == original_pauses["pauses"][0]
    holds = {h["name"]: h for h in pauses["pauses"]}
    assert set(holds[reg.reg.LOCAL_PAUSE]["ids"]) == {"other-held-row", reg.COVERING + "-v2", PICK + "-v2"}
    assert holds[reg.OBSOLETE_PAUSE]["ids"] == [PICK]
    reg.require_holds(root, record)
    assert not (root / f"runs/setup/{reg.STEM}.pending.json").exists()
    assert (root / f"runs/setup/{reg.STEM}.json").exists()
    assert not (root / "runs" / (reg.COVERING + "-v2")).exists()
    with pytest.raises(RuntimeError, match="journal/proof"):
        reg.execute(root, record)


@pytest.mark.parametrize("mutation,match", [
    ("queue", "queue changed"), ("registry", "registry changed"), ("pauses", "pauses changed"),
    ("checkpoint", "inputs changed"), ("output", "output appeared"), ("old_output", "source output appeared"),
])
def test_execute_refuses_stale_plan_without_publishing(campaign, monkeypatch, mutation, match):
    root, _, _ = campaign
    record = prepare(root)
    allow_fixture_publication(root, record, monkeypatch)
    paths = {"queue": root / "experiments/queue.yaml", "registry": root / "experiments/registry.jsonl",
             "pauses": root / reg.PAUSES}
    if mutation in paths:
        paths[mutation].write_bytes(paths[mutation].read_bytes() + b"\n")
    elif mutation == "checkpoint":
        next((root / "runs").rglob("step_0400000.pt")).write_bytes(b"changed checkpoint")
    else:
        (root / "runs" / (PICK if mutation == "old_output" else reg.COVERING + "-v2")).mkdir()
    before = snapshot(root)
    with pytest.raises(RuntimeError, match=match):
        reg.execute(root, record)
    assert (root / "experiments/queue.yaml").read_bytes() == before["experiments/queue.yaml"]
    assert (root / reg.PAUSES).read_bytes() == before[reg.PAUSES]
    assert not (root / f"runs/setup/{reg.STEM}.pending.json").exists()


def test_interruption_after_holds_leaves_originals_paused_and_pending_journal(campaign, monkeypatch):
    root, before, _ = campaign
    record = prepare(root)
    allow_fixture_publication(root, record, monkeypatch)

    def interrupted(*args):
        raise RuntimeError("interrupted after holds")

    monkeypatch.setattr(reg, "require_holds", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        reg.execute(root, record)
    assert reg.R.queue_document((root / "experiments/queue.yaml").read_bytes()) == before
    assert (root / f"runs/setup/{reg.STEM}.pending.json").is_file()
    holds = json.loads((root / reg.PAUSES).read_bytes())["pauses"]
    assert any(h["name"] == reg.OBSOLETE_PAUSE and h["active"] and h["ids"] == [PICK] for h in holds)


def test_attempted_other_evaluations_are_preserved(campaign):
    root, _, _ = campaign
    path = root / "experiments/registry.jsonl"
    path.write_text(path.read_text() + json.dumps(dict(id=PICK, event="failed", time=2)) + "\n")
    record = prepare(root)
    assert [r["id"] for r in record["rows"]] == [reg.COVERING + "-v2"]
    assert record["superseded_ids"] == []
    assert record["skipped"][0]["id"] == PICK


@pytest.mark.parametrize("kind", ["existing_id", "existing_output", "symlink", "not_failed", "not_done", "pending"])
def test_prepare_fails_closed(campaign, kind):
    root, doc, _ = campaign
    if kind == "existing_id":
        row = copy.deepcopy(doc["jobs"][0])
        row.update(id=reg.COVERING + "-v2", script=reg.NEW_SCRIPT)
        doc["jobs"].append(row)
        (root / "experiments/queue.yaml").write_text(yaml.safe_dump(doc))
    elif kind in ("existing_output", "symlink"):
        path = root / "runs" / (reg.COVERING + "-v2")
        path.mkdir() if kind == "existing_output" else path.symlink_to(root / "absent")
    elif kind in ("not_failed", "not_done"):
        path = root / "experiments/registry.jsonl"
        events = [json.loads(line) for line in path.read_text().splitlines()]
        events = [e for e in events if e["id"] != (reg.COVERING if kind == "not_failed" else doc["jobs"][0]["deps"][0])]
        path.write_text("".join(json.dumps(e) + "\n" for e in events))
    else:
        (root / f"runs/setup/{reg.STEM}.pending.json").write_text("{}")
    with pytest.raises(RuntimeError):
        prepare(root)


def test_candidate_rejects_bad_original_row_with_scheduler(campaign):
    root, doc, _ = campaign
    doc["jobs"][-1]["ram_gb"] = -1
    (root / "experiments/queue.yaml").write_text(yaml.safe_dump(doc))
    with pytest.raises(ValueError, match="invalid ram_gb"):
        prepare(root)


def test_placement_helper_accepts_only_registered_eval_replacements(campaign):
    spec = importlib.util.spec_from_file_location(
        "place_ours_eval_test", ROOT / "runs/setup/tools/place_ours400k_20261113.py"
    )
    place = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(place)
    root, document, _ = campaign
    rows = document["jobs"] + prepare(root)["rows"]
    by_id = {r["id"]: r for r in rows}
    assert place.valid_id(reg.COVERING + "-v2", by_id) and place.valid_id(PICK + "-v2", by_id)
    assert not place.valid_id(PICK + "-v2")  # A suffix alone cannot authorize placement.
    assert place.valid_id(reg.COVERING)
    assert not place.valid_id("rwv2-20261113-pre400k-ours-covering-S-v2")
    assert not place.valid_id(reg.COVERING + "-v3")
    assert not place.valid_id("../" + reg.COVERING + "-v2")


def test_other_registered_non_drawer_eval_ids_are_replaced_too(campaign):
    root, document, _ = campaign
    source = copy.deepcopy(document["jobs"][1])
    source["id"] = "another-campaign-ours-pick-and-place-eval"
    source["args"][source["args"].index("--name") + 1] = source["id"]
    document["jobs"].append(source)
    (root / "experiments/queue.yaml").write_text(yaml.safe_dump(document))
    record = prepare(root)
    assert record["rows"][-1]["id"] == source["id"] + "-v2"
    assert source["id"] in record["superseded_ids"]


def test_placement_releases_only_moved_replacement_and_retains_old_hold(campaign, monkeypatch):
    root, document, _ = campaign
    record = prepare(root)
    moved = reg.COVERING + "-v2"
    record["rows"][0]["host"] = "local"
    queue = {**document, "jobs": document["jobs"] + record["rows"]}
    (root / "experiments/queue.yaml").write_text(yaml.safe_dump(queue))
    path = root / reg.PAUSES
    path.write_bytes(reg.pause_candidate(path.read_bytes(), record["holds"]))
    spec = importlib.util.spec_from_file_location(
        "place_ours_eval_hold_test", ROOT / "runs/setup/tools/place_ours400k_20261113.py"
    )
    place = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(place)
    monkeypatch.setattr(place.host_roles, "read_ownership", lambda: {moved: dict(owner_host="local")})
    before = json.loads(path.read_bytes())
    assert place.lift_moved(root, [moved]) == [moved]
    expected = copy.deepcopy(before)
    hold = next(h for h in expected["pauses"] if h["name"] == reg.reg.LOCAL_PAUSE)
    hold["ids"].remove(moved)
    hold["rows"] = list(hold["ids"])
    assert json.loads(path.read_bytes()) == expected
    obsolete = next(h for h in expected["pauses"] if h["name"] == reg.OBSOLETE_PAUSE)
    assert obsolete["active"] and obsolete["ids"] == [PICK]


def test_dsreset_registrar_guards_future_length_evals_only(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "dsreset_eval_reg_test", ROOT / "runs/setup/tools/register_dsreset_20261113.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = [old_eval("pick-and-place"), dict(script=reg.OLD_SCRIPT, realworld_target="drawer-open"),
              dict(script="scripts/realworld_dsreset_artifact.py", realworld_target="covering")]
    monkeypatch.setattr(module, "build_reset_rows", lambda: copy.deepcopy(source))
    rows = module.build_rows()
    expected = copy.deepcopy(source)
    expected[0]["script"] = reg.NEW_SCRIPT
    assert rows == expected and source[0]["script"] == reg.OLD_SCRIPT
