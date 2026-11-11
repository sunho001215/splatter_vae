"""Pure admission checks; campaign sources are only read, never executed."""

import ast
import copy
from pathlib import Path

import pytest

from ops.jobs import fair_share as fair


def row(name, family="metaworld", ram=10, mem=1, **extra):
    return {"id": name, "family": family, "ram_bytes": ram, "mem_gb": mem, **extra}


def select(rows, active=(), ram=100, jobs=10, mem=100, used=0):
    return fair.select(rows, list(active), {"mig": {"max_jobs": jobs, "max_mem_gb": mem}}, ram, used)


def test_two_families_get_equal_guarantees_despite_queue_order():
    rows = [row(f"m{i}") for i in range(10)] + [row(f"r{i}", "realworld") for i in range(10)]
    before = copy.deepcopy(rows)
    result = select(rows)
    assert result["admitted"] == [f"m{i}" for i in range(5)] + [f"r{i}" for i in range(5)]
    assert rows == before
    for family in result["report"]["families"].values():
        assert family["dominant_share"] == 0.5
        assert family["guarantee"] == {"ram_bytes": 50, "gpu_jobs": 5, "gpu_mem_gb": 50}


def test_one_family_can_use_every_slot_and_all_ram():
    assert select([row(str(i)) for i in range(15)])["admitted"] == [str(i) for i in range(10)]


def test_unclaimed_guarantee_is_borrowed_without_wasting_capacity():
    rows = [row(f"m{i}") for i in range(10)] + [row("r", "realworld", ram=10)]
    result = select(rows)
    assert result["admitted"] == [f"m{i}" for i in range(9)] + ["r"]
    assert result["total_ram_bytes"] == 100


@pytest.mark.parametrize("resource", fair.RESOURCES)
def test_each_resource_is_a_dominant_share_and_a_hard_cap(resource):
    capacity = dict.fromkeys(fair.RESOURCES, 100)
    usage = {f: dict.fromkeys(fair.RESOURCES, 0) for f in ("metaworld", "realworld")}
    usage["metaworld"][resource] = 50
    total = dict(usage["metaworld"])
    pending = {"realworld": {resource: 50}}
    assert not fair.permits("metaworld", {resource: 1}, usage, capacity, pending, total)
    assert fair.permits("realworld", {resource: 50}, usage, capacity, pending, total)
    assert not fair.permits("realworld", {resource: 51}, usage, capacity, pending, total)


def test_active_family_counts_even_without_a_ready_candidate():
    result = select([row(f"r{i}", "realworld") for i in range(10)], [row("active", ram=20, gpu="mig")])
    assert len(result["admitted"]) == 8  # active's unused guarantee is borrowable; its usage is charged
    assert set(result["report"]["families"]) == {"metaworld", "realworld"}
    assert result["total_ram_bytes"] == 100


def test_other_familys_small_rows_cannot_starve_a_large_head():
    rows = [row(f"small{i}", ram=5) for i in range(20)] + [row("big", "realworld", ram=50)]
    result = select(rows, [row("active", ram=40, gpu="mig")], jobs=30)
    assert result["admitted"] == ["small0", "small1", "big"]
    assert result["blocked"]["metaworld"] == "small2"


def test_head_reservation_is_per_family_and_evaluators_keep_their_exception():
    rows = [row("large", ram=60), row("small", ram=1),
            row("eval", ram=5, evaluator=True, trainer_ready=True), row("rw", "realworld", ram=20)]
    result = select(rows, [row("active", ram=45, gpu="mig")], jobs=30)
    assert result["admitted"] == ["eval", "rw"]
    assert result["blocked"] == {"metaworld": "large"}


def test_evaluator_requires_active_finished_or_admitted_trainer():
    rows = [row("unready-eval", evaluator=True, trainer="absent"), row("trainer"),
            row("companion", evaluator=True, trainer="trainer"),
            row("finished-eval", evaluator=True, trainer_ready=True)]
    assert select(rows)["admitted"] == ["trainer", "companion", "finished-eval"]


def test_gpu_memory_and_pinned_mig_packing_are_enforced():
    rows = [row("wrong-gpu", gpu="other"), row("later"),
            row("rw", "realworld", mem=8), row("rw-later", "realworld", mem=4)]
    result = select(rows, mem=10)
    assert result["admitted"] == ["rw"]
    assert result["blocked"] == {"metaworld": "wrong-gpu", "realworld": "rw-later"}


def test_unattributed_ram_and_registry_gpu_reservations_consume_capacity():
    result = select([row("r", "realworld", ram=20)], [row("m", ram=5, mem=8, gpu="mig")], mem=10, used=90)
    assert result["admitted"] == []
    assert result["total_ram_bytes"] == 90


def test_three_families_and_empty_pass():
    result = select([row(f, f, ram=30) for f in ("metaworld", "realworld", "droid")], jobs=3)
    assert len(result["admitted"]) == 3
    assert all(f["dominant_share"] == pytest.approx(1 / 3) for f in result["report"]["families"].values())
    assert select([])["report"]["families"] == {}


@pytest.mark.parametrize('impossible', [{'gpu': 'unavailable'}, {'ram': 101}, {'mem': 101}])
def test_capacity_that_no_other_ready_head_can_claim_is_borrowed(impossible):
    rows = [row(f'm{i}') for i in range(10)] + [row('impossible', 'realworld', **impossible),
                                               row('behind-head', 'realworld')]
    assert select(rows)['admitted'] == [f'm{i}' for i in range(10)]


def test_unready_evaluator_is_not_a_contending_family():
    result = select([row('r', 'realworld', evaluator=True, trainer='absent'), row('m')])
    assert set(result['report']['families']) == {'metaworld'}


def test_driver_integration_uses_all_active_local_jobs_and_preserves_order():
    source = (Path(__file__).resolve().parents[1] / "ops/campaign/baseline_local_driver.py").read_text()
    ast.parse(source)
    assert "fair_share.select(fair_candidates, fair_active, queue['hosts']['local']['limits'], CAP, total)" in source
    assert "print('FAIR_SHARE '" in source
    assert "trainer_blocked = fair['blocked']" in source
    assert "for i, current in state.items() if current.get('host', 'local') == 'local'" in source
    assert "jobs.reservation(current, by_queue.get(i, {}), 'mem_gb')" in source
    assert "max(MEASURED[job['script']], float(job.get('ram_gb', 0)) * 2**30)" in source
    assert "job_class(job, by_queue).domain or 'metaworld'" in source
    assert source.index("candidates.sort(") < source.index("fair_candidates =")


def test_two_heads_larger_than_their_share_do_not_deadlock_an_empty_machine():
    """Codex review 1 (P1): two families whose head rows each exceed 1/n of the RAM cap; one starts, the other waits
    for capacity (it is not starved by the guarantee rule itself)."""
    result = select([row("a", "metaworld", ram=80), row("b", "realworld", ram=80)], ram=150)
    assert result["admitted"] == ["a"] and result["blocked"] == {"realworld": "b"}
    later = select([row("b", "realworld", ram=80), row("a2", "metaworld", ram=10)],
                   active=[row("a", "metaworld", ram=80, gpu="mig")], ram=150, used=80)
    assert later["admitted"] == [] or "b" not in later["admitted"]  # b still does not fit beside a (80 + 80 > 150)
    assert "a2" not in later["admitted"]  # metaworld is over its share and b's claimable guarantee is protected
