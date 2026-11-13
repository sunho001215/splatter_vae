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
    # protected heads (each family below its guarantee) start first, then rows in order
    assert sorted(result["admitted"]) == sorted([f"m{i}" for i in range(5)] + [f"r{i}" for i in range(5)])
    assert result["admitted"][:2] == ["m0", "r0"]
    assert rows == before
    for family in result["report"]["families"].values():
        assert family["dominant_share"] == 0.5
        assert family["guarantee"] == {"ram_bytes": 50, "gpu_jobs": 5, "gpu_mem_gb": 50}


def test_one_family_can_use_every_slot_and_all_ram():
    assert select([row(str(i)) for i in range(15)])["admitted"] == [str(i) for i in range(10)]


def test_unclaimed_guarantee_is_borrowed_without_wasting_capacity():
    rows = [row(f"m{i}") for i in range(10)] + [row("r", "realworld", ram=10)]
    result = select(rows)
    assert sorted(result["admitted"]) == sorted([f"m{i}" for i in range(9)] + ["r"])
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
    assert result["admitted"] == ["big", "small0", "small1"]  # the under-served head starts first
    assert result["blocked"]["metaworld"] == "small2"


def test_head_reservation_is_per_family_and_evaluators_keep_their_exception():
    rows = [row("large", ram=60), row("small", ram=1),
            row("eval", ram=5, evaluator=True, trainer_ready=True), row("rw", "realworld", ram=20)]
    result = select(rows, [row("active", ram=45, gpu="mig")], jobs=30)
    assert result["admitted"] == ["rw", "eval"]  # the same-family evaluator still fills (2026-10-29 exception)
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
    # an intrinsically impossible row (pinned to an unknown instance) never blocks its family
    assert result["admitted"] == ["later", "rw"]
    assert result["blocked"] == {"realworld": "rw-later"}


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
    # the impossible head claims nothing; the possible trainer behind it is the family's head and gets its turn
    admitted = select(rows)['admitted']
    assert sorted(admitted) == sorted(['behind-head'] + [f'm{i}' for i in range(9)]) and 'impossible' not in admitted


def test_unready_evaluator_is_not_a_contending_family():
    result = select([row('r', 'realworld', evaluator=True, trainer='absent'), row('m')])
    assert set(result['report']['families']) == {'metaworld'}


def test_driver_integration_uses_all_active_local_jobs_and_preserves_order():
    source = (Path(__file__).resolve().parents[1] / "ops/campaign/baseline_local_driver.py").read_text()
    ast.parse(source)
    assert "fair_share.select(ready_candidates, fair_active, _host['limits'], CAP, total," in source
    assert "print('FAIR_SHARE '" in source
    assert "trainer_blocked = fair['blocked']" in source
    assert "for i, current in state.items() if current.get('host', 'local') == 'local'" in source
    assert "jobs.reservation(current, by_queue.get(i, {}), 'mem_gb')" in source
    assert "return local_footprint(job," in source  # shared with memory_watch remaining growth
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


def test_a_family_below_its_share_cannot_keep_refilling_past_a_less_served_head():
    """Codex review 2 (P1): Meta-World at 70/150 must not overshoot with another 10 while a real-world 80-unit head
    waits; the head fits exactly in the 80 left and starts."""
    active = [row(f"a{i}", ram=10, gpu="mig") for i in range(7)]
    result = select([row("m-next", ram=10), row("rw", "realworld", ram=80)], active, ram=150, jobs=30, mem=300,
                    used=70)
    assert result["admitted"] == ["rw"] and result["blocked"] == {"metaworld": "m-next"}


def test_selected_mig_instances_follow_the_scheduler_rule_and_blocked_heads_reserve_one():
    """Codex review 2 (P1): the instance chosen here is dispatched with the row; a head blocked only by MIG packing
    reserves the instance with the most free memory, so small rows cannot keep every instance fragmented."""
    limits = {f"mig{i}": {"max_jobs": 3, "max_mem_gb": 22} for i in range(8)}
    active = [row(f"a{i}", ram=1, mem=6, gpu=f"mig{i // 2}") for i in range(16)]  # 12 GB on mig0..mig7
    rows = [row("big", "realworld", ram=1, mem=12)] + [row(f"s{i}", ram=1, mem=4) for i in range(4)]
    result = fair.select(rows, active, limits, 1000, 16)
    assert result["blocked"] == {"realworld": "big"} and result["reserved"] == {"mig0": "big"}
    assert set(result["admitted"]) == {"s0", "s1", "s2", "s3"}
    assert "mig0" not in {result["assigned"][i] for i in result["admitted"]}
    assert [result["assigned"][i] for i in result["admitted"]] == ["mig1", "mig2", "mig3", "mig4"]  # least memory
    freed = [a for a in active if a["id"] not in ("a0", "a1")]  # mig0 drains on a later pass
    later = fair.select(rows[:1] + [row("s9", ram=1, mem=4)], freed, limits, 1000, 14)
    assert later["admitted"][0] == "big" and later["assigned"]["big"] == "mig0"


def test_driver_dispatches_the_selected_instance():
    source = (Path(__file__).resolve().parents[1] / "ops/campaign/baseline_local_driver.py").read_text()
    assert "'gpu': fair['assigned'][j['id']]}" in source


def test_protected_head_holds_ram_and_its_instance_against_every_later_pass_refill():
    """Codex review 3: (1) within-share refills of another family cannot take the RAM a protected head still lacks;
    (2) the head's instance is kept from every other row (other families and its own evaluators) at every pass, so it
    only drains; (3) the head starts as soon as enough has drained."""
    active = [row(f"a{i}", ram=10, gpu="mig") for i in range(7)] + [row("b", ram=5, gpu="mig")]  # metaworld at 75
    result = select([row("m-refill", ram=5), row("rw", "realworld", ram=80)], active, ram=150, jobs=30, mem=300,
                    used=75)
    assert result["admitted"] == [] and result["blocked"]["realworld"] == "rw"
    assert result["report"]["held_for_protected_heads"]["realworld"]["ram_bytes"] == 75
    drained = [a for a in active if a["id"] != "b"]  # 70 in use: the head fits in the 80 left, before any refill
    assert select([row("m-refill", ram=5), row("rw", "realworld", ram=80)], drained, ram=150, jobs=30, mem=300,
                  used=70)["admitted"] == ["rw"]
    limits = {f"mig{i}": {"max_jobs": 3, "max_mem_gb": 22} for i in range(8)}
    busy = [row(f"x{i}", ram=1, mem=6, gpu=f"mig{i // 2}") for i in range(16)]
    rows = [row("big", "realworld", ram=1, mem=12), row("own-eval", "realworld", ram=1, mem=4, evaluator=True,
                                                        trainer_ready=True)] + [row(f"s{i}", ram=1, mem=4)
                                                                                for i in range(8)]
    first = fair.select(rows, busy, limits, 1000, 16)
    assert first["reserved"] == {"mig0": "big"}
    assert "mig0" not in {first["assigned"][i] for i in first["admitted"]}  # own evaluator and other family excluded
    one_left = [b for b in busy if b["id"] != "x0"]  # mig0 drains one job: 6 GB in use, the 12 GB head fits
    second = fair.select(rows, one_left, limits, 1000, 15)
    assert second["admitted"][0] == "big" and second["assigned"]["big"] == "mig0"


def test_single_family_keeps_finished_trainer_evaluators_ahead_of_the_next_trainer():
    """Codex review 4: with one free instance slot, the evaluator of a finished trainer (ordered first by the driver)
    starts before the family's next trainer, pass after pass, exactly as before fair share."""
    limits = {f"mig{i}": {"max_jobs": 3, "max_mem_gb": 22} for i in range(8)}
    active = [row(f"a{i}", ram=3.5, mem=6, gpu=f"mig{i // 3}") for i in range(23)]
    rows = [row("eval", ram=4.5, mem=4, evaluator=True, trainer_ready=True), row("next", ram=3.5, mem=6)]
    result = fair.select(rows, active, limits, 140, 23 * 3.5)
    assert result["admitted"] == ["eval"] and result["blocked"] == {"metaworld": "next"}


def test_evaluators_started_before_the_head_are_not_admitted_twice():
    """Codex review 5: 10 ready evaluators then 14 DrM trainers on an empty host (24 slots): every row starts once and
    the reported usage equals the admitted rows."""
    limits = {f"mig{i}": {"max_jobs": 3, "max_mem_gb": 22} for i in range(8)}
    rows = [row(f"e{i}", ram=4.5, mem=4, evaluator=True, trainer_ready=True) for i in range(10)] + [
        row(f"t{i}", ram=3.5, mem=6) for i in range(14)]
    result = fair.select(rows, [], limits, 140, 0)
    assert len(result["admitted"]) == len(set(result["admitted"])) == 24 and not result["blocked"]
    assert set(result["assigned"]) == set(result["admitted"])
    usage = result["report"]["families"]["metaworld"]["usage"]
    assert usage["gpu_jobs"] == 24 and usage["ram_bytes"] == 10 * 4.5 + 14 * 3.5
    slots = {}
    for gpu in result["assigned"].values():
        slots[gpu] = slots.get(gpu, 0) + 1
    assert max(slots.values()) <= 3


def test_random_scenarios_keep_every_hard_invariant():
    """Uniqueness, MIG limits, the RAM cap and instance reservations hold for 300 random mixed scenarios."""
    import random

    rng = random.Random(20261111)
    families = ("metaworld", "realworld", "droid")
    for case in range(300):
        limits = {f"mig{i}": {"max_jobs": 3, "max_mem_gb": 22} for i in range(8)}
        active = [row(f"a{i}", rng.choice(families), ram=rng.choice((3.5, 4.5, 12, 16, 24)),
                      mem=rng.choice((4, 6, 8, 12)), gpu=f"mig{rng.randrange(8)}") for i in range(rng.randrange(12))]
        per = {}
        active = [a for a in active if per.setdefault(a["gpu"], []).append(a) or
                  (len(per[a["gpu"]]) <= 3 and sum(x["mem_gb"] for x in per[a["gpu"]]) <= 22)]
        rows = []
        for i in range(rng.randrange(1, 20)):
            evaluator = rng.random() < 0.3
            rows.append(row(f"c{i}", rng.choice(families), ram=rng.choice((3.5, 4.5, 12, 16, 24, 80)),
                            mem=rng.choice((4, 6, 8, 12, 30)), evaluator=evaluator, trainer_ready=evaluator))
        used = sum(a["ram_bytes"] for a in active)
        result = fair.select(rows, active, limits, 140, used)
        admitted = result["admitted"]
        assert len(admitted) == len(set(admitted)) and set(result["assigned"]) == set(admitted), case
        by_id = {r["id"]: r for r in rows}
        assert used + sum(by_id[i]["ram_bytes"] for i in admitted) <= 140 + 1e-9 or not admitted, case
        for gpu, limit in limits.items():
            here = [a for a in active if a["gpu"] == gpu] + [by_id[i] for i in admitted if result["assigned"][i] == gpu]
            assert len(here) <= limit["max_jobs"] and sum(r["mem_gb"] for r in here) <= limit["max_mem_gb"], case
        for gpu, owner in result["reserved"].items():
            assert all(result["assigned"][i] != gpu or i == owner for i in admitted), case


def test_queue_priority_precedes_family_guarantees_and_prevents_lower_priority_refill():
    rows = [row("old-fql", priority=2), row("ours-fql", "realworld", priority=0),
            row("pretrain", "realworld", priority=-1), row("default")]
    assert select(rows)["admitted"] == ["pretrain", "ours-fql", "old-fql", "default"]
    rows[2]["ram_bytes"] = 80
    busy = [row("running", ram=30, gpu="mig")]
    original = copy.deepcopy(busy)
    for _ in range(3):
        assert select(rows, busy)["admitted"] == []
    assert busy == original  # no preemption or mutation
    assert select(rows)["admitted"][0] == "pretrain"


def test_priority_multireservations_preserve_other_mig_and_host_resources():
    gib = 2**30
    limits = {g: dict(max_jobs=3,max_mem_gb=10 if g != 'm2' else 4) for g in ('m0','m1','m2')}
    active = [row('a0',ram=0,mem=6,gpu='m0'),row('a1',ram=0,mem=6,gpu='m1')]
    rows = [row('h0',priority=-1,ram=2*gib,mem=8,cpu_threads=2),
            row('h1',priority=-1,ram=2*gib,mem=8,cpu_threads=2),
            row('low',priority=2,ram=gib,mem=1,cpu_threads=1),
            row('excess',priority=2,ram=2*gib,mem=1,cpu_threads=1)]
    result = fair.select(rows,active,limits,20*gib,14*gib,cpu_budget=10,cpu_used=4)
    assert result['admitted'] == ['low']
    assert result['assigned'] == {'low':'m2'}
    assert {r['gpu'] for r in result['report']['priority_reservations'].values()} == {'m0','m1'}
