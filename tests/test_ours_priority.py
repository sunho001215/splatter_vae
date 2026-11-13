"""Exercise real scheduling policy AST fragments; never import or run campaign entry points."""

import ast
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from ops.jobs.priority import PriorityBarrier

ROOT = Path(__file__).resolve().parents[1]


def tree(path):
    return ast.parse((ROOT / path).read_text())


def run(nodes, names):
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<isolated policy>", "exec"), names)
    return names


def functions(path, names, *wanted):
    return run([n for n in tree(path).body if isinstance(n, ast.FunctionDef) and n.name in wanted], names)


def rows():
    return [dict(id="older-fql", priority=2, script="scripts/train_rl.py", args=[]),
            dict(id="ours-fql", priority=0, script="scripts/train_rl.py", args=[]),
            dict(id="pretrain", priority=-1, script="scripts/train.py", args=[]),
            dict(id="default", script="scripts/train_rl.py", args=[])]


@pytest.mark.parametrize("script", ["scripts/train_ours_continuation.py", "scripts/evaluate_ours_length.py"])
def test_real_scheduler_guard_accepts_new_entry_preamble(script):
    # Compile the scheduler's actual regex and acceptance function, without importing its entry module.
    nodes = [n for n in tree("ops/jobs/scheduler.py").body
             if (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "GUARD_CALL" for t in n.targets))
             or (isinstance(n, ast.FunctionDef) and n.name == "check_guarded")]
    ns = run(nodes, {"re": re, "Path": Path, "REPO": ROOT})
    assert ns["check_guarded"](script) == ROOT / script
    body = tree(script).body
    guard = next(i for i, n in enumerate(body) if isinstance(n, ast.Expr)
                 and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name)
                 and n.value.func.id == "guard_gpus")
    bootstrap = next(i for i, n in enumerate(body) if isinstance(n, ast.ImportFrom) and n.module == "_bootstrap")
    assert bootstrap < guard < next(i for i, n in enumerate(body) if isinstance(n, ast.FunctionDef))


def test_remote_dispatch_and_stage2_keys_put_priority_before_existing_rank():
    ready = rows()
    ns = {"by_id": {r["id"]: r for r in ready}, "state": {}, "remote": ready,
          "jobs": SimpleNamespace(PENDING="pending", ACTIVE={"running"}, DONE="done"),
          "kind": lambda i: "final_pretrain" if i == "pretrain" else "brl",
          "RANK": {"final_pretrain": 10, "brl": 0}, "ENC_PREFIXES": ("enc-",),
          "HYB_PREFIXES": ("hyb-",), "CST_PREFIXES": ("cst-",)}
    functions("ops/campaign/remote_campaign_driver.py", ns, "dispatch_key")
    assert [r["id"] for r in sorted(ready, key=ns["dispatch_key"])] == ["pretrain", "ours-fql", "older-fql", "default"]
    ns.update(status=lambda _: "pending", order=[r["id"] for r in ready])
    functions("ops/campaign/stage2_local_driver.py", ns, "rank", "key")
    assert [r["id"] for r in sorted(ready, key=ns["key"])] == ["pretrain", "ours-fql", "older-fql", "default"]
    # Equal-priority ties retain the old rank, seed, then registration order.
    ready[2]["priority"] = 2
    assert ns["dispatch_key"](ready[0]) < ns["dispatch_key"](ready[2])


def test_baseline_sort_and_pool_unit_sort_use_priority_first():
    ready = rows()
    ns = {"candidates": ready.copy(), "registered": ready, "ENC": "enc-",
          "status": lambda _: "done", "trained_by": lambda j: j["id"], "jobs": SimpleNamespace(DONE="done")}
    nodes = [n for n in tree("ops/campaign/baseline_local_driver.py").body
             if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
             and isinstance(n.value.func, ast.Attribute) and ast.unparse(n.value.func) == "candidates.sort"]
    run(nodes, ns)
    assert [r["id"] for r in ns["candidates"]] == ["pretrain", "ours-fql", "older-fql", "default"]
    ns.update(units=[[r] for r in ready], HYB_PREFIXES=("hyb-",), CST_PREFIXES=("cst-",), ENC_PREFIXES=("enc-",))
    nodes = [n for n in ast.walk(tree("ops/campaign/remote_campaign_driver.py"))
             if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
             and isinstance(n.value.func, ast.Attribute) and ast.unparse(n.value.func) == "units.sort"]
    run(nodes, ns)
    assert [m[0]["id"] for m in ns["units"]] == ["pretrain", "ours-fql", "older-fql", "default"]


@pytest.mark.parametrize("pool_head", [False, True])
@pytest.mark.parametrize("blocked", [None, "gpu", "ram", "cpu"])
def test_remote_and_pool_admission_share_priority_and_no_unsafe_backfill(pool_head, blocked):
    ready = rows()[:3]
    for r in ready:
        r.update(mem_gb=1, ram_gb=1, cpu_threads=1)
    head = ready[2]
    if blocked:
        head[{"gpu": "mem_gb", "ram": "ram_gb", "cpu": "cpu_threads"}[blocked]] = 8
    source = tree("ops/campaign/remote_campaign_driver.py")
    ns = {"queue": {"hosts": {"remote": {"limits": {"gpu": {"max_jobs": 1, "max_mem_gb": 10}}}}},
          "cpu_used": 0, "CPU_BUDGET": 10, "gpu_exclusive": set(), "gpu_jobs": {"gpu": 0},
          "gpu_load": {"gpu": 0.0}, "gpu_mem": {"gpu": 0.0}, "GPU_CAP": {"gpu": 10}, "GPUS": ["gpu"],
          "exclusive": lambda _: False, "load": lambda _: 0.1, "GPU_LOAD_LIMIT": 10,
          "kind": lambda _: "role", "REMOTE_MIG_ACTIVE": True,
          "hyb_reserved": None, "hyb_reserved_for": None, "state": {},
          "jobs": SimpleNamespace(ACTIVE={"running"}, DONE="done", thread_limit=int),
          "candidates": [r for r in sorted(ready, key=lambda j: j["priority"]) if not (pool_head and r is head)],
          "units": [[head]] if pool_head else [], "priority_barrier": PriorityBarrier(), "plan": [],
          "chosen": [], "taken": set(), "reviwo": 0, "MAX_REVIWO": 10,
          "total": 0, "CAP": 10 * 2**30, "ram_gb": lambda j: j["ram_gb"], "with_role": lambda j: j,
          "HYB_PREFIXES": ("hyb-",), "CST_PREFIXES": ("cst-",), "GPU_HEAVY": ("heavy-",),
          "by_id": {r["id"]: r for r in ready}, "messages": []}
    functions("ops/campaign/remote_campaign_driver.py", ns,
              "capacity_fits", "place", "may_use_reserved", "reserve_for_hybrid")
    loop = next(n for n in source.body if isinstance(n, ast.For) and ast.unparse(n.target) == "priority")
    if blocked == "gpu":
        ns["gpu_mem"]["gpu"] = 6
    elif blocked == "ram":
        ns["total"] = 6 * 2**30
    elif blocked == "cpu":
        ns["cpu_used"] = 6
    run([loop], ns)
    selected = [r["id"] for r in ns["chosen"]] + [t[0] for unit in ns["plan"] for t in unit]
    assert selected == ([] if blocked else ["pretrain"])
    if blocked:
        # Once existing work drains, the high-priority head gets the first slot on the next pass.
        ns.update(total=0, cpu_used=0, priority_barrier=PriorityBarrier())
        ns["gpu_mem"]["gpu"] = 0
        run([loop], ns)
        selected = [r["id"] for r in ns["chosen"]] + [t[0] for unit in ns["plan"] for t in unit]
        assert selected == ["pretrain"]


def test_remote_release_batches_do_not_bypass_priority():
    ns = {"chosen": rows(), "groups": {}, "job_release": lambda j: ("A" if j["id"] != "ours-fql" else "B", "image")}
    source = tree("ops/campaign/remote_campaign_driver.py")
    grouping = next(n for n in ast.walk(source) if isinstance(n, ast.For) and ast.unparse(n.target) == "job"
                    and ast.unparse(n.iter) == "chosen" and "groups.setdefault" in ast.unparse(n))
    run([grouping], ns)
    dispatch = next(n for n in ast.walk(source) if isinstance(n, ast.For)
                    and ast.unparse(n.target) == "((priority, commit, image), group)")
    batches = eval(compile(ast.Expression(dispatch.iter), "<release order>", "eval"), ns)
    assert [group[0]["id"] for _, group in batches] == ["pretrain", "ours-fql", "older-fql", "default"]


def test_stage2_capacity_loop_blocks_lower_priority_rows():
    source = tree("ops/campaign/stage2_local_driver.py")
    loop = next(n for n in source.body if isinstance(n, ast.For) and ast.unparse(n.target) == "job"
                and ast.unparse(n.iter) == "candidates")
    ready = sorted(rows()[:3], key=lambda j: j["priority"])
    ns = {"candidates": ready, "priority_barrier": PriorityBarrier(),
          "footprint": lambda j: 20 if j["priority"] == -1 else 1,
          "total": 0, "CAP": 10, "admitted": [], "taken": set()}
    run([loop], ns)
    assert ns["admitted"] == []


def test_baseline_yield_cannot_put_stage2_ahead_of_ours():
    ready = rows()[:3]
    stage2 = dict(id="stage2", priority=2)
    ns = {"stage2_waiting": ["stage2"], "messages": [], "candidates": ready,
          "by_queue": {"stage2": stage2, **{j["id"]: j for j in ready}}}
    node = next(n for n in tree("ops/campaign/baseline_local_driver.py").body
                if isinstance(n, ast.If) and ast.unparse(n.test) == "stage2_waiting")
    run([node], ns)
    assert [j["id"] for j in ns["candidates"]] == ["ours-fql", "pretrain"]


def test_stage2_defers_to_other_drivers_higher_priority_ready_rows():
    # Execute the actual cross-driver ceiling; pause/role/dependency gates stay under the existing helpers.
    ready = rows()[:3]
    source = tree("ops/campaign/stage2_local_driver.py")
    start = next(i for i, n in enumerate(source.body) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "_own" for t in n.targets))
    ns = {"candidates": [ready[0]], "queue": {"jobs": ready}, "eligible_view": {},
          "jobs": SimpleNamespace(eligible=lambda rows, _: rows), "paused": lambda _: False,
          "host_roles": SimpleNamespace(refusal=lambda *args: None), "_roles": {}, "_ownership": {}, "_context": {}}
    run(source.body[start:start + 3], ns)
    assert ns["candidates"] == []


def test_release_tick_capacity_refusal_holds_lower_priority_batches():
    source = tree("ops/campaign/remote_campaign_driver.py")
    dispatch = next(n for n in ast.walk(source) if isinstance(n, ast.For)
                    and ast.unparse(n.target) == "((priority, commit, image), group)")
    check = dispatch.body[-1]
    assert isinstance(check, ast.If)
    ns = {"observed": {}, "group": [dict(id="pretrain", priority=-1)], "dispatch_barrier": PriorityBarrier(),
          "jobs": SimpleNamespace(ACTIVE={"running", "launching", "sync_pending"}, DONE="done")}
    run([check], ns)
    assert not ns["dispatch_barrier"].allows(dict(priority=0))
    assert not ns["dispatch_barrier"].allows(dict(priority=2))


def test_failed_higher_priority_pool_claim_holds_lower_priority_remote_dispatch():
    source = tree("ops/campaign/remote_campaign_driver.py")
    start = next(i for i, n in enumerate(source.body) if isinstance(n, ast.If) and ast.unparse(n.test) == "plan")
    end = next(i for i, n in enumerate(source.body) if isinstance(n, ast.FunctionDef) and n.name == "job_release")
    low = dict(id="older-fql", priority=2, gpu="gpu", mem_gb=1, ram_gb=1, cpu_threads=1)
    high = dict(id="pretrain", priority=-1)
    ns = {"plan": [[("pretrain", "gpu", 0.1, 1, 2**30)]], "pool_placement": SimpleNamespace(move=lambda *a: []),
          "messages": [], "priority_barrier": PriorityBarrier(), "by_id": {"pretrain": high}, "chosen": [low],
          "cpu_used": 2, "gpu_jobs": {"gpu": 2}, "gpu_load": {"gpu": 0.2}, "gpu_mem": {"gpu": 2},
          "total": 2 * 2**30, "taken": {"older-fql"}, "exclusive": lambda _: False,
          "load": lambda _: 0.1, "ram_gb": lambda r: r["ram_gb"], "jobs": SimpleNamespace(thread_limit=int)}
    run(source.body[start:end], ns)
    assert ns["chosen"] == [] and ns["total"] == 2**30
    assert not ns["taken"] and ns["gpu_jobs"]["gpu"] == 1
