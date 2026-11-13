"""Exercise real scheduling policy AST fragments; never import or run campaign entry points."""

import ast
import os
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from ops.jobs.priority import PriorityReservations

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



def reservation_policy(limits=None, *, used=None, free=None, budgets=None, messages=None):
    limits = limits or {g: dict(max_jobs=3, max_mem_gb=10) for g in ('mig0', 'mig1', 'mig2')}
    used = used or {}
    free = free or dict(ram_gb=100, cpu_threads=100, disk_gb=100)
    budgets = budgets or dict(ram_gb=100, cpu_threads=100, disk_gb=100)
    return PriorityReservations(limits, budgets, lambda: used, lambda: free, messages=messages)


def high(name='high', **kw):
    return dict(id=name, priority=-1, gpu='any', mem_gb=8, ram_gb=4, cpu_threads=2, disk_gb=3, **kw)


def test_reservations_use_most_free_memory_then_fewest_jobs_then_device_order():
    used = {'mig0': dict(mem_gb=6, jobs=1), 'mig1': dict(mem_gb=5, jobs=2), 'mig2': dict(mem_gb=5, jobs=1)}
    original = {g: dict(v) for g, v in used.items()}
    policy = reservation_policy(used=used)
    assert [policy.reserve(high(str(i))) for i in range(4)] == ['mig2', 'mig1', 'mig0', None]
    assert used == original
    assert reservation_policy().reserve(high()) == 'mig0'


@pytest.mark.parametrize('key,value', [('mem_gb', 11), ('ram_gb', 101), ('cpu_threads', 101), ('disk_gb', 101),
                                      ('gpu', 'unapproved')])
def test_infeasible_requests_reserve_nothing_and_log_once(key, value):
    messages = []
    policy = reservation_policy(messages=messages)
    row = {**high(), key: value}
    for _ in range(3):
        assert policy.reserve(row) is None
    assert len(messages) == 1 and not policy.reservations
    assert policy.allows(dict(id='low', priority=2, ram_gb=1), 'mig0')


def test_not_ready_rows_reserve_nothing():
    policy = reservation_policy(messages=[])
    assert policy.reserve(high(), ready=False) is None
    assert not policy.reservations
    assert policy.allows(dict(id='low', priority=2), 'mig0')


@pytest.mark.parametrize('resource', ['ram_gb', 'cpu_threads', 'disk_gb'])
def test_cumulative_host_reservations_are_exact(resource):
    free = dict(ram_gb=100, cpu_threads=100, disk_gb=100)
    policy = reservation_policy(free=free)
    first = {**high('first'), resource: 7}
    second = {**high('second'), resource: 11}
    policy.reserve(first)
    policy.reserve(second)
    free[resource] = 20
    low = dict(id='low', priority=2, **{resource: 2})
    assert policy.allows(low, 'mig2')  # exactly 18 remain
    low[resource] = 3
    assert not policy.allows(low, 'mig2')
    assert not policy.allows(dict(id='low', priority=2), 'mig0')
    assert policy.allows({**low, 'priority': -1}, 'mig0')  # unchanged tie behavior
    assert not reservation_policy().reservations  # no state persists to another pass


def remote_context(pool=False, **head_overrides):
    ready = [dict(id='pretrain', priority=-1, gpu='mig0', mem_gb=8, ram_gb=1, cpu_threads=1,
                  disk_gb=1, script='scripts/train.py', args=[]),
             dict(id='other-mig', priority=2, gpu='mig1', mem_gb=1, ram_gb=1, cpu_threads=1,
                  disk_gb=1, script='scripts/train.py', args=[]),
             dict(id='reserved-mig', priority=2, gpu='mig0', mem_gb=1, ram_gb=1, cpu_threads=1,
                  disk_gb=1, script='scripts/train.py', args=[])]
    ready[0].update(head_overrides)
    limits = {g: dict(max_jobs=3, max_mem_gb=10) for g in ('mig0', 'mig1')}
    ns = dict(queue={'hosts': {'remote': {'limits': limits}}}, _host={'limits': limits},
              cpu_used=0, CPU_BUDGET=10, disk_budget=100, disk_free=100,
              gpu_exclusive=set(), gpu_jobs={'mig0': 1, 'mig1': 0}, gpu_load={'mig0': .1, 'mig1': 0},
              gpu_mem={'mig0': 6, 'mig1': 0}, GPU_CAP={g: 10 for g in limits}, GPUS=list(limits),
              exclusive=lambda _: False, load=lambda _: .1, GPU_LOAD_LIMIT=10, kind=lambda _: 'role',
              REMOTE_MIG_ACTIVE=True, hyb_reserved=None, hyb_reserved_for=None, state={},
              jobs=SimpleNamespace(ACTIVE={'running'}, DONE='done', thread_limit=int),
              candidates=ready[1:] if pool else ready, units=[[ready[0]]] if pool else [],
              PriorityReservations=PriorityReservations, plan=[], chosen=[], taken=set(), reviwo=0, MAX_REVIWO=10,
              total=0, reserved=0, _ram_budget=10, _ram_available=10, CAP=10 * 2**30,
              ram_gb=lambda j: j.get('ram_gb', 0), with_role=lambda j: j,
              HYB_PREFIXES=('hyb-',), CST_PREFIXES=('cst-',), GPU_HEAVY=('heavy-',),
              by_id={r['id']: r for r in ready}, messages=[])
    functions('ops/campaign/remote_campaign_driver.py', ns, 'priority_request', 'priority_policy',
              'capacity_fits', 'place', 'may_use_reserved', 'reserve_for_hybrid')
    source = tree('ops/campaign/remote_campaign_driver.py')
    start = next(i for i,n in enumerate(source.body) if isinstance(n, ast.Assign)
                 and any(isinstance(t,ast.Name) and t.id == 'priority_reservations' for t in n.targets))
    end = next(i for i,n in enumerate(source.body) if isinstance(n, ast.If) and ast.unparse(n.test) == 'plan')
    run(source.body[start:end], ns)
    return ns


@pytest.mark.parametrize('pool', [False, True])
def test_remote_and_pool_backfill_other_mig_but_keep_reserved_mig_free(pool):
    ns = remote_context(pool)
    assert [j['id'] for j in ns['chosen']] == ['other-mig']
    assert ns['priority_reservations'].reservations['pretrain']['gpu'] == 'mig0'
    assert ns['gpu_jobs']['mig0'] == 1  # running occupancy was never removed


@pytest.mark.parametrize('pool', [False, True])
@pytest.mark.parametrize('overrides', [dict(mem_gb=80), dict(ram_gb=80), dict(cpu_threads=80), dict(disk_gb=800)])
def test_remote_and_pool_infeasible_head_leaves_all_capacity_open(pool, overrides):
    ns = remote_context(pool, **overrides)
    assert [j['id'] for j in ns['chosen']] == ['other-mig', 'reserved-mig']
    assert not ns['priority_reservations'].reservations


def test_remote_release_batches_keep_priority_order():
    ns = {'chosen': rows(), 'groups': {}, 'job_release': lambda j: ('A' if j['id'] != 'ours-fql' else 'B', 'image')}
    source = tree('ops/campaign/remote_campaign_driver.py')
    grouping = next(n for n in ast.walk(source) if isinstance(n, ast.For) and ast.unparse(n.target) == 'job'
                    and ast.unparse(n.iter) == 'chosen' and 'groups.setdefault' in ast.unparse(n))
    run([grouping], ns)
    dispatch = next(n for n in ast.walk(source) if isinstance(n, ast.For)
                    and ast.unparse(n.target) == '((priority, commit, image), group)')
    batches = eval(compile(ast.Expression(dispatch.iter), '<release order>', 'eval'), ns)
    assert [group[0]['id'] for _, group in batches] == ['pretrain', 'ours-fql', 'older-fql', 'default']


def test_failed_pool_claim_reserves_only_its_mig_and_resources():
    ns = remote_context(True, mem_gb=2)
    ns.update(pool_placement=SimpleNamespace(move=lambda *a: []), ready=lambda _: True, release_ready=lambda _: True)
    ns['jobs'].read_registry = lambda: {}
    source = tree('ops/campaign/remote_campaign_driver.py')
    start = next(i for i,n in enumerate(source.body) if isinstance(n,ast.If) and ast.unparse(n.test) == 'plan')
    end = next(i for i,n in enumerate(source.body) if isinstance(n,ast.FunctionDef) and n.name == 'dispatch_policy')
    run(source.body[start:end], ns)
    # The running mig0 load was included when choosing the waiting claim's best instance (mig1).
    assert [j['id'] for j in ns['chosen']] == ['reserved-mig']
    assert ns['priority_reservations'].reservations['pretrain']['gpu'] == 'mig1'


def stage2_context(*, external=False, **overrides):
    high_row = dict(id='pretrain', priority=-1, gpu='mig0', mem_gb=8, ram_gb=4, cpu_threads=2,
                    disk_gb=3, script='scripts/train.py')
    high_row.update(overrides)
    low = dict(id='low', priority=2, gpu='mig1', mem_gb=1, ram_gb=1, cpu_threads=1,
               disk_gb=1, script='scripts/train.py')
    active = dict(id='running', gpu='mig0', mem_gb=6, ram_gb=1, cpu_threads=1, disk_gb=1,
                  host='local', status='running')
    ns = dict(queue={'hosts': {'local': {'cpu_thread_budget': 10,
                     'limits': {g: dict(max_jobs=3, max_mem_gb=10) for g in ('mig0', 'mig1')}}}},
              state={'running': active}, _context={r['id']: r for r in (high_row, low, active)},
              candidates=[low] if external else [high_row, low], _other_ready=[high_row] if external else [],
              total=2**30, CAP=20 * 2**30, footprint=lambda r: r.get('ram_gb', 0) * 2**30,
              status=lambda _: 'pending', os=os, ROOT=ROOT, messages=[], _roles={}, _ownership={},
              jobs=SimpleNamespace(RESERVING={'running', 'launching'}, ACTIVE={'running'}, DONE='done',
                    reservation=lambda s,r,k: max(s.get(k,0), r.get(k,0)),
                    disk_usage=lambda _: (100 * 2**30, 90 * 2**30)),
              host_roles=SimpleNamespace(remote_resident=lambda *a: False))
    source = tree('ops/campaign/stage2_local_driver.py')
    start = next(i for i,n in enumerate(source.body) if isinstance(n,ast.ImportFrom) and n.module == 'ops.jobs.priority')
    end = next(i for i,n in enumerate(source.body) if isinstance(n,ast.Assign)
               and any(isinstance(t,ast.Tuple) and ast.unparse(t) == '(counts, crop_counts)' for t in n.targets))
    run(source.body[start:end], ns)
    return ns


@pytest.mark.parametrize('external', [False, True])
def test_stage2_backfills_other_instances_behind_own_or_other_drivers_waiting_row(external):
    ns = stage2_context(external=external)
    assert [r['id'] for r in ns['admitted']] == ['low']
    assert ns['assigned'] == {'low': 'mig1'}
    assert ns['priority_reservations'].reservations['pretrain']['gpu'] == 'mig0'


@pytest.mark.parametrize('overrides', [dict(mem_gb=80), dict(ram_gb=80), dict(cpu_threads=80), dict(disk_gb=800)])
def test_stage2_infeasible_rows_from_other_drivers_do_not_hold_capacity(overrides):
    ns = stage2_context(external=True, **overrides)
    assert [r['id'] for r in ns['admitted']] == ['low']
    assert not ns['priority_reservations'].reservations


def test_release_dispatch_recomputes_reservations_from_live_state():
    ns = remote_context()
    fresh = SimpleNamespace(resources_ready=True, available_gb=10, disk_free_bytes=90 * 2**30)
    charges = dict(unbounded=[], cpus=0, gpus={}, ram_pending_bytes=0)
    ns.update(cotenant_charges=lambda *a: charges, owned_cpu_threads=lambda *a: 0,
              RELEASE_HOLD=None, _disk_floor=0)
    functions('ops/campaign/remote_campaign_driver.py', ns, 'dispatch_policy')
    live = dict(running=dict(host='remote', status='running', gpu='mig0', mem_gb=6))
    ns['jobs'].RESERVING = {'running'}
    ns['jobs'].reservation = lambda s,r,k: max(s.get(k,0), r.get(k,0))
    policy = ns['dispatch_policy'](live, SimpleNamespace(snapshot=lambda: fresh))
    row = {**high(), 'gpu': 'mig0'}
    assert policy.reserve(row) == 'mig0'
    assert policy.allows(dict(id='low', priority=2, ram_gb=1, cpu_threads=1, disk_gb=1), 'mig1')
    assert not policy.allows(dict(id='low', priority=2), 'mig0')


def test_release_batch_host_reservations_are_cumulative():
    policy = reservation_policy(free=dict(ram_gb=10,cpu_threads=10,disk_gb=10))
    policy.reserve({**high(),'ram_gb':7})
    rows = [dict(id='first',priority=2,gpu='mig1',ram_gb=3),
            dict(id='second',priority=2,gpu='mig2',ram_gb=1)]
    assert [j['id'] for j in policy.backfill(rows)] == ['first']


def test_baseline_driver_passes_cpu_disk_and_other_drivers_reservations_to_fair_share():
    source = tree('ops/campaign/baseline_local_driver.py')
    high_row = dict(id='high',priority=-1,gpu='mig0',mem_gb=8,ram_gb=2,cpu_threads=2,disk_gb=3,
                    script='scripts/train.py')
    low = dict(id='low',priority=2,gpu='mig1',mem_gb=1,ram_gb=1,cpu_threads=1,disk_gb=1,
               script='scripts/train.py')
    active = dict(id='running',status='running',host='local',gpu='mig0',mem_gb=6,ram_gb=1,cpu_threads=1,
                  disk_gb=1,script='scripts/train.py')
    by_id = {j['id']:j for j in (high_row,low,active)}
    jobs = SimpleNamespace(RESERVING={'running'},ACTIVE={'running'},DONE='done',
                           reservation=lambda s,r,k: max(s.get(k,0),r.get(k,0)),
                           eligible=lambda *a: [high_row,low],disk_usage=lambda _: (100*2**30,90*2**30))
    from ops.jobs import fair_share
    ns = dict(host_roles=SimpleNamespace(job_class=lambda *a: SimpleNamespace(domain='realworld'),
                                         refusal=lambda *a: None,remote_resident=lambda *a: False),
              by_queue=by_id,by_id={low['id']: low},footprint=lambda r:r['ram_gb']*2**30,
              active_footprint=lambda i,c:c['ram_gb']*2**30,pss={},FROZEN=set(),state={'running':active},
              candidates=[low],trained_by=lambda j:j['id'],status=lambda _: 'pending',jobs=jobs,view={},
              queue=dict(jobs=list(by_id.values()),hosts={'local':dict(cpu_thread_budget=10,
                         limits={g:dict(max_jobs=3,max_mem_gb=10) for g in ('mig0','mig1')})}),
              paused=lambda _: False,_owner_ready=lambda _: True,_roles={},_ownership={},_context=by_id,ROOT=ROOT,os=os,
              CAP=20*2**30,total=2**30,messages=[],fair_share=fair_share)
    ns["_owner_ready"].eligible_state = lambda: {}
    start = next(i for i,n in enumerate(source.body) if isinstance(n,ast.FunctionDef) and n.name == 'family')
    end = next(i for i,n in enumerate(source.body) if isinstance(n,ast.Assign)
               and any(isinstance(t,ast.Name) and t.id == 'admitted' for t in n.targets))
    run(source.body[start:end],ns)
    assert ns['fair']['admitted'] == ['low']
    assert ns['fair']['assigned'] == {'low':'mig1'}
    assert ns['fair']['report']['priority_reservations']['high']['gpu'] == 'mig0'


def test_filtered_waiting_rows_are_logged_once_without_reserving():
    from ops.jobs.priority import log_unready

    rows = [dict(id=i) for i in ('deps','paused','mirror','ready','running','failed')]
    state = dict(running=dict(status='running'),failed=dict(status='crashed',failures=2))
    messages = ['mirror: inputs missing']
    for _ in range(2):
        log_unready(rows, {'ready'}, state, messages)
    assert messages == ['mirror: inputs missing','deps: not ready; no priority reservation',
                        'paused: not ready; no priority reservation']


def test_pool_claim_lost_to_running_job_reserves_nothing_on_remote():
    ns = remote_context(True,mem_gb=2)
    ns.update(pool_placement=SimpleNamespace(move=lambda *a: []),ready=lambda _: True,release_ready=lambda _: True)
    ns['jobs'].read_registry = lambda: {'pretrain': dict(status='running',host='local')}
    source = tree('ops/campaign/remote_campaign_driver.py')
    start = next(i for i,n in enumerate(source.body) if isinstance(n,ast.If) and ast.unparse(n.test) == 'plan')
    end = next(i for i,n in enumerate(source.body) if isinstance(n,ast.FunctionDef) and n.name == 'dispatch_policy')
    run(source.body[start:end],ns)
    assert [j['id'] for j in ns['chosen']] == ['other-mig','reserved-mig']
    assert not ns['priority_reservations'].reservations
