"""Trusted MIG capacity input for the external rwprep launcher.

The launcher embeds SOURCE and this input under admission.lock, after its episode/name
checks. Never take the source, allowlist, or test_paths from a submitted job. The
integration patch lives in the reboot report directory because pre/tools is outside
this task's writable scope.
"""
from __future__ import annotations

import math
from pathlib import Path

import yaml

from ops.containers.slot_guard import SOURCE
from ops.jobs import release_reservation
from ops.resources.gpu_guard import REMOTE_MIG_INSTANCES
from ops.resources.mig_inventory import queue_limits

ROOT = "/home/compu/kaist/sunho"
QUEUE = Path(__file__).resolve().parents[2] / "experiments/queue.yaml"
# Defaults identical to the scheduler (ops/jobs/scheduler.py REMOTE_CPU_THREAD_BUDGET, hosts.remote.host_ram_reserve_gb).
DEFAULT_CPU_THREAD_BUDGET = 320
DEFAULT_HOST_RAM_RESERVE_GIB = 120


def remote_host_policy(queue_path: Path = QUEUE) -> dict:
    """The remote host's CPU-thread quota budget and host RAM reserve, read from the SAME queue configuration
    (experiments/queue.yaml hosts.remote) the scheduler and its in-lock guard use, so rwprep and the scheduler admit
    against one budget (2026-11-12: a hard-coded 320 here could diverge from the scheduler's configured budget).
    Fails closed on an unreadable or invalid value."""
    queue = yaml.safe_load(Path(queue_path).read_text())
    hosts = queue.get("hosts") if isinstance(queue, dict) else None
    host = hosts.get("remote") if isinstance(hosts, dict) else None
    if not isinstance(host, dict):
        raise ValueError("invalid remote host configuration (hosts.remote must be a mapping)")
    budget = host.get("cpu_thread_budget", DEFAULT_CPU_THREAD_BUDGET)
    reserve = host.get("host_ram_reserve_gb", DEFAULT_HOST_RAM_RESERVE_GIB)
    if type(budget) is not int or not 1 <= budget <= 1024:
        raise ValueError("invalid remote cpu_thread_budget")
    # the same floor as ops/containers/launch.validate_spec (host RAM reserve >= 120 GiB)
    if (type(reserve) not in (int, float) or isinstance(reserve, bool) or not math.isfinite(reserve)
            or reserve < DEFAULT_HOST_RAM_RESERVE_GIB):
        raise ValueError("invalid remote host_ram_reserve_gb")
    return {"cpu_thread_budget": budget, "reserve_bytes": math.ceil(reserve * 2**30)}


def guard_payload(job: dict, queue_path: Path = QUEUE) -> dict:
    policy = remote_host_policy(queue_path)
    instances = REMOTE_MIG_INSTANCES or ()
    gpu = job['gpu']
    item = next((item for item in instances if item['uuid'] == gpu), None)
    if item is None:
        raise ValueError('rwprep MIG UUID is not allowlisted')
    if gpu in release_reservation.reserved_gpus():
        raise ValueError('rwprep GPU is reserved for release verification')
    limit = queue_limits(instances)[gpu]
    max_jobs = job.get('max_per_gpu', limit['max_jobs'])
    cap = job.get('gpu_cap_bytes', int(limit['max_mem_gb'] * 2**30))
    if type(max_jobs) is not int or not 1 <= max_jobs <= limit['max_jobs']:
        raise ValueError('invalid rwprep MIG job cap')
    if type(cap) is not int or not 0 < cap <= limit['max_mem_gb'] * 2**30:
        raise ValueError('invalid rwprep MIG memory cap')
    if type(job.get('exclusive_gpu', False)) is not bool:
        raise ValueError('invalid exclusive_gpu')
    threads = job['cpus']
    if type(threads) is not int or not 1 <= threads <= 64:
        raise ValueError('invalid rwprep CPU quota')
    data = {'uuid': gpu, 'parent': item['parent'], 'gpu_instance_id': item['gpu_instance_id'],
            'device': item['device'], 'max_jobs': max_jobs, 'exclusive_gpu': job.get('exclusive_gpu', False),
            'idle_mib': 256, 'gpu_bytes': job['gpu_need_bytes'], 'capacity_bytes': cap,
            'ram_bytes': job['ram_bytes'], 'reserve_bytes': policy['reserve_bytes'],
            'disk_bytes': job.get('disk_bytes', 0), 'floor_bytes': 0, 'disk_floor_fraction': 0.15,
            'cpu_threads': threads, 'cpu_thread_budget': policy['cpu_thread_budget'], 'root': ROOT}
    return {'source': SOURCE, 'input': data}
