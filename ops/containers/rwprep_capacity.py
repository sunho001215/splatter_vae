"""Trusted MIG capacity input for the external rwprep launcher.

The launcher embeds SOURCE and this input under admission.lock, after its episode/name
checks. Never take the source, allowlist, or test_paths from a submitted job. The
integration patch lives in the reboot report directory because pre/tools is outside
this task's writable scope.
"""
from __future__ import annotations

from ops.containers.slot_guard import SOURCE
from ops.jobs import release_reservation
from ops.resources.gpu_guard import REMOTE_MIG_INSTANCES
from ops.resources.mig_inventory import queue_limits

ROOT = "/home/compu/kaist/sunho"


def guard_payload(job: dict) -> dict:
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
            'ram_bytes': job['ram_bytes'], 'reserve_bytes': 120 * 2**30,
            'disk_bytes': job.get('disk_bytes', 0), 'floor_bytes': 0, 'disk_floor_fraction': 0.15,
            'cpu_threads': threads, 'cpu_thread_budget': 320, 'root': ROOT}
    return {'source': SOURCE, 'input': data}
