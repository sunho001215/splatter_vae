"""Pure byte-based MIG admission shared by placement and the in-lock guard.

Queue ``*_gb`` values are GiB; framebuffer readings are MiB. Every live owned
container contributes its enforced RAM/CPU limits and declared disk reservation.
GPU reservations charge max(declared, measured), pending growth their positive
difference. No extra placement margin is added to the guard's capacity headroom.
SOURCE embeds these exact functions in the isolated remote Python interpreter.
"""
import inspect
import math
import re


def integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError("invalid " + name)
    return value


def decimal(value, name):
    if not isinstance(value, str) or not re.fullmatch(r"0|[1-9][0-9]*", value):
        raise ValueError("unreadable " + name)
    return int(value)


def tenant_totals(uuid, processes, records):
    """Checked live owned (inspect, PID set, anon+shmem bytes) records."""
    pending_ram = reserved_disk = count = gpu_reserved = gpu_pending = cpu_used = 0
    known = set()
    tenant_exclusive = False
    for item, pids, ram_used in records:
        labels = item['Config']['Labels']
        if labels.get('s4d.project') != 'splatter4d' or labels.get('s4d.owner') != 'kaist-sunho':
            raise ValueError('container ownership changed during inspection')
        state = item['State']
        if not (state.get('Running') or state.get('Restarting') or state.get('Paused')
                or state.get('Status') in ('running', 'restarting', 'paused')):
            continue
        config = item['HostConfig']
        if uuid is not None and labels.get('s4d.gpu') == uuid:
            count += 1
            tenant_exclusive |= labels.get('s4d.exclusive_gpu') == 'true' or labels.get('s4d.role') == 'migval'
            if labels.get('s4d.role') != 'migval':
                managed = labels.get('s4d.managed') == 'true' and labels.get('s4d.host') == 'remote'
                cotenant = labels.get('s4d.managed') == 'false' and labels.get('s4d.role') in ('rwprep', 'rwprep-study', 'rwprep-maint')
                if not (managed or cotenant):
                    raise ValueError('unidentified owned GPU tenant')
                requests = config.get('DeviceRequests') or []
                request = requests[0] if len(requests) == 1 else {}
                if (config.get('Devices') or request.get('Driver') != 'cdi' or request.get('Count', 0) != 0
                        or request.get('Capabilities') or request.get('Options')
                        or request.get('DeviceIDs') != ['nvidia.com/gpu=' + uuid]):
                    raise ValueError('GPU tenant binding is not exact CDI')
                declared = decimal(labels.get('s4d.mem_bytes'), 'GPU reservation')
                if declared <= 0 or pids is None:
                    raise ValueError('missing GPU reservation/process evidence')
                if known & pids:
                    raise ValueError('ambiguous process ownership')
                known |= pids
                measured = sum(processes.get(pid, 0) for pid in pids)
                gpu_reserved += max(declared, measured)
                gpu_pending += max(0, declared - measured)
        limit = integer(config['Memory'], 'container RAM limit', 1)
        cpu_used += integer(config.get('NanoCpus'), 'container CPU quota', 1)
        pending_ram += max(0, limit - integer(ram_used, 'cgroup anon+shmem'))
        if ((labels.get('s4d.managed') == 'true' and labels.get('s4d.host') == 'remote')
                or (labels.get('s4d.managed') == 'false'
                    and labels.get('s4d.role') in ('rwprep', 'rwprep-study', 'rwprep-maint'))):
            reserved_disk += decimal(labels.get('s4d.disk_bytes'), 'container disk reservation')
    return dict(pending_ram=pending_ram, reserved_disk=reserved_disk, count=count,
                gpu_reserved=gpu_reserved, gpu_pending=gpu_pending, cpu_used=cpu_used,
                known=known, tenant_exclusive=tenant_exclusive)


def gpu_reasons(total, used, need, capacity, reserved, pending, count, max_jobs,
                exclusive=False, tenant_exclusive=False, foreign=False, idle_mib=256):
    """All byte-exact guard GPU conditions; total/used alone are in MiB."""
    if total <= 0 or not 0 <= used <= total or capacity > total * 2**20:
        raise ValueError('declared capacity exceeds fresh framebuffer or invalid framebuffer')
    reasons = []
    if need > capacity:
        reasons.append('declared GPU need exceeds capacity')
    if foreign:
        reasons.append('foreign processes on instance')
    if not count and used > idle_mib:
        reasons.append('instance framebuffer above idle threshold')
    if (exclusive and count) or tenant_exclusive:
        reasons.append('exclusive GPU placement')
    if reserved + need > capacity:
        reasons.append('GPU reservation sum exceeds capacity')
    headroom = max(2 * 2**30, total * 2**20 - capacity)
    if (total - used) * 2**20 - pending < need + headroom:
        reasons.append('measured free GPU memory/headroom')
    if count >= max_jobs:
        reasons.append('slot container limit reached')
    return reasons


def host_reasons(available, pending_ram, ram, reserve, cpu_used, cpu_need, cpu_budget,
                 disk_total, disk_free, reserved_disk, disk_need, floor, fraction):
    """Host guard conditions, all memory/disk in bytes and CPU in nanocpus."""
    reasons = []
    if cpu_used + cpu_need > cpu_budget:
        reasons.append('host CPU thread budget')
    if available - pending_ram < reserve + ram:
        reasons.append('host RAM reserve')
    if disk_free - reserved_disk - disk_need < max(floor, math.ceil(disk_total * fraction)):
        reasons.append('disk floor')
    return reasons


SOURCE = 'import math\nimport re\n' + '\n\n'.join(inspect.getsource(function) for function in (
    integer, decimal, tenant_totals, gpu_reasons, host_reasons))
