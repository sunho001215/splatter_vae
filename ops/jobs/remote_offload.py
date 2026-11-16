"""Same-ID placement of untouched S rows. Planning is offline; only execute uses SSH.

No release substitution, launch, registry write, pause removal or checkpoint migration.
The two admission drivers and registry are locked only while publishing ownership,
the pool overlay and queue. An interrupted publication fails closed via ownership.
"""
from __future__ import annotations

import contextlib
import copy
import fcntl
import json
import math
import os
import re
import shlex
import tempfile
import time
from pathlib import Path, PurePosixPath

import yaml

from ops.jobs import admission_pause, host_roles, scheduler
from ops.jobs import realworld_local as local
from s4d.rwlearn.dsreset_entrypoints import legacy_row

REMOTE_ROOT = "/home/compu/kaist/sunho"
TOOLS = {
    "runs/setup/tools/register_dsreset_20261113.py",
    "runs/setup/tools/register_drawer_rgb_20261115.py",
    "runs/setup/tools/register_ours_ext_20261113.py",
    "runs/setup/tools/register_ours_matrix_20261115.py",
}
CLASSES = {"realworld_preprocess": 0, "realworld_pretrain": 1, "realworld_fql": 2}


class Refused(ValueError):
    pass


def require(condition, reason):
    if not condition:
        raise Refused(reason)


def stamp(path):
    """Cheap revalidation token, including replacement and in-place writes."""
    path = Path(path)
    require(not any(p.is_symlink() for p in (path, *path.parents)), f"symlink: {path}")
    s = path.stat()
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


def pauses(root):
    path = root / "runs/setup/admission-pauses.json"
    doc = json.loads(path.read_bytes()) if path.exists() else {"pauses": []}
    # Cancellation rows stay excluded even if a historical hold was deactivated.
    result = [p for p in doc["pauses"] if p.get("active") is True
              or 'cancel' in (str(p.get('name', '')) + ' ' + str(p.get('reason', ''))).lower()]
    setup = root / "runs/setup"
    files = [setup / "rwprep-mig-priority.json", *sorted((setup / "rwprep-mig-priority.d").glob("*.json"))]
    for path in files:
        if path.exists() and json.loads(path.read_bytes())["active"] is True:
            result.append(dict(name="rwprep-mig-priority:" + path.stem, prefixes=[""]))
    return result


def valid_s(job):
    identity = job.get("id", "")
    return (identity.startswith(("rwv2-20261113-", "rwv3-20261115-"))
            and re.fullmatch(r"[A-Za-z0-9_.-]+", identity) is not None
            and "-SF" not in identity and "mug" not in json.dumps(job).lower()
            and (re.search(r"-S(?:-|$)", identity) is not None or job.get("realworld_variant") == "S")
            and job.get("realworld_variant", "S") == "S"
            and job.get("registration_tool") in TOOLS)


def pin(job):
    commit, image = job.get("planned_commit", ""), job.get("planned_image", "")
    require(re.fullmatch(r"[0-9a-f]{40}", commit), "full planned_commit required")
    require(re.fullmatch(r"sha256:[0-9a-f]{64}", image), "immutable planned_image required")
    return commit, image


def priority(job, by_id):
    activity = CLASSES.get(host_roles.job_class(job, by_id).cls, 99)
    # Keep the campaign's strict numeric priority tiers within each activity.
    ours = bool(re.search(r"(?:^|-)ours(?:-|$)", job["id"]))
    return activity, not ours, job.get("priority", 5)


def untouched(job, state, ownership, root, holds, roles, by_id, *, remote_host=None):
    require(valid_s(job), "not an approved S row")
    require(job.get("host") == "local", "not local")
    require(job["id"] not in state, "registry history exists; never-launched rows only")
    require(not admission_pause.active_pauses(job["id"], pauses=holds), "intentional admission pause")
    require(not (root / "runs/setup/remote-hold-admissions").exists(), "remote admissions held")
    require(not job.get("continues") and not job.get("verification_purpose"), "continuation/verification")
    commit, _ = pin(job)
    owner = ownership.get(job["id"], {})
    require(owner.get("kind") == "role_placement" and owner.get("state") == "placed"
            and owner.get("owner_host") == "local" and not owner.get("attempts")
            and owner.get("execution_scope", {}).get("source_commit") == commit,
            "exact local placement ownership/pin required")
    for path in host_roles._output_dirs(job, root):
        require(not path.exists() and not path.is_symlink(), f"output exists: {path}")
    require(not list((root / "runs/remote/jobs").glob(job["id"] + "-a*.json")), "remote launch specification exists")
    require(all(scheduler.dependency_ready({**job, "host": "remote"}, state.get(d, {}))
                for d in job.get("deps", [])), "dependencies not complete")
    reason = host_roles.refusal({**job, "host": "remote"}, "remote", roles, {}, by_id)
    require(reason is None, reason)
    require(host_roles.job_class(job, by_id).cls in CLASSES, "not a real-world placement class")
    host = (remote_host if remote_host is not None
            else scheduler.load_queue(root / "experiments/queue.yaml")["hosts"]["remote"])
    require(not host.get('hold'), 'remote queue held')
    require(any(float(job.get("mem_gb", 0)) <= float(limit["max_mem_gb"])
                and int(limit["max_jobs"]) > 0 for limit in host.get("limits", {}).values()),
            "infeasible on empty remote host")


def release_manifest(job, root):
    """Discover only an exact accepted pin, without changing the global role release."""
    commit, image = pin(job)
    options = [f"runs/remote/acceptance/{commit[:7]}-cc/manifest.json"]
    catalog = root / "runs/setup/remote-mig-releases.json"
    if catalog.exists():
        options = [r["manifest"] for r in json.loads(catalog.read_bytes())
                   if (r["commit"], r["image"]) == (commit, image)] + options
    from ops.jobs.remote_backend import acceptance_manifest
    for name in options:
        path = root / name
        if path.is_file():
            evidence = acceptance_manifest(dict(commit=commit, image_digest=image, evidence_manifest=name), root)
            return name, evidence
    raise Refused(f"exact remote acceptance missing for {commit[:7]}; global role release cannot substitute")


def input_files(job, root, state):
    """Byte-exact remote closure. Localized descriptors upload their original bytes.

    Absolute local RGB/native publications cannot be read by the pinned remote
    mounts. Refuse them until a host-portable publication is separately approved;
    rewriting scientific argv or descriptor bytes here would violate the pin.
    """
    row = legacy_row(job)
    require(not job.get("rgb_bundle") or row['script'] == local.DSRESET_ARTIFACT,
            "RGB publication uses local absolute paths; remote mount contract unavailable")
    if row['script'] == 'scripts/train_realworld_fql.py':
        from ops.jobs.arguments import option
        require(option(row.get('args', []), '--name') == job['id']
                and option(row.get('args', []), '--output-root') == 'runs', 'FQL outputs must use own id under runs')
    needed = local.needed_inputs(row)
    checked = {local.REMOTE_PREFIX + "/" + n for n in needed["mirror"]}
    checked |= {f"runs/{dep}/{name}" for dep, name in needed["dependency_files"]}
    checked |= {f"runs/{dep}" for dep, name in needed["dependency_files"] if name == "."}
    require(local._resolved_paths(row, root, absolute=True) <= checked,
            "resolved configuration contains unchecked/local absolute inputs")
    mirror = root / "runs/realworld-inputs"
    manifest = local._manifest(mirror / "MANIFEST.json")
    items, seen = {}, set()

    def add(source, target, expected):
        target = str(PurePosixPath(target))
        require(target.startswith(REMOTE_ROOT + "/") and ".." not in PurePosixPath(target).parts,
                "unsafe remote input target")
        local._regular(source)
        before = stamp(source)
        digest = local._hash(source)
        require(before == stamp(source), f"input changed while hashing: {source}")
        require(expected and digest == expected, f"input identity/closeout mismatch: {source}")
        item = dict(source=str(source), target=target, sha256=digest, bytes=before[2], stat=before)
        require(target not in items or items[target] == item, "conflicting input targets")
        items[target] = item

    def mirrored(name):
        name = local._rel(name)
        if name in seen:
            return
        seen.add(name)
        item = manifest["files"][name]
        source = mirror / ("_remote/" + name if item["kind"] == "localized" else name)
        expected = item.get("remote_sha256", item["sha256"])
        add(source, local.REMOTE_PREFIX + "/" + name, expected)
        for child in item.get("requires", []):
            mirrored(child)
        if source.suffix in (".yaml", ".yml"):
            descriptor = yaml.safe_load(source.read_bytes())
            if isinstance(descriptor, dict) and descriptor.get("kind") == "compact":
                require(not descriptor.get('depth_overlays'), 'native descriptor overlay closure needs portable publication')
                # Every referenced episode must be inventoried, including completion.
                for directory in descriptor.get("paths", []):
                    rel = local._mapped(directory)
                    require(rel is not None, "descriptor contains local absolute paths")
                    for suffix in ("episode.h5", "_COMPLETE.json"):
                        mirrored(rel + "/" + suffix)

    for name in needed["mirror"]:
        mirrored(name)
    for dep, name in needed["dependency_files"]:
        if name == ".":
            continue  # export --run: config and checkpoint are explicit implicit inputs
        local._rel(name)
        current = state[dep]
        if current.get("host") == "local":
            expected = local._closeout(dep, root / "runs/setup/realworld-local-finished")["files"].get(name)
        else:
            expected = (current.get("remote_verified", {}).get("files", {}).get(name)
                        or current.get("sync", {}).get("files", {}).get(name))
        source = root / "runs" / dep / name
        if name.endswith((".yaml", ".yml")) and source.is_file():
            doc = yaml.safe_load(source.read_bytes())
            require(not (isinstance(doc, dict) and doc.get("kind") == "compact"),
                    "dependency descriptor needs a host-portable publication")
        add(source, f"{REMOTE_ROOT}/runs/{dep}/{name}", expected)
    return list(items.values())


def cost(items, inventory, local_wait, remote_start, *, bytes_per_second=6_000_000, queued_bytes=0):
    require(all(math.isfinite(v) and v >= 0 for v in (local_wait, remote_start, queued_bytes))
            and math.isfinite(bytes_per_second) and bytes_per_second > 0, "invalid wait/bandwidth estimate")
    missing = 0
    for item in items:
        cached = inventory.get(item["target"])
        if cached is not None:
            require(cached.get("sha256") == item["sha256"], "remote input conflict: " + item["target"])
        else:
            missing += item["bytes"]
    seconds = (missing + queued_bytes) / bytes_per_second + remote_start
    return dict(missing_bytes=missing, transfer_seconds=missing / bytes_per_second,
                estimated_remote_start_seconds=seconds, expected_local_wait_seconds=local_wait,
                beneficial=seconds < local_wait)


def prepare(job, root, state, ownership, roles, holds, inventory, local_wait, remote_start,
            *, bytes_per_second=6_000_000, queued_bytes=0, queue=None):
    queue = queue or scheduler.load_queue(root / "experiments/queue.yaml")
    by_id = {j["id"]: j for j in queue["jobs"]}
    untouched(job, state, ownership, root, holds, roles, by_id, remote_host=queue['hosts']['remote'])
    items = input_files(job, root, state)
    estimate = cost(items, inventory, local_wait, remote_start,
                    bytes_per_second=bytes_per_second, queued_bytes=queued_bytes)
    manifest, evidence = release_manifest(job, root)
    require(estimate["beneficial"], "transfer + remote start is not sooner than local wait")
    # Acceptance verifies the pinned commit, independent of coordinator source drift.
    paths = {Path(i["source"]) for i in items} | {root / manifest}
    for key in ("full_suite", "cross_host", "data"):
        paths.add(root / evidence[key]["path"])
    paths.update(root / v["path"] for v in evidence["gpu_isolation"].values())
    require(all(stamp(i["source"]) == tuple(i["stat"]) for i in items), "input changed during preparation")
    return dict(job=copy.deepcopy(job), owner=copy.deepcopy(ownership[job["id"]]),
                dependencies={d: copy.deepcopy(state[d]) for d in job.get("deps", [])},
                roles_sha256=roles["sha256"], items=items, estimate=estimate,
                evidence_manifest=manifest, acceptance_sha256=evidence["manifest_sha256"],
                stats={str(p): stamp(p) for p in paths})


def remote_room(job, root, snapshot):
    """Fresh, conservative resource check using the scheduler's guard accounting.

    This is a placement check, not a reservation or admission. The driver still
    applies ready-row priority reservations, load limits and its 1 TB RAM cap.
    A forecast must include the remote queue delay; we require room now as well.
    """
    from ops.jobs import release_reservation
    require(snapshot.resources_ready and not (snapshot.launch_hold or {}).get('held'), 'remote resource/launch hold')
    require(snapshot.admission_records is not None, 'fresh complete remote admission inventory required')
    queue = scheduler.load_queue(root / 'experiments/queue.yaml')
    state = scheduler.read_registry(root / 'experiments/registry.jsonl')
    by_id = {j['id']: j for j in queue['jobs']}
    host = queue['hosts']['remote']
    require(not host.get('hold'), 'remote queue held')
    resource = dict(available=snapshot.available_gb, disk_total=snapshot.disk_total_bytes,
                    disk_free=snapshot.disk_free_bytes, snapshot=snapshot)
    resources = {'remote': resource}
    scheduler._charge_cotenants(resources, state, by_id, [])
    require('remote' in resources and resource.get('guard_accounted'), 'remote accounting unavailable')
    charge = release_reservation.charges(release_reservation.load(root / 'runs/setup/release-reservation.json'))
    resource['available'] -= charge['ram_gb']
    resource['disk_free'] -= int(charge['disk_gb'] * 2**30)
    running = [dict(host='remote', gpu=s['gpu'], mem_gb=scheduler.reservation(s, by_id.get(i, {}), 'mem_gb'),
                    exclusive_gpu=by_id.get(i, {}).get('exclusive_gpu', False))
               for i, s in state.items() if s.get('host') == 'remote' and s['status'] in scheduler.RESERVING]
    running += [dict(host='remote', gpu=g, mem_gb=c['mem_gb'], exclusive_gpu=g.startswith('MIG-'))
                for g, c in charge['gpus'].items()]
    destination = host_roles.destination_row(job, 'remote', '')
    # Charge already waiting, ready remote rows before a new placement. This also
    # accounts for earlier offloads in this pass before the driver launches them.
    owned = host_roles.read_ownership(root / 'runs/setup/ownership')
    roles = host_roles.load_roles(root / 'configs/hosts/roles.yaml')
    holds = pauses(root)
    role_file = root / 'runs/setup/role-release.json'
    role = json.loads(role_file.read_bytes()) if role_file.exists() else {}
    waiting = scheduler.eligible([r for r in queue['jobs'] if r.get('host') == 'remote'], state)
    queued_ram = 0.0
    cpu_used = resource['guard_cpu_used'] + charge['cpu_threads']
    for row in sorted(waiting, key=lambda r: priority(r, by_id)):
        if (row['id'] == job['id'] or admission_pause.active_pauses(row['id'], pauses=holds)
                or host_roles.refusal(row, 'remote', roles, owned, by_id) is not None):
            continue
        if row.get('priority', 5) > job.get('priority', 5) and priority(row, by_id) > priority(job, by_id):
            continue
        proof = admitted_pin(row, owned)
        if row.get('release_source') == 'local' and proof is None:
            continue
        manifest = proof[1] if proof else f"runs/remote/acceptance/{role.get('commit', '')[:7]}-cc/manifest.json"
        if not (root / manifest).is_file():
            continue
        device = scheduler.choose_gpu(row, host['limits'], running, set(), resource.get('cotenant'))
        require(device is not None, 'ready higher-priority remote row is capacity blocked')
        running.append(dict(host='remote', gpu=device, mem_gb=float(row['mem_gb'])))
        from ops.jobs.footprints import ram_gb
        queued_ram += ram_gb(row)
        resource['available'] -= ram_gb(row)
        resource['disk_free'] -= int(float(row.get('disk_gb', 0)) * 2**30)
        cpu_used += scheduler.thread_limit(row.get('cpu_threads', 8))
    # The complete guard inventory includes unknown processes and other users;
    # choose_gpu uses its foreign flag and measured framebuffer headroom.
    gpu = scheduler.choose_gpu(destination, host['limits'], running, set(), resource.get('cotenant'))
    require(gpu is not None, 'no remote GPU room after other tenants and release reservations')
    require(not scheduler.remote_host_refusals(resource, destination, host,
            cpu_used), 'remote RAM/CPU/disk capacity unavailable')
    # Match the remote driver's declaration cap as well as measured free RAM.
    from ops.jobs.footprints import ram_gb
    declared = sum(scheduler.reservation(s, by_id.get(i, {}), 'ram_gb') for i, s in state.items()
                   if s.get('host') == 'remote' and s['status'] in scheduler.RESERVING)
    require((declared + queued_ram + charge['ram_gb'] + ram_gb(job)) * 2**30 <= 1_000_000_000_000,
            'remote campaign RAM cap')
    return gpu


def probe(client, prepared):
    """Fresh absence evidence for output directories and deterministic containers."""
    identity = prepared["job"]["id"]
    output = shlex.quote(f"{REMOTE_ROOT}/runs/{identity}")
    command = (f"test ! -e {output} && test ! -L {output} && "
               "docker ps -a --format '{{.Names}}'")
    names = client.checked_ssh(command, idempotent=True).stdout.splitlines()
    require(not any(n.startswith("s4d-" + identity + "-a") for n in names), "remote container already exists")


def publish(client, item):
    """Resumable staging + atomic no-clobber hard link; never overwrite shared inputs."""
    source, target, digest = Path(item["source"]), item["target"], item["sha256"]
    require(stamp(source) == tuple(item["stat"]), "source changed before upload")
    client.ensure_remote_path(target)
    existing = client._remote_checksum(target, allow_missing=True)
    require(existing in (None, digest), "remote input conflict: " + target)
    if existing == digest:
        return
    # Same parent/filesystem as the final file: link is atomic even when runs and
    # realworld inputs occupy different mounts. rsync's partial is resumable.
    staged = target + '.offload-' + digest
    client.rsync_upload(source, staged)
    require(client._remote_checksum(staged) == digest, "staged checksum mismatch")
    require(stamp(source) == tuple(item["stat"]), "source changed during upload")
    client.ensure_remote_path(target, create_parent=True)
    # A concurrent publisher may win; ln never replaces its inode. Check both outcomes.
    command = ("ln -T -- " + shlex.quote(staged) + " " + shlex.quote(target)
               + " 2>/dev/null || test -f " + shlex.quote(target))
    client.checked_ssh(command, idempotent=True)
    client.ensure_remote_path(target)
    require(client._remote_checksum(target) == digest, "published input conflict/checksum mismatch")


def admitted_pin(job, ownership):
    """Pure driver gate; returns (pin, manifest) only for a matching offload record."""
    owner = ownership.get(job["id"], {})
    record = owner.get("remote_offload", {})
    try:
        exact = pin(job)
    except Refused:
        return None
    if (not valid_s(job) or job.get("host") != "remote" or owner.get("kind") != "role_placement"
            or owner.get("state") != "placed" or owner.get("owner_host") != "remote"
            or record.get("fingerprint") != scheduler.job_fingerprint(job)
            or (record.get("commit"), record.get("image")) != exact
            or not record.get("inputs_verified") or not record.get("evidence_manifest")):
        return None
    return exact, record["evidence_manifest"]


@contextlib.contextmanager
def locks(root):
    with contextlib.ExitStack() as stack:
        for name in ("runs/setup/stage2-driver.lock", "runs/setup/remote-driver.lock", "experiments/registry.lock"):
            stream = stack.enter_context((root / name).open("a"))
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


@contextlib.contextmanager
def stage_rewrite(job, root):
    """Render and validate YAML outside admission locks; compare exact bytes inside."""
    queue_path = root / 'experiments/queue.yaml'
    before = queue_path.read_bytes()
    queue = scheduler.load_queue(queue_path)
    require(queue_path.read_bytes() == before, 'queue changed while preparing rewrite')
    require(next((j for j in queue['jobs'] if j['id'] == job['id']), None) == job, 'row changed before rewrite')
    effective = host_roles.destination_row(job, 'remote', f'remote-offload {host_roles.now()}')
    require(scheduler.job_fingerprint(effective) == scheduler.job_fingerprint(job), 'fingerprint changed')
    rows = [effective if j['id'] == job['id'] else j for j in queue['jobs']]
    text = before.decode()
    node = next(v for k, v in yaml.compose(text).value if k.value == 'jobs')
    after = (text[:node.start_mark.index] + yaml.safe_dump(rows, sort_keys=False) + text[node.end_mark.index:]).encode()
    fd, name = tempfile.mkstemp(prefix='.offload-queue-', suffix='.yaml', dir=queue_path.parent)
    path = Path(name)
    try:
        with os.fdopen(fd, 'wb') as stream:
            os.fchmod(stream.fileno(), queue_path.stat().st_mode & 0o777)
            stream.write(after)
            stream.flush()
            os.fsync(stream.fileno())
        check = scheduler.load_queue(path)
        require(check['jobs'] == rows and check['hosts'] == queue['hosts'], 'invalid staged queue')
        yield dict(path=path, before=before, queue=queue, effective=effective)
    finally:
        path.unlink(missing_ok=True)


def replace_queue(staged, target):
    os.replace(staged, target)


def commit_placement(prepared, root):
    """No hashing, git, SSH or transfer while the three admission locks are held."""
    job, identity = prepared["job"], prepared["job"]["id"]
    require(prepared.get("remote_verified") is True, "fresh remote input/output verification required")
    queue_path = root / "experiments/queue.yaml"
    registry = root / "experiments/registry.jsonl"
    directory = root / "runs/setup/ownership"
    pool_path = root / "runs/setup/pool-placement.json"
    with stage_rewrite(job, root) as staged, locks(root):
        require(queue_path.read_bytes() == staged['before'], 'queue changed after rewrite preparation')
        queue, state = staged['queue'], scheduler.read_registry(registry)
        by_id = {j["id"]: j for j in queue["jobs"]}
        owned = host_roles.read_ownership(directory)
        roles = host_roles.load_roles(root / "configs/hosts/roles.yaml")
        require(by_id.get(identity) == job and owned.get(identity) == prepared["owner"], "row/ownership changed")
        require(roles["sha256"] == prepared["roles_sha256"], "roles changed")
        require(all(state.get(d) == value for d, value in prepared["dependencies"].items()), "dependency state changed")
        untouched(job, state, owned, root, pauses(root), roles, by_id, remote_host=queue['hosts']['remote'])
        require(all(stamp(p) == tuple(token) for p, token in prepared["stats"].items()), "input/acceptance changed")
        timestamp = host_roles.now()
        effective, before = staged['effective'], staged['before']
        backup = root / "runs/setup" / ("offload-backup-" + str(time.time_ns()))
        backup.mkdir()
        # A complete recovery record precedes every operational write.
        (backup / "queue.yaml").write_bytes(before)
        (backup / "ownership.json").write_text(json.dumps(prepared["owner"], indent=2) + "\n")
        if pool_path.exists():
            (backup / "pool-placement.json").write_bytes(pool_path.read_bytes())
        (backup / "transaction.json").write_text(json.dumps(dict(id=identity, original=job, effective=effective,
            estimate=prepared["estimate"], inputs=prepared["items"]), indent=2) + "\n")
        previous = prepared["owner"]
        commit, image = pin(job)
        record = dict(commit=commit, image=image, fingerprint=scheduler.job_fingerprint(job),
                      evidence_manifest=prepared["evidence_manifest"], acceptance_sha256=prepared["acceptance_sha256"],
                      inputs_verified=True, inputs={i["target"]: i["sha256"] for i in prepared["items"]},
                      estimate=prepared["estimate"], backup=str(backup.relative_to(root)))
        # Ownership first: either driver's stale queue snapshot is refused after a crash.
        host_roles.write_ownership(dict(job_id=identity, kind="role_placement", state="placed", owner_host="remote",
            from_host="local", to_host="remote", attempts=0, time=timestamp, reason="earlier resource-aware start",
            execution_scope=previous["execution_scope"], remote_offload=record,
            history=previous.get("history", []) + [{k: v for k, v in previous.items() if k != "history"}]), directory)
        host_roles.append_pool_record(pool_path, dict(time=timestamp, to="remote", ids=[identity],
            reason="earlier resource-aware start", original={identity: job}, backup=str(backup.relative_to(root))),
            {identity: effective})
        replace_queue(staged['path'], queue_path)
    return identity
