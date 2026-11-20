"""Exact-ID BH local source/targeted-acceptance adapter, outside the suite source map.

Imported only by fresh local driver wrappers. No scheduler daemon adoption, queue
mutation, ownership rewrite, reservation or admission-policy change. Scientific
entrypoints run byte-identical authorized source; this operational adapter is separately
hash-pinned in the authorization and recorded in each attempt's provenance.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
from pathlib import Path

MAIN = Path("/home/ws/ws/splatter4d")
AUTHORIZATION = MAIN / "runs/setup/bh-isolated-release.d/local-rgb.json"
STEM = "bh-20261120"
METHODS = (
    "ours_orig",
    "ours_ch01",
    "dynamo",
    "reviwo",
    "sincro",
    "snerl",
    "mvp3d_single",
    "mvp3d_multiview",
    "e2e_cnn",
    "e2e_cnn_noaug",
)
PRE = ("ours_orig", "ours_ch01", "dynamo", "reviwo", "sincro", "snerl", "mvp3d_multiview")


def row_id(kind, method, seed=None):
    return f"{STEM}-{kind}-{method}-ball-hiding-S" + (f"-s{seed}" if seed is not None else "")


IDS = frozenset(
    [row_id("gate", w) for w in ("rgb", "geometry")]
    + [row_id("stats", "ours"), row_id("native", "mvp3d")]
    + [row_id(k, m) for k in ("pre", "export") for m in PRE]
    + [row_id("fqldata", m) for m in METHODS]
    + [row_id(k, m, s) for k in ("fql", "eval") for m in METHODS for s in range(3)]
    + [row_id("taskeval", m) for m in ("ours_orig", "ours_ch01")]
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def regular(path):
    path = Path(path)
    if not path.is_absolute() or path.resolve() != path or not path.is_file():
        raise ValueError(f"not an absolute regular symlink-free file: {path}")
    return path.read_bytes()


def load_authorization(path=AUTHORIZATION):
    """Verify sealed source, exact local row identities and honest focused evidence.

    This explicit owner-scoped acceptance is separate from test_gate full/selective
    acceptance. It never writes docs/tests.json or claims remote native acceptance.
    """
    import re

    raw = regular(path)
    grant = json.loads(raw)
    ids = set(grant.get("rows", {}))
    if (
        grant.get("kind") != "bh-isolated-targeted-v1"
        or grant.get("active") is not True
        or not re.fullmatch(r"[0-9a-f]{40}", str(grant.get("commit", "")))
        or not ids
        or not ids <= IDS
        or not grant.get("accepted_by")
        or grant.get("full_suite_claimed") is not False
        or grant.get("adapter_sha256") != digest(regular(Path(__file__).resolve()))
        or grant.get("worker_sha256") != digest(regular(Path(__file__).resolve().with_name("bh_isolated_worker.py")))
    ):
        raise ValueError("invalid exact-ID BH authorization")
    snapshot_raw = regular(Path(grant["snapshot"]))
    if digest(snapshot_raw) != grant["snapshot_sha256"]:
        raise ValueError("snapshot manifest changed")
    snapshot = json.loads(snapshot_raw)
    root = Path(snapshot["code_root"])
    if root.resolve() != root or root.stat().st_mode & 0o222 or snapshot["commit"] != grant["commit"]:
        raise ValueError("isolated source root is not sealed at the authorized commit")
    for name, spec in snapshot["source_files"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("source path escapes snapshot")
        source = root / relative
        if source.stat().st_mode & 0o222 or digest(regular(source)) != spec["sha256"]:
            raise ValueError(f"isolated source changed: {name}")
    evidence_raw = regular(Path(grant["targeted_evidence"]))
    if digest(evidence_raw) != grant["targeted_evidence_sha256"]:
        raise ValueError("targeted evidence changed")
    evidence = json.loads(evidence_raw)
    if (
        evidence.get("commit") != grant["commit"]
        or evidence.get("ids") != sorted(ids)
        or evidence.get("science_unchanged") is not True
        or not evidence.get("artifacts_sha256")
    ):
        raise ValueError("targeted evidence identity incomplete")
    for name, expected in evidence["artifacts_sha256"].items():
        if digest(regular(Path(name))) != expected:
            raise ValueError("focused evidence artifact changed")
    report = json.loads(regular(Path(evidence["test_report"])))
    counts = report.get("counts", {})
    if (
        report.get("exit_code") != 0
        or counts.get("passed", 0) <= 0
        or any(counts.get(k, -1) != 0 for k in ("failed", "error", "skipped"))
        or evidence["test_report"] not in evidence["artifacts_sha256"]
    ):
        raise ValueError("focused tests did not pass")
    for identity, item in grant["rows"].items():
        row = item["row"]
        if (
            row["id"] != identity
            or row["host"] != "local"
            or row.get("pin_host") != "local"
            or row["planned_commit"] != grant["commit"]
            or row.get("release_source") != "local"
            or row.get("local_placement_pause") != "bh-await-placement-20261120"
            or digest(canonical(row)) != item["row_sha256"]
        ):
            raise ValueError("row authorization differs from registered identity")
    return grant, snapshot, digest(raw)


def check_row(job, grant):
    """A dispatch may bind gpu=any to one instance; nothing else may change."""
    original = grant["rows"][job["id"]]["row"]
    normalized = dict(job)
    if original.get("gpu") == "any":
        normalized["gpu"] = "any"
    if normalized != original:
        raise ValueError(f"{job['id']}: registered row changed")


def fence(path, expected):
    """Cheap revocation/identity fence before localization and launch, no source walk."""
    if digest(regular(path)) != expected:
        raise ValueError("isolated authorization changed or revoked")


def configure_worker_temp(identity):
    """After _bootstrap: short AF_UNIX paths, private to this complete row ID.

    The bootstrap deliberately replaces TMPDIR with its long cache path. Override
    only socket-bearing temp locations after that import; keep all caches intact.
    The full-ID marker rejects truncated-hash collisions and pre-created paths.
    """
    import multiprocessing
    import tempfile

    if identity not in IDS:
        raise ValueError("temp directory requires an exact authorized ID")
    directory = Path("/tmp") / ("bhi-" + digest(identity.encode())[:10])
    try:
        directory.mkdir(mode=0o700)
        created = True
    except FileExistsError:
        created = False
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        info = os.fstat(fd)
        if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o700:
            raise ValueError("unsafe isolated temp directory owner or mode")
        flags = os.O_NOFOLLOW | (os.O_WRONLY | os.O_CREAT | os.O_EXCL if created else os.O_RDONLY)
        marker = os.open("row-id", flags, 0o600, dir_fd=fd)
        with os.fdopen(marker, "wb" if created else "rb") as stream:
            info = os.fstat(stream.fileno())
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.geteuid()
                or stat.S_IMODE(info.st_mode) != 0o600
                or info.st_nlink != 1
            ):
                raise ValueError("unsafe isolated temp identity marker")
            expected = (identity + "\n").encode()
            if created:
                stream.write(expected)
            elif stream.read(len(expected) + 1) != expected:
                raise ValueError("isolated temp directory belongs to another row")
    finally:
        os.close(fd)
    # tempfile can have cached the bootstrap's path during earlier imports.
    for name in ("TMPDIR", "TEMP", "TMP", "XDG_RUNTIME_DIR"):
        os.environ[name] = str(directory)
    tempfile.tempdir = str(directory)
    multiprocessing.current_process()._config.pop("tempdir", None)
    return directory


def install_aliases():
    from ops.jobs import host_roles, realworld_local
    from s4d.rwlearn.pnpv2_spec import WRAPPERS

    aliases = {k.replace("scripts/pnpv2_", "runs/setup/tools/bh_"): v for k, v in WRAPPERS.items()}
    from s4d.rwlearn import dsreset_entrypoints

    for module in (dsreset_entrypoints, realworld_local):
        previous = module.legacy_row
        if getattr(previous, "bh_alias", False):
            continue

        def translated(row, previous=previous):
            script = aliases.get(row.get("script")) if row.get("id") in IDS else None
            return previous({**row, "script": script}) if script else previous(row)

        translated.bh_alias = True
        module.legacy_row = translated
    # Existing strict parser recognizes pnp-v2 config names. Parse a path-only
    # substitute for exact BH taskeval rows, then restore the candidate config.
    previous_arguments = realworld_local._arguments
    if not getattr(previous_arguments, "bh_alias", False):

        def arguments(job):
            if job.get("id") not in IDS or "-taskeval-" not in job["id"]:
                return previous_arguments(job)
            argv = list(job["args"])
            index = argv.index("--config") + 1
            original = argv[index]
            if original not in {
                f"configs/realworld/ours_matrix/ball-hiding-S-{a}-scratch400k.yaml" for a in ("orig", "ch01")
            }:
                raise realworld_local.LocalInputError("unapproved BH evaluation config")
            argv[index] = original.replace("ball-hiding", "pick-and-place-v2")
            parsed = previous_arguments({**job, "args": argv})
            parsed.config = [original]
            return parsed

        arguments.bh_alias = True
        realworld_local._arguments = arguments
    original_refusal = host_roles.refusal
    if not getattr(original_refusal, "bh_alias", False):

        def refusal(job, host, roles, *args, **kwargs):
            if job.get("id") in {row_id("gate", w) for w in ("rgb", "geometry")}:
                # Reuse the existing pnp gate role exception only for these two
                # exact read-only gate identities. Preserve ownership check on BH ID.
                proxy = {
                    **job,
                    "id": job["id"].replace("bh-20261120", "pnpv2-20261117"),
                    "script": "scripts/pnpv2_realworld_dsreset_check.py",
                }
                reason = original_refusal(proxy, host, roles, *args, **kwargs)
                if reason is not None:
                    return reason
                ownership = args[0] if args else kwargs.get("ownership", {})
                return host_roles.ownership_refusal(job["id"], host, ownership)
            return original_refusal(job, host, roles, *args, **kwargs)

        refusal.bh_alias = True
        host_roles.refusal = refusal


def install(path=AUTHORIZATION):
    # Default is a literal no-op, including no imports of production modules.
    if not Path(path).exists():
        return False
    try:
        grant, snapshot, authorization_sha = load_authorization(path)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        # A bad optional grant must not hold other drivers' rows.
        print(f"BH_ISOLATED unavailable: {type(exc).__name__}", file=sys.stderr)
        return False
    from ops.jobs import realworld_local, scheduler

    installed = getattr(scheduler, "_bh_isolated_installed", set())
    if str(path) in installed:
        return True
    active_ids = set(grant["rows"])
    root = Path(snapshot["code_root"])
    original_localize, original_launch, original_record = (realworld_local.localize, scheduler.launch, scheduler.record)
    source = {name: spec["sha256"] for name, spec in snapshot["source_files"].items()}
    execution = {"source_commit": grant["commit"], "files": source, "source_root": str(root)}

    install_aliases()

    def localize(job, state, **kwargs):
        if job["id"] not in active_ids:
            return original_localize(job, state, **kwargs)
        try:
            check_row(job, grant)
            fence(path, authorization_sha)
            if not realworld_local.applies(job, kwargs["roles"], kwargs.get("by_id")):
                raise realworld_local.LocalInputError("existing local resolver/role policy refuses row")
            # Preserve the entire existing input/dependency/closeout resolver. Only
            # repository configuration lookup uses the authorized isolated root.
            effective, record = realworld_local._inputs(
                job,
                state,
                Path(kwargs.get("mirror") or realworld_local.MIRROR),
                Path(kwargs.get("manifest") or realworld_local.MANIFEST),
                Path(kwargs.get("runs") or MAIN / "runs"),
                kwargs.get("remote_prefix", realworld_local.REMOTE_PREFIX),
                root,
                Path(kwargs.get("finished_dir") or realworld_local.FINISHED_DIR),
            )
            return effective, {**record, "execution_scope": execution, "isolated_authorization_sha256": authorization_sha}
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise realworld_local.LocalInputError(str(exc)) from exc

    def launch(job, gpu, attempt, *args, **kwargs):
        if job["id"] not in active_ids:
            return original_launch(job, gpu, attempt, *args, **kwargs)
        check_row(job, grant)
        fence(path, authorization_sha)
        effective = kwargs.pop("args", job.get("args", []))
        wrapped = {**job, "script": "runs/setup/tools/bh_isolated_worker.py"}
        return original_launch(wrapped, gpu, attempt, *args, args=[str(path), job["id"], *map(str, effective)], **kwargs)

    def record(event, *args, **kwargs):
        if event.get("id") in active_ids and event.get("event") == "launched" and event.get("host") == "local":
            event = {
                **event,
                "commit": grant["commit"],
                "source_sha256": source,
                "isolated_source_root": str(root),
                "acceptance_kind": grant["kind"],
                "isolated_authorization_sha256": authorization_sha,
                "execution_adapter_sha256": grant["adapter_sha256"],
            }
        return original_record(event, *args, **kwargs)

    realworld_local.localize = localize
    scheduler.launch, scheduler.record = launch, record
    scheduler._bh_isolated_installed = installed | {str(path)}
    return True


def prepare_worker(path, identity, arguments):
    """Fresh isolated process: immutable science + separately pinned operational gate."""
    grant, snapshot, authorization_sha = load_authorization(path)
    if identity not in grant["rows"]:
        raise ValueError("worker requires one exact authorized ID")
    row = grant["rows"][identity]["row"]
    root = Path(snapshot["code_root"])
    mirror = str(MAIN / "runs/realworld-inputs")
    prefix = "/home/compu/kaist/sunho/realworld"
    # Match precisely the existing local resolver's path-only substitution.
    # Source paths are made absolute below; numeric/scientific arguments remain unchanged.
    expected = []
    for arg in row["args"]:
        value = str(arg).partition("=")[2] if "=" in str(arg) else str(arg)
        # _inputs maps REMOTE_REALWORLD_ROOT, validated again against the isolated constant below.
        expected.append(
            str(arg)[: len(str(arg)) - len(value)] + mirror + value[len(prefix) :]
            if value == prefix or value.startswith(prefix + "/")
            else str(arg)
        )
    if "-fqldata-" in identity or "-native-" in identity:
        expected += ["--raw-root", "/home/compu/kaist/sunho/data/real_world=/home/ws/data/real_world"]
    sys.path[:0] = [str(root / "scripts"), str(root)]
    os.chdir(MAIN)
    os.environ["S4D_HOST_CONFIG"] = str(root / "configs/hosts/local.yaml")
    os.environ["S4D_CACHE_ROOT"] = str(MAIN / ".cache/bh-isolated" / identity)
    from ops.resources import gpu_guard
    from s4d.rwlearn.configs import REMOTE_REALWORLD_ROOT

    if REMOTE_REALWORLD_ROOT != prefix or list(arguments) != expected:
        raise ValueError("worker arguments differ from exact registered/localized argv")
    original_validate = gpu_guard.validate_runtime_path
    own_output = MAIN / "runs" / identity
    own_cache = MAIN / ".cache/bh-isolated" / identity

    def validate_runtime_path(candidate, *, repository=None, cache=False):
        resolved = Path(candidate).expanduser().resolve()
        if (
            resolved == own_output
            or own_output in resolved.parents
            or resolved == own_cache
            or own_cache in resolved.parents
        ):
            return original_validate(candidate, repository=MAIN, cache=cache)
        return original_validate(candidate, repository=repository, cache=cache)

    # Source remains read-only. Grant only this row's ordinary main run directory
    # and private cache through the same unchanged host/path validator.
    gpu_guard.validate_runtime_path = validate_runtime_path
    import _bootstrap

    def require_targeted_tests():
        fence(path, authorization_sha)

    _bootstrap.require_passed_tests = require_targeted_tests
    own_output.mkdir(parents=True, exist_ok=True)
    policy = {
        "science_commit": grant["commit"],
        "source_root": str(root),
        "acceptance_kind": grant["kind"],
        "authorization_sha256": authorization_sha,
        "execution_adapter_sha256": grant["adapter_sha256"],
        "execution_worker_sha256": grant["worker_sha256"],
        "full_suite_claimed": False,
    }
    from ops.jobs.run_identity import _atomic_json

    _atomic_json(own_output / "isolated-execution.json", policy)
    argv = [str(root / arg) if arg.startswith("configs/") else arg for arg in arguments]
    script = root / row["script"]
    return script, argv


def install_all():
    directory = MAIN / "runs/setup/bh-isolated-release.d"
    return [install(path) for path in sorted(directory.glob("*.json"))] if directory.exists() else []
