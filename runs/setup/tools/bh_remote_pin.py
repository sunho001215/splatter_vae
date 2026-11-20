"""Exact-ID remote pin hook. Requires ORIGINAL native acceptance; grants no acceptance.

Fresh remote driver wrapper reads optional proofs before its lock. Absent, invalid,
revoked or row-mismatched proofs delegate byte-for-byte to the previous pin gate.
No queue/ownership/role-release/remote writes. No widened offload prefix.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

MAIN = Path("/home/ws/ws/splatter4d")
DIRECTORY = MAIN / "runs/setup/bh-remote-pins.d"


def source_root(grant):
    """A sealed candidate supplies script existence without dirtying accepted main."""
    from runs.setup.tools.bh_isolated_release import regular

    snapshot_raw = regular(Path(grant["snapshot"]))
    if hashlib.sha256(snapshot_raw).hexdigest() != grant["snapshot_sha256"]:
        raise ValueError("remote source seal changed")
    snapshot = json.loads(snapshot_raw)
    root = Path(snapshot["code_root"])
    if snapshot["commit"] != grant["commit"] or root.resolve() != root or root.stat().st_mode & 0o222:
        raise ValueError("remote source root is not sealed at the accepted commit")
    for item in grant["rows"].values():
        script = item["row"]["script"]
        if not script.startswith("scripts/bh_") or ".." in Path(script).parts:
            raise ValueError("remote BH requires shipped task scripts")
        path = root / script
        if path.stat().st_mode & 0o222 or hashlib.sha256(regular(path)).hexdigest() != snapshot["source_files"][script]["sha256"]:
            raise ValueError("remote task script differs from seal")
    return root


def install_remote_aliases(ids):
    from ops.jobs import realworld_local, remote_offload
    from s4d.rwlearn import dsreset_entrypoints
    from s4d.rwlearn.pnpv2_spec import WRAPPERS

    aliases = {k.replace("scripts/pnpv2_", "scripts/bh_"): v for k, v in WRAPPERS.items()}
    ids = frozenset(ids)
    for module in (dsreset_entrypoints, realworld_local, remote_offload):
        previous = module.legacy_row

        def translated(row, previous=previous):
            script = aliases.get(row.get("script")) if row.get("id") in ids else None
            return previous({**row, "script": script}) if script else previous(row)

        module.legacy_row = translated


def load(path, repo=MAIN):
    from ops.jobs.remote_backend import acceptance_manifest
    from runs.setup.tools.bh_isolated_release import IDS, canonical, regular

    raw = regular(path)
    grant = json.loads(raw)
    ids = set(grant.get("rows", {}))
    if (
        grant.get("kind") != "bh-remote-native-pin/1"
        or grant.get("active") is not True
        or not ids
        or not ids <= IDS
        or grant.get("adapter_sha256") != hashlib.sha256(regular(Path(__file__).resolve())).hexdigest()
    ):
        raise ValueError("invalid exact-ID remote pin proof")
    name = grant["evidence_manifest"]
    evidence = acceptance_manifest(dict(commit=grant["commit"], image_digest=grant["image"], evidence_manifest=name), repo)
    if evidence["manifest_sha256"] != grant["acceptance_sha256"]:
        raise ValueError("remote native acceptance changed")
    for identity, item in grant["rows"].items():
        row = item["row"]
        if (
            row["id"] != identity
            or row.get("host") != "remote"
            or row.get("pin_host") != "remote"
            or row.get("release_source") != "local"
            or row.get("realworld_target") != "ball-hiding"
            or (row.get("planned_commit"), row.get("planned_image")) != (grant["commit"], grant["image"])
            or hashlib.sha256(canonical(row)).hexdigest() != item["row_sha256"]
        ):
            raise ValueError("remote exact row identity differs")
    source_root(grant)
    return grant, hashlib.sha256(raw).hexdigest()


def install(directory=DIRECTORY):
    if not directory.exists():
        return False
    proofs = {}
    for path in sorted(directory.glob("*.json")):
        try:
            grant, digest = load(path)
            for identity, item in grant["rows"].items():
                if identity in proofs:
                    raise ValueError("overlapping BH pin grants")
                proofs[identity] = (path, grant, digest, item["row"])
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(f"BH_REMOTE unavailable: {type(exc).__name__}", file=sys.stderr)
            return False
    if not proofs:
        return False
    from runs.setup.tools.bh_isolated_release import fence, install_aliases

    install_aliases()  # remote input/role consumers need the same exact-ID wrapper aliases
    install_remote_aliases(proofs)
    from ops.jobs import remote_offload

    previous = remote_offload.admitted_pin

    def admitted_pin(job, ownership):
        if job.get("id") not in proofs:
            return previous(job, ownership)
        path, grant, digest, expected = proofs[job["id"]]
        try:
            fence(path, digest)
            if job != expected:
                return None
            owner = (ownership or {}).get(job["id"])
            if owner and (owner.get("owner_host") != "remote" or owner.get("state") != "placed"):
                return None
            return (grant["commit"], grant["image"]), grant["evidence_manifest"]
        except (OSError, ValueError):
            return None

    remote_offload.admitted_pin = admitted_pin

    from ops.jobs.remote_backend import RemoteBackend
    import copy

    previous_prepare = RemoteBackend.prepare

    def prepare(backend, job, gpu, attempt, host, evidence):
        if job.get("id") not in proofs:
            return previous_prepare(backend, job, gpu, attempt, host, evidence)
        path, grant, digest, expected = proofs[job["id"]]
        fence(path, digest)
        if job != expected or (host["commit"], host["image_digest"]) != (grant["commit"], grant["image"]):
            raise ValueError("remote BH prepare pin/row differs")
        # Only prepare's source-file lookup uses the seal. The same backend/client,
        # canonical runs directory, original launcher and acceptance gates remain.
        scoped = copy.copy(backend)
        scoped.repo = source_root(grant)
        return previous_prepare(scoped, job, gpu, attempt, host, evidence)

    RemoteBackend.prepare = prepare
    return True
