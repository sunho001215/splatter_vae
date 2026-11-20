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
    from ops.jobs import remote_offload
    from runs.setup.tools.bh_isolated_release import fence

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
    return True
