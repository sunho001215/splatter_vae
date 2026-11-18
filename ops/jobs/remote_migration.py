"""Remote admission of a started local-release row migrated to the remote (user directive 2026-11-18 ~13:15: GPU 6/7
jobs continue losslessly on GPU 4/5 or on the remote from a verified checkpoint).

The remote campaign driver (ops/campaign/remote_campaign_driver.py) offers a ``release_source: local`` row only with an
offload pin (never-launched rows, ops/jobs/remote_offload.py) or with a pin from ``admitted_pins`` here, which requires
ALL of (anything else, including an unreadable record or a git / acceptance error, refuses):
  * an ownership record of kind ``migration`` in state switched / verified, owned by and destined to the remote, from
    local (written under the admission locks by runs/setup/tools/migrate_paused_to_remote_20261118.py);
  * the queue row on the remote with the fingerprint of the launched row and, until its first destination attempt, the
    transferred state unchanged (host_roles.migrated_state_refusal);
  * a registry state moved by the scheduler's ``migrated`` event: host remote, ``migrated_from`` naming the record's
    source host, attempt and checkpoint step, commit and image equal to the record's destination release, no re-pin;
  * an ACCEPTED destination role release: listed in runs/setup/role-release.json (current or previous) and its
    acceptance manifest validating (remote_backend.acceptance_manifest);
  * the recorded execution scope of exactly the row's script (files and third-party pins) equal to BOTH the destination
    release commit and the row's launch commit (host_roles.migrated_source_refusal, evaluated on the two commits' git
    objects instead of a working tree).
The scheduler and the job scripts never import this module, so no placed row's execution scope changes.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Callable
from pathlib import Path

from ops.jobs import host_roles
from ops.jobs import scheduler as jobs

REPO = Path(__file__).resolve().parents[2]
ROLE_RELEASE = "runs/setup/role-release.json"
STATES = ("switched", "verified")


class Refused(ValueError):
    pass


def require(condition, reason: str) -> None:
    if not condition:
        raise Refused(reason)


def role_releases(root: Path) -> dict[tuple[str, str], str]:
    """{(commit, image): acceptance manifest path} of the current and previous role releases (the driver's mapping)."""
    doc = json.loads((Path(root) / ROLE_RELEASE).read_text())
    entries = [doc] + list(doc.get("previous", []))
    return {(e["commit"], e["image"]): f"runs/remote/acceptance/{e['commit'][:7]}-cc/manifest.json" for e in entries}


def accepted(commit: str, image: str, manifest: str, root: Path) -> None:
    from ops.jobs.remote_backend import acceptance_manifest

    acceptance_manifest({"commit": commit, "image_digest": image, "evidence_manifest": manifest}, Path(root))


def commit_digests(root: Path, commit: str, paths: list[str]) -> dict[str, str]:
    """{path: sha256 of its blob at ``commit`` or "absent"} in one ``git cat-file --batch``."""
    names = sorted(set(paths))
    request = "".join(f"{commit}:{name}\n" for name in names).encode()
    out = subprocess.run(["git", "-C", str(root), "cat-file", "--batch"], input=request, capture_output=True,
                         check=True, timeout=300).stdout
    found, position = {}, 0
    for name in names:
        end = out.index(b"\n", position)
        header = out[position:end].split()
        if header[-1] == b"missing":
            found[name], position = "absent", end + 1
            continue
        size = int(header[2])
        body = out[end + 1:end + 1 + size]
        found[name] = hashlib.sha256(body).hexdigest() if header[1] == b"blob" else "absent"
        position = end + 1 + size + 1
    return found


def third_party(root: Path, commit: str) -> dict[str, str]:
    from ops.provenance.upstreams import commit_identity

    return commit_identity(Path(root), commit)


def check(job: dict, ownership: dict, current: dict | None, *, root: Path = REPO,
          releases: dict | None = None, accept: Callable = accepted, digests: Callable = commit_digests,
          pins: Callable = third_party) -> tuple[tuple[str, str], str]:
    """((commit, image), acceptance manifest) of an admissible migrated row; raises Refused otherwise."""
    record = (ownership or {}).get(job["id"])
    require(record is not None and record.get("kind") == "migration", "no migration ownership record")
    require(record.get("state") in STATES, f"migration state {record.get('state')} (switched/verified required)")
    require(record.get("owner_host") == "remote" and record.get("to_host") == "remote"
            and record.get("from_host") == "local", "migration is not local -> remote with remote ownership")
    require(job.get("host") == "remote", "queue row is not on the remote")
    member, destination = record.get("member") or {}, record.get("destination") or {}
    commit, image = destination.get("commit"), destination.get("image_digest")
    source = member.get("commit")
    require(member.get("launched") is True and isinstance(source, str) and len(source) == 40,
            "migration record names no launched source commit")
    require(isinstance(commit, str) and len(commit) == 40 and isinstance(image, str) and image.startswith("sha256:"),
            "migration record names no destination release")
    current = current or {}
    moved = current.get("migrated_from") or {}
    require(current.get("host") == "remote" and moved.get("host") == record["from_host"]
            and moved.get("attempt") == member.get("attempt") and record.get("checkpoint_step") is not None
            and moved.get("checkpoint_step") == record["checkpoint_step"],
            "registry state was not moved by the matching migrated event")
    require(current.get("commit") == commit and current.get("image_digest") == image,
            "registry release differs from the migration's destination release")
    require(not current.get("repinned_from"), "release re-pinned after the migration")
    refused = host_roles.migrated_state_refusal(job, ownership, current, jobs.job_fingerprint(job), Path(root))
    require(refused is None, f"transferred state: {refused}")
    releases = role_releases(root) if releases is None else releases
    manifest = releases.get((commit, image))
    require(manifest is not None, f"destination {commit[:7]} is not a role release (runs/setup/role-release.json)")
    try:
        accept(commit, image, manifest, root)
    except Exception as exc:  # noqa: BLE001 (any refusal of the acceptance evidence)
        raise Refused(f"destination {commit[:7]} acceptance invalid: {type(exc).__name__}: {str(exc)[:120]}") from None
    files, third = destination.get("scope_files") or {}, destination.get("scope_third_party") or {}
    require(files and third, "migration record has no execution scope")
    require(destination.get("scripts") == [job["script"]] and destination.get("source_commit") == source,
            "execution scope recorded for another script or source commit")
    for name, revision in (("destination release", commit), ("launch commit", source)):
        found = {**digests(root, revision, list(files)), **pins(root, revision)}
        refused = host_roles.migrated_source_refusal(job["id"], ownership, found, Path(root))
        require(refused is None, f"{name} {revision[:7]}: {refused}")
    return (commit, image), manifest


def admitted_pins(rows: list[dict], ownership: dict, state: dict, root: Path = REPO,
                  **kwargs) -> tuple[dict[str, tuple], dict[str, str]]:
    """({job id: ((commit, image), manifest)}, {job id: refusal}) over the rows that have a migration record to the
    remote; rows without one appear in neither."""
    pins, refused = {}, {}
    for job in rows:
        record = (ownership or {}).get(job["id"]) or {}
        if record.get("kind") != "migration" or record.get("to_host") != "remote":
            continue
        try:
            pins[job["id"]] = check(job, ownership, state.get(job["id"]), root=root, **kwargs)
        except Exception as exc:  # noqa: BLE001 fail closed
            refused[job["id"]] = str(exc) if isinstance(exc, Refused) else f"{type(exc).__name__}: {str(exc)[:160]}"
    return pins, refused
