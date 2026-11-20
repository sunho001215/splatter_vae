"""Read-only arrival detection and dataset validation for pick-and-place-v2 (S only).

Arrival is the operator's verified pull of the Robot-PC recordings into ``<raw>/pick-and-place-v2/success``
(dsreset 2026-11-13 transfer layout). Nothing here opens a network connection, reads or writes the Robot PC, or
writes anywhere: every function only lists, stats, hashes and opens recordings read-only.

A partial upload is never complete: in-flight names (``.rsync-partial``, ``.incoming``, ``*.part``, ``*.tmp``,
dotfiles), a count other than 100, or a moving inventory all keep the state below ``complete``. Completeness needs
either the ``success.VERIFIED`` marker whose local and Robot-PC SHA lists equal the directory, or (without a marker)
an inventory that is unchanged across the settle window and whose newest file is older than that window.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import re
import time
from pathlib import Path

from s4d.rwlearn.pnpv2_spec import STEM, TASK, V1_PAUSE, values

EPISODE = re.compile(r"episode_[0-9]{6}\.h5\Z")
COMPANION = re.compile(r"episode_[0-9]{6}[A-Za-z0-9_.-]*\.(mp4|json)\Z")
PARTIAL = re.compile(r"(\.part|\.tmp|\.partial|~)\Z")
ROBOT_PC = ("192.168.50.20", "robotpc", "irsl")


def _forbid_robot_pc(path):
    text = str(path).lower()
    if any(name in text for name in ROBOT_PC):
        raise ValueError(f"refusing a Robot-PC path: {path}")


def _sha_list(path):
    """``sha256sum`` output -> {filename: sha}; None if absent; malformed lines refuse."""
    path = Path(path)
    if not path.is_file():
        return None
    out = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        m = re.fullmatch(r"([0-9a-f]{64}) [ *](\S+)", line.strip())
        if not m or m[2] in out:
            raise ValueError(f"malformed or duplicate SHA list line in {path.name}")
        out[Path(m[2]).name] = m[1]
    return out


def scan(raw_success):
    """Inventory of the success root (no hashing). Unknown names are reported, never followed."""
    root = Path(raw_success)
    _forbid_robot_pc(root)
    inv = dict(root=str(root), exists=root.is_dir() and not root.is_symlink(), episodes={}, companions=[],
               partial=[], unexpected=[], newest_mtime=None)
    marker_base = root.parent / root.name
    inv["markers"] = {
        name: Path(str(marker_base) + suffix).is_file()
        for name, suffix in (("verified", ".VERIFIED"), ("local_sha256", ".local.sha256"),
                             ("robotpc_sha256", ".robotpc.sha256"))
    }
    if not inv["exists"]:
        return inv
    for p in sorted(root.iterdir()):
        name = p.name
        if name.startswith(".") or PARTIAL.search(name) or name in (".incoming", ".rsync-partial"):
            inv["partial"].append(name)
            continue
        st = p.lstat()
        if p.is_symlink() or not p.is_file():
            inv["unexpected"].append(name)
            continue
        if EPISODE.fullmatch(name):
            inv["episodes"][name] = [st.st_size, st.st_mtime_ns]
        elif COMPANION.fullmatch(name):
            inv["companions"].append(name)
        else:
            inv["unexpected"].append(name)
        inv["newest_mtime"] = max(inv["newest_mtime"] or 0, st.st_mtime_ns)
    return inv


def arrival(spec, *, raw_root=None, now=None, sleep=time.sleep, settle_seconds=None):
    """State of the new dataset: absent | partial | unstable | complete (+ basis marker|stable)."""
    p = values(spec)
    expected = spec["collection"]["expected_success"]
    root = Path(raw_root) / TASK / "success" if raw_root else Path(p["raw_success"])
    settle = spec["arrival"]["stable_seconds"] if settle_seconds is None else settle_seconds
    first = scan(root)
    result = dict(root=str(root), expected=expected, found=len(first["episodes"]), markers=first["markers"],
                  partial=first["partial"], unexpected=first["unexpected"], companions=len(first["companions"]),
                  state="absent", basis=None, problems=[])
    if not first["exists"]:
        result["problems"].append("success root absent")
        return result
    problems = result["problems"]
    if first["partial"]:
        problems.append(f"in-flight upload entries present: {first['partial'][:5]}")
    if first["unexpected"]:
        problems.append(f"unexpected entries: {first['unexpected'][:5]}")
    if len(first["episodes"]) != expected:
        problems.append(f"expected {expected} complete episode files, found {len(first['episodes'])}")
    marker = first["markers"]["verified"]
    if marker:
        lists = [_sha_list(Path(str(root) + s)) for s in (".local.sha256", ".robotpc.sha256")]
        if None in lists:
            problems.append("VERIFIED marker without both local and robotpc SHA lists")
        elif lists[0] != lists[1]:
            problems.append("local and robotpc SHA lists differ")
        elif set(lists[0]) != set(first["episodes"]):
            problems.append("SHA list membership differs from the episode files")
    if problems:
        result["state"] = "partial"
        return result
    if marker:
        result.update(state="complete", basis="marker")
        return result
    # No marker: require an unchanged inventory across the settle window, all files older than the window.
    sleep(settle)
    second = scan(root)
    now_ns = int((time.time() if now is None else now) * 1e9)
    if second["episodes"] != first["episodes"] or second["partial"] or second["unexpected"]:
        result.update(state="unstable")
        problems.append("inventory changed during the settle window")
    elif (now_ns - (second["newest_mtime"] or now_ns)) < settle * 1e9:
        result.update(state="unstable")
        problems.append(f"newest file is younger than the {settle} s settle window")
    else:
        result.update(state="complete", basis="stable")
    return result


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@contextlib.contextmanager
def task_known():
    """paired_manifests only checks membership of dsreset.TASKS; register the new task for this call only."""
    from s4d.rwlearn import dsreset

    before = dsreset.TASKS
    dsreset.TASKS = tuple(dict.fromkeys((*before, TASK)))
    try:
        yield dsreset
    finally:
        dsreset.TASKS = before


def check_data(spec, *, raw_root=None, inspect=None, settle_seconds=None, now=None, sleep=time.sleep):
    """Full read-only validation; returns a report with ``passed`` and the canonical S manifest (or problems)."""
    from s4d.rwlearn.dsreset import inspect_raw

    state = arrival(spec, raw_root=raw_root, settle_seconds=settle_seconds, now=now, sleep=sleep)
    report = dict(schema="pick-and-place-v2-check-data/1", task=TASK, passed=False, arrival=state,
                  problems=list(state["problems"]), episodes=0, split=None, manifest=None, manifest_sha256=None,
                  writes=False, robot_pc_io=False)
    if state["state"] != "complete":
        report["problems"].insert(0, f"dataset not complete ({state['state']})")
        return report
    root = Path(state["root"])
    c = spec["collection"]
    robotpc = _sha_list(Path(str(root) + ".robotpc.sha256"))
    hashes = {}
    for name in sorted(p.name for p in root.iterdir() if EPISODE.fullmatch(p.name)):
        digest = sha256(root / name)
        if robotpc is not None and robotpc.get(name) != digest:
            report["problems"].append(f"{name}: sha256 differs from the Robot-PC list")
        hashes[name] = digest
    dupes = len(hashes) - len(set(hashes.values()))
    if dupes:
        report["problems"].append(f"{dupes} duplicate recording contents")
    failures = Path(state["root"]).parent / "failure"
    if failures.exists():
        report["notes"] = ["failure recordings present; S-only: never read, never in membership"]
    bh_amendment = None
    if TASK == 'ball-hiding':
        from s4d.rwlearn.ball_hiding_membership import amendment, verify_rejection
        bh_amendment = amendment()
        try:
            for name in bh_amendment['excluded']:
                verify_rejection(root / Path(name).name)
        except (ValueError, OSError, KeyError) as exc:
            report['problems'].append(f'exclusion evidence: {exc}')
            return report
    try:
        with task_known() as dsreset:
            manifests = dsreset.paired_manifests(
                TASK, {"success": root}, seed=c["split_seed"], validation_fraction=c["validation_fraction"],
                expected={"success": c["expected_success"]}, require_verified=state["basis"] == "marker",
                inspect=inspect or inspect_raw, variants=("S",), amendment=bh_amendment)
    except (ValueError, OSError, KeyError) as exc:
        report["problems"].append(f"episode validation: {exc}")
        return report
    m = manifests["S"]
    if {r["filename"]: r["sha256"] for r in m["episodes"] + m.get("excluded", [])} != hashes:
        report["problems"].append("recordings changed between hashing and validation")
    if TASK == 'ball-hiding':
        from s4d.rwlearn.ball_hiding_membership import validate_membership
        try:
            validate_membership(m)
        except ValueError as exc:
            report['problems'].append(str(exc))
    elif any(r["label"] != "success" for r in m["episodes"]) or m.get("excluded") or m.get("amendment"):
        report["problems"].append("S membership must be the 100 success recordings, no exclusions/amendment")
    if (len(m["split"]["train"]), len(m["split"]["validation"])) != ((89, 10) if TASK == "ball-hiding" else (90, 10)):
        report["problems"].append("expected the canonical seed-0 episode split (BH 89/10; PNP 90/10)")
    report.update(episodes=len(m["episodes"]), manifest=m, split={k: len(v) for k, v in m["split"].items()},
                  manifest_sha256=hashlib.sha256(json.dumps(m, sort_keys=True).encode()).hexdigest())
    report["passed"] = not report["problems"]
    return report


def v1_collisions(ids, pauses):
    """(v1 hits, other active-pause hits, problems) for v2 identities against admission-pauses.json content.

    A v1 hit is fatal: the discontinued pick-and-place v1 prefixes/IDs must never match (hold or resume) a v2 row.
    """
    problems = []
    holds = [p for p in pauses.get("pauses", []) if p.get("name") == V1_PAUSE]
    if len(holds) != 1 or holds[0].get("active") is not True or not holds[0].get("prefixes"):
        problems.append(f"{V1_PAUSE} must stay active with its prefixes (v1 stays discontinued)")
    v1 = holds[0] if holds else {}

    def matches(pause, i):
        return i in pause.get("ids", []) or any(i.startswith(x) for x in pause.get("prefixes", []))

    v1_hits = sorted(i for i in ids if matches(v1, i))
    other = sorted({p.get("name") for p in pauses.get("pauses", []) if p.get("active") is True
                    and p.get("name") != V1_PAUSE for i in ids if matches(p, i)})
    if any(not i.startswith(STEM + "-") or TASK not in i for i in ids):
        problems.append("every v2 identity must start with the pnpv2 stem and name pick-and-place-v2")
    return v1_hits, other, problems


def v1_unpaused(queue_rows, pauses):
    """Queue rows of the discontinued pick-and-place v1 campaign that its pause would NOT hold (resumable)."""
    holds = [p for p in pauses.get("pauses", []) if p.get("name") == V1_PAUSE and p.get("active") is True]
    hold = holds[0] if holds else {}

    def held(i):
        return i in hold.get("ids", []) or any(i.startswith(x) for x in hold.get("prefixes", []))

    v1 = [r["id"] for r in queue_rows if r.get("realworld_target") == "pick-and-place"
          or ("-pick-and-place" in r["id"] and TASK not in r["id"] and r["id"].startswith("rw"))]
    return sorted(i for i in v1 if not held(i))
