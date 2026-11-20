"""BH2 registration. Preview by default; --apply is Claude-only. No placement or launch.

RGB rows can be appended now HELD; release separately verifies check-data and the
RGB-only publication. Geometry waits for its measured scene so no dummy scene
value can enter a live row. All slow work occurs before nonblocking locks.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
MAIN = Path("/home/ws/ws/splatter4d")
PAUSE = "bh-await-placement-20261120"
LOCKS = ("runs/setup/stage2-driver.lock", "runs/setup/remote-driver.lock", "experiments/registry.lock")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def require(ok, why):
    if not ok:
        raise ValueError(why)


def token(path):
    info = path.stat()
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def atomic(path, raw):
    import tempfile

    path = Path(path)
    fd, name = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        directory = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(name):
            os.unlink(name)


@contextlib.contextmanager
def locks(repo):
    with contextlib.ExitStack() as stack:
        for name in LOCKS:
            stream = stack.enter_context((repo / name).open("a"))
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def setup():
    from s4d.rwlearn.ball_hiding import bootstrap

    bootstrap()


def publication(spec, wave):
    from ops.jobs import pnpv2_registration as reg
    from s4d.rwlearn.pnpv2_arrival import check_data

    report = check_data(spec)
    require(report["passed"], "check-data refused")
    ev = reg.evidence(spec, report)
    require(
        wave in reg.ready_waves(ev),
        f"{wave}: publication not verified: " + str({k: v["detail"] for k, v in ev.items() if v["status"] != "done"}),
    )
    return report, ev


def prepare(repo, spec, release, wave):
    import yaml

    from ops.jobs import pnpv2_registration as reg
    from s4d.rwlearn import bh_training

    queue_path, registry_path = repo / reg.QUEUE, repo / reg.REGISTRY
    before = queue_path.read_bytes()
    registry = registry_path.read_bytes()
    current = yaml.safe_load(before)["jobs"]
    scene, inputs = None, {}
    if wave == "geometry":
        _, ev = publication(spec, wave)
        scene = ev["scene"]["scene"]
        for stage in ev.values():
            inputs.update(stage.get("inputs", {}))
    rows = bh_training.rows(spec, release, waves=(wave,), scene=scene, current=current)
    ids = {r["id"] for r in rows}
    known = {r["id"] for r in current}
    require(not ids & known, "wave already registered")
    require(not any(json.loads(line).get("id") in ids for line in registry.splitlines() if line), "target registry history")
    for row in rows:
        require(not (repo / "runs" / row["id"]).exists(), "existing target output")
        require(not (repo / "runs/setup/ownership" / (row["id"] + ".json")).exists(), "target ownership exists")
        require(all(d in ids or d in known for d in row["deps"]), "missing dependency")
        require((ROOT / row["script"]).is_file(), "missing wrapper")
        merged = reg._merged_config(row)
        if merged:
            from s4d.rwlearn.pnpv2_spec import leftover_tokens

            require(not leftover_tokens(merged[1]), "unresolved config tokens")
    # Preserve the original queue bytes and parsed previous rows, as every reviewed registrar does.
    candidate = reg._reviewed().queue_candidate(before, rows)
    pauses = repo / reg.PAUSES
    pause_before = pauses.read_bytes()
    doc = json.loads(pause_before)
    matches = [h for h in doc["pauses"] if h.get("name") == PAUSE]
    require(len(matches) <= 1, "duplicate BH hold")
    if matches:
        hold = matches[0]
        require(not hold.get("prefixes"), "BH hold must use exact IDs")
    else:
        hold = dict(name=PAUSE, ids=[], prefixes=[], reason="BH exact-ID release/publication verification pending")
        doc["pauses"].append(hold)
    hold["ids"] = list(dict.fromkeys([*hold["ids"], *sorted(ids)]))
    hold["active"] = True
    pause_after = (json.dumps(doc, indent=2) + "\n").encode()
    return dict(
        schema="bh-registration/1",
        wave=wave,
        rows=rows,
        release=release,
        queue_sha256=sha(before),
        registry_sha256=sha(registry),
        pause_sha256=sha(pause_before),
        queue_after_sha256=sha(candidate),
        queue_after=candidate.decode(),
        pause_after=pause_after.decode(),
        inputs=inputs,
        input_tokens={name: token(Path(name)) for name in inputs},
        held=True,
        applied=False,
    )


def apply(repo, record):
    # Recompute priority and entire plan immediately before apply, then CAS under NB locks.
    proof = repo / "runs/setup" / ("bh-20261120-" + record["wave"] + ".registration.json")
    with locks(repo):
        require(not proof.exists(), "registration proof already exists; inspect")
        for name, key in [
            ("experiments/queue.yaml", "queue_sha256"),
            ("experiments/registry.jsonl", "registry_sha256"),
            ("runs/setup/admission-pauses.json", "pause_sha256"),
        ]:
            require(sha((repo / name).read_bytes()) == record[key], name + " changed; rerun")
        require(all(token(Path(p)) == tuple(v) for p, v in record["input_tokens"].items()), "publication evidence changed")
        # Hold first. An interrupted transaction cannot admit any new row.
        atomic(repo / "runs/setup/admission-pauses.json", record["pause_after"].encode())
        atomic(repo / "experiments/queue.yaml", record["queue_after"].encode())
        atomic(proof, (json.dumps({**record, "applied": True}, indent=2) + "\n").encode())
    return {**record, "applied": True}


def main():
    setup()
    from s4d.rwlearn.pnpv2_spec import SPEC, load_spec

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, default=MAIN)
    p.add_argument("--wave", choices=("rgb", "geometry"), required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--apply", action="store_true")
    a = p.parse_args()
    try:
        commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
        import re

        require(re.fullmatch(r"sha256:[0-9a-f]{64}", a.image), "immutable image digest required")
        require(
            not subprocess.check_output(
                ["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"], text=True
            ),
            "candidate tracked source must be clean",
        )
        record = prepare(a.repo, load_spec(ROOT / SPEC), dict(commit=commit, image=a.image, source="local"), a.wave)
        if a.apply:
            require(a.repo.resolve() == MAIN, "apply only to canonical main queue")
            record = apply(a.repo, record)
        print(json.dumps({k: v for k, v in record.items() if k not in ("queue_after", "pause_after")}, indent=2))
    except (OSError, ValueError, KeyError) as exc:
        print("REFUSED: " + str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
