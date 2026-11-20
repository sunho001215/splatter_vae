"""Exact-ID BH release/unhold or rollback. Preview by default; --apply is Claude-only.

Publication + source/native acceptance are checked before nonblocking locks.
Local and remote grants release independently. Rollback holds selected IDs FIRST,
then revokes their grant; it never signals or resets an executing row.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bh_isolated_release as local  # noqa: E402
import bh_register as registration  # noqa: E402
import bh_remote_pin as remote  # noqa: E402


def revision(path):
    info = path.stat()
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def pause_bytes(raw, ids, hold):
    doc = json.loads(raw)
    matches = [p for p in doc["pauses"] if p.get("name") == registration.PAUSE]
    registration.require(len(matches) == 1 and not matches[0].get("prefixes"), "expected exact-ID BH hold")
    entry = matches[0]
    registration.require(set(entry.get("ids", [])) <= local.IDS, "BH hold contains foreign IDs")
    if hold:
        entry["ids"] = list(dict.fromkeys([*entry["ids"], *sorted(ids)]))
    else:
        entry["ids"] = [i for i in entry["ids"] if i not in ids]
    entry["active"] = bool(entry["ids"])
    return (json.dumps(doc, indent=2) + "\n").encode()


def main_acceptance(repo):
    """The existing drivers' own pre-launch predicate, checked before unholding BH."""
    from ops.jobs import scheduler

    previous = scheduler.REPO
    try:
        scheduler.REPO = repo
        identity = scheduler.source_identity()
    finally:
        scheduler.REPO = previous
    report_path = repo / "docs/tests.json"
    report = json.loads(report_path.read_bytes())
    registration.require(
        report.get("full_suite_passed") is True and report.get("source_sha256") == identity["source_sha256"],
        "main driver source/test evidence is stale; BH scoped acceptance cannot replace it",
    )
    return {
        str(report_path): revision(report_path),
        **{str(repo / name): revision(repo / name) for name in identity["source_sha256"] if "@" not in name},
    }


def plan(mode, path, repo=registration.MAIN):
    import yaml

    raw = local.regular(path)
    grant = json.loads(raw)
    ids = set(grant.get("rows", {}))
    registration.require(ids and ids <= local.IDS, "nonempty exact BH IDs required")
    is_local = grant.get("kind") == "bh-isolated-targeted-v1"
    registration.require(is_local or grant.get("kind") == "bh-remote-native-pin/1", "unknown grant type")
    directory = "bh-isolated-release.d" if is_local else "bh-remote-pins.d"
    waves = {
        "rgb"
        if any("-" + m + "-ball-hiding-" in i for m in ("dynamo", "reviwo", "e2e_cnn", "e2e_cnn_noaug")) or "-gate-rgb-" in i
        else "geometry"
        for i in ids
    }
    registration.require(len(waves) == 1, "one wave per grant")
    wave = next(iter(waves))
    target = repo / "runs/setup" / directory / (("local-" if is_local else "remote-") + wave + ".json")
    queue_path, registry_path = repo / "experiments/queue.yaml", repo / "experiments/registry.jsonl"
    queue_raw = queue_path.read_bytes()
    queue = {r["id"]: r for r in yaml.safe_load(queue_raw)["jobs"]}
    for identity in ids:
        registration.require(queue.get(identity) == grant["rows"][identity]["row"], "registered row changed: " + identity)
    registry_token = revision(registry_path)
    inputs, source_tokens = {}, {}
    if mode == "release":
        registration.setup()  # select task before any remote admission consumer imports
        if is_local:
            local.load_authorization(path)
            source_tokens = main_acceptance(repo)
        else:
            remote.load(path, repo)
        # Refuse a grant that cannot be consumed by the installed fresh wrappers.
        names = ("bh_isolated_release.py", "bh_isolated_worker.py") if is_local else ("bh_remote_pin.py",)
        for name in names:
            expected = registration.sha((ROOT / "runs/setup/tools" / name).read_bytes())
            registration.require(
                registration.sha((repo / "runs/setup/tools" / name).read_bytes()) == expected,
                "adopt exact tools before release: " + name,
            )
        from s4d.rwlearn.pnpv2_spec import SPEC, load_spec

        _, ev = registration.publication(load_spec(ROOT / SPEC), wave)
        for stage in ev.values():
            inputs.update(stage.get("inputs", {}))
        import ast

        driver_names = ("stage2_local_driver.py", "baseline_local_driver.py") if is_local else ("remote_campaign_driver.py",)
        module_name = "runs.setup.tools.bh_isolated_release" if is_local else "runs.setup.tools.bh_remote_pin"
        entry_name = "install_all" if is_local else "install"
        for name in driver_names:
            tree = ast.parse((repo / "runs/setup/tools" / name).read_text())
            registration.require(
                any(
                    isinstance(node, ast.ImportFrom)
                    and node.module == module_name
                    and any(alias.name == entry_name for alias in node.names)
                    for node in ast.walk(tree)
                ),
                "adopt driver hook before release: " + name,
            )
        for item in grant["rows"].values():
            script = item["row"]["script"]
            registration.require(
                registration.sha((repo / script).read_bytes()) == registration.sha((ROOT / script).read_bytes()),
                "adopt guarded task wrapper: " + script,
            )
        # Snapshot history only after slow source/publication verification.
        registry_token = revision(registry_path)
        # Never unlock an already launched row with an unreviewed new grant.
        for line in registry_path.open("rb"):
            if b"bh-20261120-" in line:
                registration.require(json.loads(line).get("id") not in ids, "target already has registry history")
    registration.require(revision(registry_path) == registry_token, "registry changed; retry preview")
    pause_path = repo / "runs/setup/admission-pauses.json"
    pause_raw = pause_path.read_bytes()
    after = pause_bytes(pause_raw, ids, mode == "rollback")
    registration.require(not target.exists() or target.read_bytes() == raw, "different installed grant; refuse replacement")
    # Tokens fence evidence without hashing or network work inside admission locks.
    tokens = {**source_tokens, **{str(p): revision(Path(p)) for p in inputs}}
    tokens[str(path)] = revision(path)
    return dict(
        mode=mode,
        wave=wave,
        ids=sorted(ids),
        target=target,
        grant_raw=raw,
        queue_path=queue_path,
        queue_raw=queue_raw,
        registry_path=registry_path,
        registry_token=registry_token,
        pause_path=pause_path,
        pause_raw=pause_raw,
        pause_after=after,
        tokens=tokens,
    )


def apply(prepared, repo=registration.MAIN):
    p = prepared
    with registration.locks(repo):
        registration.require(p["queue_path"].read_bytes() == p["queue_raw"], "queue changed; retry")
        registration.require(p["pause_path"].read_bytes() == p["pause_raw"], "holds changed; retry")
        registration.require(revision(p["registry_path"]) == p["registry_token"], "registry changed; retry")
        registration.require(all(revision(Path(name)) == token for name, token in p["tokens"].items()), "evidence changed")
        registration.require(
            not p["target"].exists() or p["target"].read_bytes() == p["grant_raw"], "installed grant changed"
        )
        if p["mode"] == "release":
            p["target"].parent.mkdir(exist_ok=True)
            registration.atomic(p["target"], p["grant_raw"])
            registration.atomic(p["pause_path"], p["pause_after"])  # unhold LAST
        else:
            registration.atomic(p["pause_path"], p["pause_after"])  # hold FIRST
            p["target"].unlink(missing_ok=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("release", "rollback"))
    p.add_argument("grant", type=Path)
    p.add_argument("--repo", type=Path, default=registration.MAIN)
    p.add_argument("--apply", action="store_true")
    a = p.parse_args()
    try:
        result = plan(a.mode, a.grant, a.repo)
        if a.apply:
            registration.require(a.repo.resolve() == registration.MAIN, "apply only to canonical main")
            apply(result, a.repo)
        print(
            json.dumps(
                dict(mode=a.mode, wave=result["wave"], ids=result["ids"], apply=a.apply, target=str(result["target"])),
                indent=2,
            )
        )
    except (OSError, ValueError, KeyError) as exc:
        print("REFUSED: " + str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
