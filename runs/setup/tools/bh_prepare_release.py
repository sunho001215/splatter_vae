"""Prepare sealed BH source and exact-ID grants. Writes only --out; never installs a grant.

seal: source-only local clone + local submodules, then read-only/hash seal.
local: owner-scoped targeted evidence, honest full_suite_claimed=false.
remote: existing full native acceptance; selective CPU evidence cannot substitute.
"""

from __future__ import annotations

import argparse
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bh_isolated_release as local  # noqa: E402
import bh_register as registration  # noqa: E402


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args])


def seal(out):
    commit = git(ROOT, "rev-parse", "HEAD").decode().strip()
    registration.require(not git(ROOT, "status", "--porcelain", "--untracked-files=no"), "clean committed source required")
    registration.require(
        not git(
            ROOT,
            "ls-files",
            "--others",
            "--exclude-standard",
            "--",
            "s4d",
            "ops",
            "scripts",
            "configs",
            "tests",
            "runs/setup/tools",
        ),
        "untracked source must be committed before sealing",
    )
    out.mkdir(parents=True, exist_ok=False)
    code = out / "code"
    subprocess.run(["git", "clone", "--shared", "--no-checkout", str(ROOT), str(code)], check=True)
    git(code, "checkout", "--detach", commit)
    listing = git(ROOT, "ls-files", "--stage", "-z").split(b"\0")
    source_files, gitlinks, unmaterialized = {}, {}, {}
    for record in listing:
        if not record:
            continue
        meta, name = record.split(b"\t", 1)
        mode, blob, stage = meta.decode().split()
        name = name.decode()
        registration.require(stage == "0", "unmerged source")
        if mode == "160000":
            source = ROOT / name
            registration.require(
                git(source, "rev-parse", "HEAD").decode().strip() == blob and not git(source, "status", "--porcelain"),
                "dirty/missing pinned submodule",
            )
            # Local refs only, no network and no hardlinks to writable source data.
            subprocess.run(["git", "clone", "--shared", "--no-checkout", str(source), str(code / name)], check=True)
            git(code / name, "checkout", "--detach", blob)
            gitlinks[name] = blob
            for item in git(source, "ls-files", "--stage", "-z").split(b"\0"):
                if item:
                    child_meta, child_name = item.split(b"\t", 1)
                    child_mode, child_blob, _ = child_meta.decode().split()
                    child_name = child_name.decode()
                    child = code / name / child_name
                    if child_mode == "160000":
                        # Upstream RVT carries optional simulation gitlinks. The
                        # existing real-world checkout does not materialize them.
                        # Record absence honestly; never fetch them from a network.
                        registration.require(
                            not child.exists() or not any(child.iterdir()), "unexpected materialized nested submodule"
                        )
                        unmaterialized[f"{name}/{child_name}"] = child_blob
                        continue
                    registration.require(child.is_file() and not child.is_symlink(), "nonregular submodule source")
                    source_files[f"{name}/{child_name}"] = {"sha256": registration.sha(child.read_bytes())}
        else:
            path = code / name
            registration.require(path.is_file() and not path.is_symlink(), "nonregular candidate source")
            source_files[name] = {"sha256": registration.sha(path.read_bytes())}
    # Seal all private contents only. Shared alternates remain untouched.
    for directory, dirs, files in os.walk(code, topdown=False, followlinks=False):
        for name in files:
            path = Path(directory) / name
            registration.require(not path.is_symlink(), "source symlink refused")
            path.chmod(stat.S_IMODE(path.stat().st_mode) & ~0o222)
        for name in dirs:
            path = Path(directory) / name
            registration.require(not path.is_symlink(), "directory symlink refused")
            path.chmod(stat.S_IMODE(path.stat().st_mode) & ~0o222)
    code.chmod(stat.S_IMODE(code.stat().st_mode) & ~0o222)
    snapshot = dict(
        commit=commit,
        code_root=str(code.resolve()),
        source_files=source_files,
        gitlinks=gitlinks,
        unmaterialized_gitlinks=unmaterialized,
    )
    path = out / "snapshot.json"
    path.write_text(json.dumps(snapshot, indent=2) + "\n")
    path.chmod(0o444)
    return dict(snapshot=str(path), sha256=registration.sha(path.read_bytes()), files=len(source_files))


def grant_rows(record, host):
    rows = [r for r in record["rows"] if r["host"] == host]
    registration.require(rows and {r["id"] for r in rows} <= local.IDS, "no exact host rows")
    return {r["id"]: dict(row=r, row_sha256=registration.sha(registration.canonical(r))) for r in rows}


def prepare_local(record, snapshot_path, binding_path, accepted_by, out):
    snapshot = json.loads(snapshot_path.read_bytes())
    binding = json.loads(binding_path.read_bytes())
    commit = record["release"]["commit"]
    rows = grant_rows(record, "local")
    registration.require(snapshot["commit"] == binding["commit"] == commit, "test/source/row commit mismatch")
    registration.require(
        binding["source_unchanged"] is True and binding["science_unchanged"] is True,
        "focused source/science proof incomplete",
    )
    registration.require(
        all(snapshot["source_files"][name]["sha256"] == digest for name, digest in binding["source_sha256"].items()),
        "tested source differs from seal",
    )
    registration.require(
        set(snapshot["source_files"]) == set(binding["source_sha256"]), "incomplete focused source inventory"
    )
    report_path = Path(binding["report"])
    registration.require(registration.sha(report_path.read_bytes()) == binding["report_sha256"], "focused report changed")
    inventory = dict(
        commit=commit,
        ids=sorted(rows),
        science_unchanged=True,
        test_report=str(report_path),
        artifacts_sha256={
            str(report_path): binding["report_sha256"],
            str(binding_path): registration.sha(binding_path.read_bytes()),
        },
    )
    inventory_path = out.with_name(out.stem + "-evidence.json")
    inventory_path.write_text(json.dumps(inventory, indent=2) + "\n")
    tools = Path(__file__).resolve().parent
    return dict(
        kind="bh-isolated-targeted-v1",
        active=True,
        commit=commit,
        rows=rows,
        accepted_by=accepted_by,
        full_suite_claimed=False,
        snapshot=str(snapshot_path),
        snapshot_sha256=registration.sha(snapshot_path.read_bytes()),
        targeted_evidence=str(inventory_path),
        targeted_evidence_sha256=registration.sha(inventory_path.read_bytes()),
        adapter_sha256=registration.sha((tools / "bh_isolated_release.py").read_bytes()),
        worker_sha256=registration.sha((tools / "bh_isolated_worker.py").read_bytes()),
    )


def prepare_remote(record, manifest):
    from ops.jobs.remote_backend import acceptance_manifest
    from runs.setup.tools import bh_remote_pin

    release = record["release"]
    evidence = acceptance_manifest(
        dict(commit=release["commit"], image_digest=release["image"], evidence_manifest=manifest), registration.MAIN
    )
    return dict(
        kind="bh-remote-native-pin/1",
        active=True,
        commit=release["commit"],
        image=release["image"],
        rows=grant_rows(record, "remote"),
        evidence_manifest=manifest,
        acceptance_sha256=evidence["manifest_sha256"],
        adapter_sha256=registration.sha(Path(bh_remote_pin.__file__).read_bytes()),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)
    s = sub.add_parser("seal")
    s.add_argument("--out", type=Path, required=True)
    for mode in ("local", "remote"):
        a = sub.add_parser(mode)
        a.add_argument("--registration", type=Path, required=True)
        a.add_argument("--out", type=Path, required=True)
        if mode == "local":
            a.add_argument("--snapshot", type=Path, required=True)
            a.add_argument("--binding", type=Path, required=True)
            a.add_argument("--accepted-by", required=True)
        else:
            a.add_argument("--evidence-manifest", required=True)
    a = p.parse_args()
    try:
        if a.mode == "seal":
            result = seal(a.out)
        else:
            registration.require(not a.out.exists(), "grant output exists")
            record = json.loads(a.registration.read_bytes())
            result = (
                prepare_local(record, a.snapshot.resolve(), a.binding.resolve(), a.accepted_by, a.out.resolve())
                if a.mode == "local"
                else prepare_remote(record, a.evidence_manifest)
            )
            a.out.write_text(json.dumps(result, indent=2) + "\n")
            if a.mode == "local":
                local.load_authorization(a.out.resolve())
        print(json.dumps(result if a.mode == "seal" else dict(grant=str(a.out), ids=sorted(result["rows"])), indent=2))
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        print("REFUSED: " + str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
