"""Development host: build the reproducible robot-PC bundle (tarball + sha256) of one commit. Never contacts the robot
PC; staging it on the 5090 (``~/shkim/robot-pc/``) is a separate copy (docs/robot_pc.md).

    .venv/bin/python -I scripts/robot_pc/make_bundle.py --commit <sha> --wheels <wheelhouse> \
        --m5-asset <exported .m5.npz> --m5-validation <realworld_m5_asset.py validate JSON> --out-dir outputs/robot_pc/bundle

Contents (prefix ``s4d-robot-pc-<short>/``): ``src/`` = ``git archive <commit>`` of the client-side code (ops/__init__,
ops/policy_serving/{__init__,protocol,client,bench,config}.py, ops/robot_pc/, s4d/, scripts/robot_pc/,
configs/robot_pc/, the deployment docs); ``setup.sh``; ``wheels/`` (pinned cp312 manylinux wheels +
hash-locked requirements: core, realsense, live = pyzmq + robot-msg-irsl); ``assets/`` (M5 surface asset, the unmodified composed URDF + manifest, the per-task rig
documents + their reference npz); ``BUNDLE.json`` (commit, every file's sha256, pins, asset provenance, validation).
Deterministic: sorted members, mtime = commit time, uid/gid 0, gzip mtime 0.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]

SRC = ["ops/__init__.py", "ops/policy_serving/__init__.py", "ops/policy_serving/protocol.py",
       "ops/policy_serving/client.py", "ops/policy_serving/bench.py", "ops/policy_serving/config.py", "ops/robot_pc",
       "s4d", "scripts/robot_pc", "configs/robot_pc", "docs/robot_pc.md", "docs/realworld_deployment.md",
       "docs/realworld_policy_serving.md"]
SETUP = "scripts/robot_pc/setup.sh"  # copied to the bundle top level
CORE = ("numpy", "pillow", "pyyaml", "scipy", "h5py")
LIVE = ("pyzmq", "robot-msg-irsl", "pyserial", "opencv-python-headless")  # controller link, gripper, session MP4
PINS = {"numpy": "1.26.4", "pillow": "11.3.0", "pyyaml": "6.0.3", "scipy": "1.15.3", "h5py": "3.16.0",  # uv.lock
        "pip": "24.3.1", "pyrealsense2": "2.59.1.11053", "pyzmq": "27.2.0", "robot-msg-irsl": "0.4.0", "pyserial": "3.5",
        "opencv-python-headless": "4.10.0.84"}
URDF = Path("/home/ws/ws/validation-20261031/panda_assets/panda_robotiq_2f85_droid_iso50.urdf")
RIG = Path("/home/ws/ws/realworld-ops/calib-loma/rel/rig")
RIG_DOCS = ("rig-loma-drawer-open", "rig-loma-mug-hanging-fx")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git(*a) -> str:
    return subprocess.run(["git", "-C", str(REPO), *a], check=True, capture_output=True, text=True).stdout.strip()


def requirements(wheels: Path, names) -> str:
    lines = []
    for w in sorted(wheels.glob("*.whl")):
        dist, version = w.name.split("-")[:2]
        name = dist.lower().replace("_", "-")
        if name in names:
            if PINS.get(name) != version or not any(t in w.name for t in ("cp312", "py3-none-any", "abi3")):
                raise SystemExit(f"{w.name}: not the pinned {name}=={PINS.get(name)} cp312 / py3 wheel")
            lines.append(f"{dist}=={version} --hash=sha256:{sha256(w)}")
    if len(lines) != len(names):
        raise SystemExit(f"wheelhouse {wheels} lacks some of {names}: {lines}")
    return "\n".join(lines) + "\n"


def build(args) -> dict:
    commit = git("rev-parse", args.commit)
    short, t_commit = commit[:7], int(git("show", "-s", "--format=%ct", commit))
    name = f"s4d-robot-pc-{short}"
    validation = json.loads(args.m5_validation.read_text())
    if not validation.get("pass") or not validation.get("episodes"):
        raise SystemExit("the M5 validation did not pass: refusing to ship the asset")
    if validation.get("asset_sha256") != sha256(args.m5_asset) or \
            (validation.get("header") or {}).get("urdf_sha256") != sha256(URDF):
        raise SystemExit("the M5 validation evidence is not for this asset / URDF: refusing to ship it")
    with tempfile.TemporaryDirectory() as tmp:
        top = Path(tmp) / name
        (top / "src").mkdir(parents=True)
        archive = subprocess.run(["git", "-C", str(REPO), "archive", "--format=tar", commit, "--", *SRC],
                                 check=True, capture_output=True).stdout
        with tarfile.open(fileobj=io.BytesIO(archive)) as t:
            t.extractall(top / "src", filter="data")
        shutil.copy2(top / "src" / SETUP, top / "setup.sh")
        (top / "wheels").mkdir()
        for w in sorted(args.wheels.glob("*.whl")):
            shutil.copy2(w, top / "wheels" / w.name)
        (top / "wheels/requirements.txt").write_text(requirements(args.wheels, CORE))
        (top / "wheels/requirements-pip.txt").write_text(requirements(args.wheels, ("pip",)))
        (top / "wheels/requirements-realsense.txt").write_text(requirements(args.wheels, ("pyrealsense2",)))
        (top / "wheels/requirements-live.txt").write_text(requirements(args.wheels, LIVE))
        for sub in ("m5", "urdf", "rig"):
            (top / "assets" / sub).mkdir(parents=True)
        shutil.copy2(args.m5_asset, top / "assets/m5" / args.m5_asset.name)
        shutil.copy2(URDF, top / "assets/urdf" / URDF.name)
        shutil.copy2(URDF.with_suffix(".manifest.json"), top / "assets/urdf" / URDF.with_suffix(".manifest.json").name)
        for doc in RIG_DOCS:
            for suffix in (".json", ".reference.npz"):
                shutil.copy2(RIG / f"{doc}{suffix}", top / "assets/rig" / f"{doc}{suffix}")
        shutil.copy2(args.m5_validation, top / "assets/m5/m5-validation.json")
        files = {str(p.relative_to(top)): sha256(p) for p in sorted(top.rglob("*")) if p.is_file()}
        manifest = {"name": name, "commit": commit, "commit_time": t_commit, "files": files,
                    "pins": {"python": "3.12", **{n: requirements(args.wheels, (n,)).split()[0] for n in
                                                  (*CORE, "pip", "pyrealsense2", *LIVE)}},
                    "layout": "<root>/releases/<name>/{src,assets,wheels,setup.sh}; <root>/{venv,src->,assets->,"
                              "config,secrets,logs,runs}",
                    "assets": {"urdf_sha256": sha256(URDF), "m5_header": validation["header"],
                               "m5_validation": {"pass": validation["pass"], "episodes": [
                                   {k: e[k] for k in ("episode", "pass")} for e in validation["episodes"]]}},
                    "safety": "execution only via infer.py without --dry-run + operator ENTER + catalog "
                              "execution_eligible (confirmed at binding and START); controller protections + "
                              "hardware e-stop unchanged"}
        (top / "BUNDLE.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
        args.out_dir.mkdir(parents=True, exist_ok=True)
        out = args.out_dir / f"{name}.tar.gz"
        raw = io.BytesIO()
        with tarfile.open(fileobj=raw, mode="w", format=tarfile.PAX_FORMAT) as t:
            for p in sorted(top.rglob("*")):
                info = t.gettarinfo(str(p), arcname=str(p.relative_to(Path(tmp))))
                info.mtime, info.uid, info.gid, info.uname, info.gname = t_commit, 0, 0, "", ""
                info.mode = 0o755 if (p.is_dir() or p.suffix == ".sh" or info.mode & 0o100) else 0o644
                if p.is_file():
                    with open(p, "rb") as f:
                        t.addfile(info, f)
                else:
                    t.addfile(info)
        with open(out, "wb") as f, gzip.GzipFile(filename="", mode="wb", fileobj=f, mtime=0) as gz:
            gz.write(raw.getvalue())
    digest = sha256(out)
    (args.out_dir / f"{name}.tar.gz.sha256").write_text(f"{digest}  {out.name}\n")
    return {"bundle": str(out), "sha256": digest, "bytes": out.stat().st_size, "commit": commit,
            "files": len(files)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="make_bundle.py", description=__doc__.split("\n")[0])
    ap.add_argument("--commit", default="HEAD")
    ap.add_argument("--wheels", type=Path, required=True)
    ap.add_argument("--m5-asset", type=Path, required=True)
    ap.add_argument("--m5-validation", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    print(json.dumps(build(ap.parse_args(argv)), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
