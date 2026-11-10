"""ROBOT PC self-check after setup (no device access, no robot, no session): interpreter and pinned packages, the
client stack imports WITHOUT torch, bundle file / asset sha256s, the M5 helper on the shipped URDF + surface asset,
the rig documents, the safety configuration (execution not armed, not permitted, manual mode refused), the token
file's permissions; with ``--server`` also the server's WELCOME through the tunnel (no session is opened).

    venv/bin/python -I src/scripts/robot_pc/selfcheck.py [--server]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import stat
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

PINS = {"numpy": "1.26.4", "PIL": "11.3.0", "yaml": "6.0.3", "scipy": "1.15.3", "h5py": "3.16.0"}
OPTIONAL = {"pyrealsense2": "2.59.1", "zmq": "27.2.0", "robot_msg": "0.4.0"}
DISTRIBUTIONS = {"robot_msg": "robot-msg-irsl"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="selfcheck.py", description=__doc__.split("\n")[0])
    ap.add_argument("--root", default="~/workspace/s4d-robot-pc")
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--server", action="store_true", help="also read the server's WELCOME (no session)")
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    root = Path(args.root).expanduser()
    checks, fail = {}, []

    def check(name, ok, detail=None):
        checks[name] = {"ok": bool(ok), "detail": detail}
        if not ok:
            fail.append(name)

    check("python", sys.version_info[:2] == (3, 12), platform.python_version())
    import importlib
    import importlib.metadata

    import ops.policy_serving.client  # noqa: F401
    import ops.robot_pc.cli  # noqa: F401
    import ops.robot_pc.dryrun  # noqa: F401
    import ops.robot_pc.replay  # noqa: F401
    import s4d.rwlearn.deploy.m5_live  # noqa: F401
    from s4d.rwlearn.deploy import preprocess_frame  # noqa: F401

    check("no_torch_loaded", "torch" not in sys.modules, sorted(m for m in sys.modules if m.startswith("torch")))
    versions = {}
    for mod, want in {**PINS, **OPTIONAL}.items():
        try:
            versions[mod] = getattr(importlib.import_module(mod), "__version__", None) or \
                importlib.metadata.version(DISTRIBUTIONS.get(mod, mod))
        except ImportError:
            versions[mod] = None
    for mod, want in PINS.items():
        check(f"pin_{mod}", versions[mod] == want, versions[mod])
    rs = versions["pyrealsense2"]
    check("pyrealsense2", rs is None or str(rs).startswith(OPTIONAL["pyrealsense2"]),
          rs or "not installed (live acquisition off)")
    checks["live_state_reader_deps"] = {"ok": True, "detail": {k: versions[k] for k in ("zmq", "robot_msg")}}

    from ops.robot_pc import cli
    from ops.robot_pc import config as C

    cfg = cli.load_config(argparse.Namespace(config=args.config, root=str(root)))
    bundle = root / "BUNDLE.json"
    if bundle.is_file():
        manifest = json.loads(bundle.read_text())
        rel = Path(manifest["name"])
        bad = [p for p, h in manifest["files"].items()
               if (root / "releases" / rel / p).is_file() and sha256(root / "releases" / rel / p) != h]
        missing = [p for p in manifest["files"] if not (root / "releases" / rel / p).is_file()]
        check("bundle_files_sha256", not bad and not missing, {"mismatch": bad[:5], "missing": missing[:5],
                                                              "files": len(manifest["files"])})
        active = {k: (root / k).resolve() == (root / "releases" / rel / k).resolve() for k in ("src", "assets")}
        check("active_release", all(active.values()) and Path(__file__).resolve().is_relative_to(
            (root / "releases" / rel).resolve()), {"release": str(rel), **active})
    else:
        checks["bundle_files_sha256"] = {"ok": True, "detail": "no BUNDLE.json (git checkout): skipped"}
    a = cfg.get("assets") or {}
    try:
        from s4d.rwlearn.deploy.m5_live import SurfaceMotion

        motion = SurfaceMotion(C.path(cfg, a["m5_surface"]), C.path(cfg, a["urdf"]))
        st = cfg["static_latency_state"]
        pts = motion.points((np.asarray(st["q7"]), st["gripper_register"]))
        check("m5_helper", pts.shape == (motion.header["points"], 3) and np.isfinite(pts).all(),
              {"asset_sha256": motion.asset_sha256, "points": int(pts.shape[0])})
    except Exception as exc:  # noqa: BLE001 - reported
        check("m5_helper", False, f"{type(exc).__name__}: {exc}")
    for task, entry in (cfg.get("tasks") or {}).items():
        try:
            from s4d.rwlearn.deploy.realsense import load_rig_session

            cams = (cfg.get("cameras") or {}).get("serials", [])
            out = load_rig_session(C.path(cfg, entry["rig_document"]), cams, {c: np.eye(3) for c in cams})
            check(f"rig_{task}", True, {c: v["factor"] for c, v in out["native_depth_correction"].items()})
        except Exception as exc:  # noqa: BLE001 - reported
            check(f"rig_{task}", False, f"{type(exc).__name__}: {exc}")
    ex = cfg.get("execution") or {}
    check("execution_not_armed", ex.get("armed") is False, ex)
    from ops.robot_pc import safety

    check("execution_not_permitted", safety.EXECUTION_PERMITTED is False, "safety.EXECUTION_PERMITTED")
    tok = C.path(cfg, (cfg.get("server") or {}).get("token_file"))
    if tok is not None and tok.is_file():
        mode = stat.S_IMODE(tok.stat().st_mode)
        check("token_file_0600", mode & 0o077 == 0, oct(mode))
    else:
        checks["token_file_0600"] = {"ok": True, "detail": f"MISSING {tok}: copy the server token there (user action)"}
    if args.server:
        try:
            w = cli.welcome(cfg, argparse.Namespace(host=args.host, port=args.port))
            check("server_welcome", True, {"server": w.get("server"), "policies": sorted(w["policies"])})
        except Exception as exc:  # noqa: BLE001 - reported
            check("server_welcome", False, f"{type(exc).__name__}: {exc}")
    out = {"ok": not fail, "failed": fail, "root": str(root), "versions": versions, "checks": checks}
    text = json.dumps(out, indent=1, default=str)
    if args.out:
        from ops.robot_pc.safety import safe_output

        safe_output(args.out).write_text(text + "\n")
    print(text)
    return 0 if not fail else 1


if __name__ == "__main__":
    raise SystemExit(main())
