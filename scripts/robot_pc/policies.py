"""ROBOT PC: list the policies the server offers, with each contract's camera / depth / calibration requirements and
whether this robot PC can run it live (single-view yes; multi-view needs the validated M5 asset). Opens no session.

    venv/bin/python -I src/scripts/robot_pc/policies.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ops.robot_pc import cli  # noqa: E402
from ops.robot_pc import config as C  # noqa: E402
from ops.robot_pc import selection  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="policies.py", description=__doc__.split("\n")[0])
    cli.add_common(ap)
    args = ap.parse_args(argv)
    cfg = cli.load_config(args)
    w = cli.welcome(cfg, args)
    a = cfg.get("assets") or {}
    m5_ready = all(C.path(cfg, a.get(k)) is not None and C.path(cfg, a.get(k)).is_file() for k in ("m5_surface", "urdf"))
    rows = []
    for r in selection.describe(w["policies"]):
        r["tasks_configured"] = r["task"] in (cfg.get("tasks") or {})
        r["live_dry_run"] = ("ok" if r["views"] == 1 else
                             "ok (validated M5 helper)" if m5_ready else "refused: M5 asset missing")
        rows.append(r)
    cli.write_json(None, {"server": w.get("server"), "auth_required": w.get("auth_required"), "policies": rows})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
