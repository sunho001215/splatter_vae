"""Claude-only two-wave BH registrar/releaser. No GPU/job launch or placement.

--apply is required. --once checks one pass, otherwise retry pending barriers every
60s. STOP in --out stops this helper; original prep/controllers/jobs are untouched.
Local targeted grants and remote native pins advance independently. A missing native
release is reported as pending; it never falls back to the role release.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MAIN = Path("/home/ws/ws/splatter4d")
TOOLS = ROOT / "runs/setup/tools"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--binding", type=Path, required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--accepted-by", required=True)
    p.add_argument("--evidence-manifest", help="exact candidate native acceptance; default candidate -cc path")
    p.add_argument("--once", action="store_true")
    p.add_argument("--apply", action="store_true", required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    commit = json.loads(a.snapshot.read_bytes())["commit"]
    manifest = a.evidence_manifest or f"runs/remote/acceptance/{commit[:7]}-cc/manifest.json"

    def run(name, args, logfile):
        with (a.out / logfile).open("a") as log:
            code = subprocess.run(
                [sys.executable, "-B", "-I", str(TOOLS / name), *map(str, args)],
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
            ).returncode
        return code == 0

    while not (a.out / "STOP").exists():
        for wave in ("rgb", "geometry"):
            proof = MAIN / "runs/setup" / f"bh-20261120-{wave}.registration.json"
            if not proof.exists():
                if not run("bh_register.py", ["--wave", wave, "--image", a.image, "--apply"], f"{wave}-register.log"):
                    print(f"{wave}: registration pending/refused; see log", flush=True)
                    continue
            record = json.loads(proof.read_bytes())
            if record["release"]["commit"] != commit:
                raise SystemExit("registered pin differs from snapshot; no re-pin allowed")
            for host in ("local", "remote"):
                target = (
                    MAIN
                    / "runs/setup"
                    / ("bh-isolated-release.d" if host == "local" else "bh-remote-pins.d")
                    / f"{host}-{wave}.json"
                )
                if target.exists():
                    continue
                grant = a.out / f"{host}-{wave}.json"
                if not grant.exists():
                    args = [host, "--registration", proof, "--out", grant]
                    args += (
                        ["--snapshot", a.snapshot, "--binding", a.binding, "--accepted-by", a.accepted_by]
                        if host == "local"
                        else ["--evidence-manifest", manifest]
                    )
                    if not run("bh_prepare_release.py", args, f"{wave}-{host}-prepare.log"):
                        print(f"{wave}/{host}: release evidence pending/refused", flush=True)
                        continue
                if not run("bh_release_control.py", ["release", grant], f"{wave}-{host}-release.log"):
                    print(f"{wave}/{host}: publication/adoption pending/refused", flush=True)
                    continue
                if host == "local" and not run(
                    "bh_fetch_inputs.py", ["--registration", proof, "--apply"], f"{wave}-fetch.log"
                ):
                    print(f"{wave}/{host}: verified local inputs pending/refused", flush=True)
                    continue
                if run("bh_release_control.py", ["release", grant, "--apply"], f"{wave}-{host}-release.log"):
                    print(f"{wave}/{host}: verified exact IDs released; normal scheduling owns admission", flush=True)
        if a.once:
            break
        if all(
            (MAIN / "runs/setup" / directory / f"{host}-{wave}.json").exists()
            for directory, host in [("bh-isolated-release.d", "local"), ("bh-remote-pins.d", "remote")]
            for wave in ("rgb", "geometry")
        ):
            break
        time.sleep(60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
