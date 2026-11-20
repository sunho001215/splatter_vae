"""Start ONE detached rwprep_remote tick controller for a pnp-v2 remote bundle (drawer-geo lib.sh pattern).

usage: /home/ws/ws/splatter4d/.venv/bin/python -I start_tick.py <bundle dir with dispatcher.json>

Refuses a duplicate controller (scans /proc), a finished bundle, or a STOP marker; runs one preview tick (no --apply)
first, then detaches `python -u -B -m ops.jobs.rwprep_remote tick --config <bundle>/dispatcher.json --apply --loop`
from the MAIN checkout with controller.pid / controller.log in the bundle directory. Never kills anything.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

MAIN = Path("/home/ws/ws/splatter4d")
PY = str(MAIN / ".venv/bin/python")


def alive(pid, config):
    try:
        args = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    except OSError:
        return False
    return all(x.encode() in args for x in ("ops.jobs.rwprep_remote", "tick", "--apply", "--loop", str(config)))


def main():
    bundle = Path(sys.argv[1]).resolve()
    assert bundle.name.startswith('bh-20261120-'), 'BH bundles only'
    cfg = json.loads((bundle / 'dispatcher.json').read_text())
    assert cfg.get('release') == 'bh-20261120', 'BH release only'
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1' 
    config = bundle / "dispatcher.json"
    assert config.is_file(), f"no dispatcher.json in {bundle}"
    # RGB and geometry may both request normalization's controller concurrently.
    import fcntl
    start_lock = (bundle / "controller-start.lock").open("a")
    fcntl.flock(start_lock, fcntl.LOCK_EX)
    existing = [int(p.name) for p in Path("/proc").iterdir() if p.name.isdigit() and alive(int(p.name), config)]
    assert len(existing) <= 1, "HELD: multiple controllers for one bundle; inspect, never kill blindly"
    if existing:
        print("controller already active:", existing[0])
        return 0
    state = bundle / "state/state.json"
    if state.exists() and all(v["status"] == "done" for v in json.loads(state.read_text())["jobs"].values()):
        print("bundle already complete; no controller")
        return 0
    assert not (bundle / "state/STOP").exists(), "HELD: state/STOP present"
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    subprocess.run(["nice", "-n", "19", "ionice", "-c", "3", PY, "-B", "-m", "ops.jobs.rwprep_remote", "tick", "--config", str(config)], cwd=MAIN, env=env,
                   check=True)
    with (bundle / "controller.log").open("ab", buffering=0) as log, open(os.devnull, "rb") as stdin:
        proc = subprocess.Popen(["setsid", "nohup", "nice", "-n", "19", "ionice", "-c", "3", PY, "-u", "-B", "-m", "ops.jobs.rwprep_remote", "tick", "--config",
                                 str(config), "--apply", "--loop", "--max-launch", "16"], cwd=MAIN, env=env,
                                stdin=stdin, stdout=log, stderr=subprocess.STDOUT, close_fds=True)
    (bundle / "controller.pid").write_text(f"{proc.pid}\n")
    time.sleep(1.0)
    assert proc.poll() is None, "controller exited; inspect controller.log"
    print("detached controller PID", proc.pid, "log:", bundle / "controller.log")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
