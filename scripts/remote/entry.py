"""Detached Docker entry for verified remote campaign jobs."""

from __future__ import annotations

import argparse
import contextlib
import os
import signal
import subprocess
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

from s4d.run_identity import (  # noqa: E402
    RunIdentityError,
    record_remote_exit,
    start_remote_run,
    verify_remote_identity,
)


def guarded_script(relative: str) -> Path:
    path = (REPO / relative).resolve()
    if REPO / "scripts" not in path.parents or path.suffix != ".py" or path == Path(__file__).resolve():
        raise RunIdentityError("Remote entry requires a repository Python script")
    tracked = subprocess.run(
        ["git", "-C", str(REPO), "ls-files", "--error-unmatch", "--", str(path.relative_to(REPO))],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if tracked.returncode or "guard_gpus" not in path.read_text():
        raise RunIdentityError("Remote entry requires a tracked, GPU-guarded script")
    return path


def run_child(script: Path, arguments: list[str], console) -> int:
    child = subprocess.Popen(
        [sys.executable, "-I", str(script), *arguments],
        cwd=REPO,
        stdout=console,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    def forward(signum, _frame):
        try:
            os.killpg(child.pid, signum)
        except ProcessLookupError:
            pass

    previous = {signum: signal.signal(signum, forward) for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    try:
        code = child.wait()
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    return code if code >= 0 else 128 - code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script", required=True)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    identity = verify_remote_identity(REPO)
    script = guarded_script(args.script)
    run_dir = start_remote_run(REPO, identity, str(script.relative_to(REPO)), arguments)
    code = 1
    with (run_dir / "console.log").open("a", buffering=1) as console:
        with contextlib.redirect_stdout(console), contextlib.redirect_stderr(console):
            try:
                from s4d.gpu_guard import enforce_allowed_gpus

                enforce_allowed_gpus()
                sys.path.insert(0, str(REPO / "scripts"))
                from _bootstrap import require_passed_tests

                require_passed_tests()
                console.flush()
                code = run_child(script, arguments, console)
            except BaseException:
                traceback.print_exc()
            finally:
                record_remote_exit(run_dir, identity, code)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
