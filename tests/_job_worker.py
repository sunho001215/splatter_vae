"""Guarded scheduler test job: validates its single GPU, then exits with the requested status."""

from __future__ import annotations

import faulthandler
import sys
import time
from pathlib import Path

faulthandler.enable()
faulthandler.dump_traceback_later(60, repeat=True)
print(f"worker bootstrap start time={time.time()}", flush=True)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from _bootstrap import guard_gpus  # noqa: E402

print(f"worker native guard start time={time.time()}", flush=True)
mapping = guard_gpus()
print(f"worker native guard passed time={time.time()}", flush=True)
assert len(mapping) == 1
if len(sys.argv) > 2:  # expected oom_score_adj, inherited from the scheduler's launch wrapper
    assert Path("/proc/self/oom_score_adj").read_text().strip() == sys.argv[2]
faulthandler.cancel_dump_traceback_later()
print(f"worker exit code={sys.argv[1]} time={time.time()}", flush=True)
sys.exit(int(sys.argv[1]))
