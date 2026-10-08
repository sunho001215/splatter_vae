"""Guarded scheduler test job: validates its single GPU, then exits with the requested status."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from _bootstrap import guard_gpus  # noqa: E402

mapping = guard_gpus()
assert len(mapping) == 1
if len(sys.argv) > 2:  # expected oom_score_adj, inherited from the scheduler's launch wrapper
    assert Path("/proc/self/oom_score_adj").read_text().strip() == sys.argv[2]
sys.exit(int(sys.argv[1]))
