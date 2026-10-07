"""Fork-safe DataLoader iteration.

Worker processes are forked from a parent that already holds CUDA tensors. If the parent has uncollected cyclic
garbage that references CUDA tensors, the garbage collector of a forked worker may free them, and freeing CUDA
memory in a forked child aborts with "CUDA error: initialization error" (observed in an evaluation loader). The
parent therefore collects garbage and freezes its collector while the workers are forked, so the workers never
collect inherited objects; the parent's collector is re-enabled once the workers exist.
"""

from __future__ import annotations

import gc
from collections.abc import Iterable, Iterator


def fork_safe_iter(loader: Iterable) -> Iterator:
    gc.collect()
    gc.freeze()
    try:
        return iter(loader)  # a multi-process DataLoader forks its workers here
    finally:
        gc.unfreeze()
