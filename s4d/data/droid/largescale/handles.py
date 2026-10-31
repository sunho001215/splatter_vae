"""Bounded per-process resources: an LRU of open file handles and a byte-bounded LRU value cache.

Both are fork-aware: a worker process never reuses a handle opened by its parent (``os.getpid`` check), and pickling
drops every open handle and cached value, so DataLoader workers start empty.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from collections.abc import Callable


class HandlePool:
    """At most ``max_open`` simultaneously open handles; the least recently used is closed first."""

    def __init__(self, opener: Callable[[str], object], closer: Callable[[object], None], max_open: int = 8):
        if max_open < 1:
            raise ValueError("max_open must be >= 1")
        self.opener, self.closer, self.max_open = opener, closer, int(max_open)
        self._open: OrderedDict[str, object] = OrderedDict()
        self._pid = os.getpid()
        self.opens = 0  # total opens (diagnostics / tests)
        self.peak = 0

    def _check_pid(self) -> None:
        if self._pid != os.getpid():  # inherited across fork: forget, never close the parent's handles
            self._open = OrderedDict()
            self._pid = os.getpid()

    def get(self, key: str):
        self._check_pid()
        if key in self._open:
            self._open.move_to_end(key)
            return self._open[key]
        while len(self._open) >= self.max_open:
            _, handle = self._open.popitem(last=False)
            self.closer(handle)
        handle = self.opener(key)
        self._open[key] = handle
        self.opens += 1
        self.peak = max(self.peak, len(self._open))
        return handle

    def __len__(self) -> int:
        return len(self._open)

    def close(self) -> None:
        self._check_pid()
        while self._open:
            _, handle = self._open.popitem(last=False)
            self.closer(handle)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_open"] = OrderedDict()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._pid = os.getpid()


class ByteLRU:
    """Values with a declared size; total size never exceeds ``max_bytes`` (a single oversize value is not kept)."""

    def __init__(self, max_bytes: int):
        self.max_bytes = int(max_bytes)
        self._items: OrderedDict[object, tuple[object, int]] = OrderedDict()
        self.bytes = 0
        self.hits = self.misses = 0

    def get(self, key, loader: Callable[[], tuple[object, int]]):
        if key in self._items:
            self._items.move_to_end(key)
            self.hits += 1
            return self._items[key][0]
        self.misses += 1
        value, size = loader()
        if size <= self.max_bytes:
            while self._items and self.bytes + size > self.max_bytes:
                _, (_, old) = self._items.popitem(last=False)
                self.bytes -= old
            self._items[key] = (value, size)
            self.bytes += size
        return value

    def clear(self) -> None:
        self._items.clear()
        self.bytes = 0

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_items"] = OrderedDict()
        state["bytes"] = 0
        return state
