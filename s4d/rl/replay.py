"""DrM replay with one storage layout and two backings, chosen by encoder type.

Each row holds one environment *state atom* plus the transition that leaves it:
- frozen encoders (splatter4d, SinCro, ReViWo): the fp16 latent computed once per environment step,
  kept in RAM (``directory=None``); no images are stored;
- pixel encoders (CNN): one uint8 RGB frame per state, in a preallocated disk memmap under
  ``runs/<id>/replay/`` that DataLoader workers read; frame stacks are rebuilt by index.

Stacks and n-step returns are reconstructed at sample time as in the reference repository's
``agents/drqv2/replay_buffer.py``: frames before the episode start repeat the first frame, and a sample never
crosses an episode boundary or an overwritten slot.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

FIELDS = ("atom", "proprio", "action", "reward", "discount", "state_id", "transition_id", "episode_id", "episode_step")
NEXT_STATE_ID, NUM_TRANSITIONS, NEXT_EPISODE_ID = 0, 1, 2


class Replay:
    def __init__(
        self,
        directory: Path | None,
        atom_shape: tuple[int, ...],
        atom_dtype,
        proprio_dim: int,
        action_dim: int,
        max_size: int,
        frame_stack: int,
        nstep: int,
        mode: str = "w+",
        snapshot_dir: Path | None = None,
    ):
        self.directory = None if directory is None else Path(directory)
        self.snapshot_dir = None if snapshot_dir is None else Path(snapshot_dir)
        self.atom_shape, self.atom_dtype = tuple(atom_shape), np.dtype(atom_dtype)
        self.proprio_dim, self.action_dim = int(proprio_dim), int(action_dim)
        self.max_size, self.frame_stack, self.nstep = int(max_size), int(frame_stack), int(nstep)
        self.capacity = self.max_size + self.frame_stack + self.nstep + 1
        c = self.capacity
        specs = {
            "atom": (self.atom_dtype, (c, *self.atom_shape)),
            "proprio": (np.float32, (c, self.proprio_dim)),
            "action": (np.float32, (c, self.action_dim)),
            "reward": (np.float32, (c,)),
            "discount": (np.float32, (c,)),
            "state_id": (np.int64, (c,)),
            "transition_id": (np.int64, (c,)),
            "episode_id": (np.int64, (c,)),
            "episode_step": (np.int64, (c,)),
            "meta": (np.int64, (3,)),
        }
        fresh = mode == "w+"
        if self.directory is not None:
            self.directory.mkdir(parents=True, exist_ok=True)
        for name, (dtype, shape) in specs.items():
            if self.directory is None:
                array = np.zeros(shape, dtype=dtype)
            else:
                array = np.memmap(self.directory / f"{name}.memmap", dtype=dtype, mode=mode, shape=shape)
            setattr(self, name, array)
        if fresh:
            for name in ("state_id", "transition_id", "episode_id", "episode_step"):
                getattr(self, name)[:] = -1
            self.meta[:] = 0
        self.current: int | None = None

    # ------------------------------------------------------------------ writing
    def __len__(self) -> int:
        return int(min(int(self.meta[NUM_TRANSITIONS]), self.max_size))

    def nbytes(self) -> int:
        return sum(getattr(self, name).nbytes for name in (*FIELDS, "meta"))

    def _write_state(self, atom, proprio, episode_id: int, episode_step: int) -> int:
        sid = int(self.meta[NEXT_STATE_ID])
        slot = sid % self.capacity
        self.atom[slot] = atom
        self.proprio[slot] = proprio
        self.state_id[slot], self.episode_id[slot], self.episode_step[slot] = sid, episode_id, episode_step
        self.transition_id[slot] = -1
        self.meta[NEXT_STATE_ID] = sid + 1
        return sid

    def add_initial(self, atom, proprio) -> None:
        self.current = self._write_state(atom, proprio, int(self.meta[NEXT_EPISODE_ID]), 0)

    def add(self, action, reward: float, discount: float, next_atom, next_proprio, done: bool) -> None:
        if self.current is None:
            raise RuntimeError("add_initial must be called before add")
        slot = self.current % self.capacity
        episode, step = int(self.episode_id[slot]), int(self.episode_step[slot])
        self.action[slot], self.reward[slot], self.discount[slot] = action, reward, discount
        nxt = self._write_state(next_atom, next_proprio, episode, step + 1)
        self.transition_id[slot] = self.current  # published last: the transition is now complete
        self.meta[NUM_TRANSITIONS] += 1
        if done:
            self.meta[NEXT_EPISODE_ID] = episode + 1
            self.current = None
        else:
            self.current = nxt

    def checkpoint(self) -> np.ndarray:
        """Persist the buffer for crash resume and return its counters (stored in the agent checkpoint).

        Disk buffers are flushed in place. RAM buffers are snapshotted to ``snapshot_dir`` (no images:
        frozen-encoder buffers hold latents only).
        """
        if self.directory is not None:
            for name in (*FIELDS, "meta"):
                getattr(self, name).flush()
        else:
            self.snapshot_dir.mkdir(parents=True, exist_ok=True)
            tmp = self.snapshot_dir / "ram_snapshot.tmp.npz"
            np.savez(tmp, **{name: getattr(self, name) for name in (*FIELDS, "meta")})
            tmp.replace(self.snapshot_dir / "ram_snapshot.npz")
        return np.array(self.meta)

    def restore(self, meta: np.ndarray) -> None:
        """Return to a checkpoint: load the RAM snapshot or drop disk rows written after it; start a new episode."""
        if self.directory is None:
            with np.load(self.snapshot_dir / "ram_snapshot.npz") as saved:
                for name in (*FIELDS, "meta"):
                    getattr(self, name)[:] = saved[name]
        self.meta[:] = meta
        stale = self.state_id >= int(meta[NEXT_STATE_ID])
        for name in ("state_id", "transition_id", "episode_id", "episode_step"):
            getattr(self, name)[stale] = -1
        self.transition_id[self.transition_id >= int(meta[NEXT_STATE_ID]) - 1] = -1  # the in-progress transition
        self.meta[NEXT_EPISODE_ID] = int(self.episode_id.max()) + 1
        self.current = None

    # ------------------------------------------------------------------ sampling
    def _valid(self, sid: np.ndarray, next_state_id: int) -> np.ndarray:
        """Vectorised reference ``_valid_start``: sid..sid+n are live states of one episode with live transitions."""
        ok = sid + self.nstep < next_state_id
        episode = self.episode_id[sid % self.capacity]
        for offset in range(self.nstep + 1):
            slots = (sid + offset) % self.capacity
            ok &= (self.state_id[slots] == sid + offset) & (self.episode_id[slots] == episode)
            if offset < self.nstep:
                ok &= self.transition_id[slots] == sid + offset
        return ok

    def _stack(self, sid: np.ndarray) -> np.ndarray:
        steps = self.episode_step[sid % self.capacity]
        offsets = np.arange(self.frame_stack - 1, -1, -1)
        ids = sid[:, None] - np.minimum(offsets[None, :], steps[:, None])
        if (self.state_id[ids % self.capacity] != ids).any():
            raise RuntimeError("a stacked state was overwritten while sampling")
        atoms = self.atom[ids % self.capacity]  # (B, frame_stack, *atom_shape)
        if self.frame_stack == 1:
            return atoms[:, 0]
        if len(self.atom_shape) == 3 and self.atom_shape[0] == 3:  # RGB frames -> channel stack
            return atoms.reshape(len(sid), 3 * self.frame_stack, *self.atom_shape[1:])
        return atoms

    def sample(self, batch_size: int, rng: np.random.Generator, gamma: float):
        next_state_id = int(self.meta[NEXT_STATE_ID])
        low = max(0, next_state_id - self.max_size)
        high = next_state_id - self.nstep
        if high <= low:
            raise RuntimeError("not enough transitions to sample")
        chosen = np.empty(0, dtype=np.int64)
        for _ in range(1000):
            candidates = rng.integers(low, high, size=2 * batch_size)
            chosen = np.concatenate([chosen, candidates[self._valid(candidates, next_state_id)]])
            if len(chosen) >= batch_size:
                break
        else:
            raise RuntimeError("could not find valid replay samples")
        sid = chosen[:batch_size]
        reward = np.zeros(batch_size, dtype=np.float32)
        discount = np.ones(batch_size, dtype=np.float32)
        for offset in range(self.nstep):
            slots = (sid + offset) % self.capacity
            reward += discount * self.reward[slots]
            discount *= self.discount[slots] * gamma
        nid = sid + self.nstep
        return (
            self._stack(sid),
            self.proprio[sid % self.capacity],
            self.action[sid % self.capacity],
            reward[:, None],
            discount[:, None],
            self._stack(nid),
            self.proprio[nid % self.capacity],
        )


class _DiskReader(IterableDataset):
    """DataLoader worker view of a disk buffer: reopens the memmaps read-only and yields whole batches."""

    def __init__(self, replay: Replay, batch_size: int, gamma: float):
        super().__init__()
        self.args = (
            replay.directory,
            replay.atom_shape,
            replay.atom_dtype,
            replay.proprio_dim,
            replay.action_dim,
            replay.max_size,
            replay.frame_stack,
            replay.nstep,
        )
        self.batch_size, self.gamma = batch_size, gamma

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        rng = np.random.default_rng(None if info is None else info.seed % 2**32)
        reader = Replay(*self.args, mode="r")  # MAP_SHARED: sees the writer's rows through the page cache
        while True:
            yield tuple(torch.from_numpy(np.ascontiguousarray(x)) for x in reader.sample(self.batch_size, rng, self.gamma))


class _RamReader:
    """In-process sampler for RAM buffers (frozen-encoder latents)."""

    def __init__(self, replay: Replay, batch_size: int, gamma: float, seed: int):
        self.replay, self.batch_size, self.gamma, self.rng = replay, batch_size, gamma, np.random.default_rng(seed)

    def __iter__(self):
        return self

    def __next__(self):
        return tuple(
            torch.from_numpy(np.ascontiguousarray(x)) for x in self.replay.sample(self.batch_size, self.rng, self.gamma)
        )


def replay_iterator(replay: Replay, batch_size: int, gamma: float, num_workers: int, seed: int):
    """Batches of (obs, proprio, action, reward, discount, next_obs, next_proprio) tensors."""
    if replay.directory is None:
        return _RamReader(replay, batch_size, gamma, seed)
    loader = DataLoader(
        _DiskReader(replay, batch_size, gamma),
        batch_size=None,
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=int(num_workers) > 0,
        prefetch_factor=4 if int(num_workers) > 0 else None,
    )
    return iter(loader)
