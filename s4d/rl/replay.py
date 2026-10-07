"""Disk-backed DrQ-v2 replay, ported unchanged in behaviour from the reference ``agents/drqv2/replay_buffer.py``.

Each row stores one environment state atom: one RGB frame (3,H,W), one per-frame feature, or one
stack-level feature. Frame stacks and n-step returns are reconstructed at sample time.
"""

from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

_FILES = {
    "obs": "obs.memmap",
    "proprio": "proprio.memmap",
    "action": "action.memmap",
    "reward": "reward.memmap",
    "discount": "discount.memmap",
    "done": "done.memmap",
    "state_id": "state_id.memmap",
    "transition_id": "transition_id.memmap",
    "episode_id": "episode_id.memmap",
    "episode_step": "episode_step.memmap",
    "meta": "meta.memmap",
}
_NEXT_STATE_ID, _NUM_TRANSITIONS, _NEXT_EPISODE_ID = 0, 1, 2


class _Layout:
    """Shared memmap layout of storage (writer) and sampler (reader)."""

    def __init__(self, replay_dir, obs_shape, obs_dtype, proprio_shape, action_shape, max_size, frame_stack, nstep):
        self.replay_dir = Path(replay_dir)
        self.obs_shape = tuple(obs_shape)
        self.obs_dtype = np.dtype(obs_dtype)
        self.proprio_shape = tuple(proprio_shape)
        self.action_shape = tuple(action_shape)
        self.max_size = int(max_size)
        self.frame_stack = int(frame_stack)
        self.nstep = int(nstep)
        self.capacity = self.max_size + self.frame_stack + self.nstep + 1

    def open_arrays(self, mode: str) -> None:
        c = self.capacity
        specs = {
            "obs": (self.obs_dtype, (c, *self.obs_shape)),
            "proprio": (np.float32, (c, *self.proprio_shape)),
            "action": (np.float32, (c, *self.action_shape)),
            "reward": (np.float32, (c, 1)),
            "discount": (np.float32, (c, 1)),
            "done": (np.bool_, (c, 1)),
            "state_id": (np.int64, (c,)),
            "transition_id": (np.int64, (c,)),
            "episode_id": (np.int64, (c,)),
            "episode_step": (np.int64, (c,)),
            "meta": (np.int64, (3,)),
        }
        for name, (dtype, shape) in specs.items():
            setattr(
                self, f"_{name}", np.memmap(self.replay_dir / _FILES[name], dtype=np.dtype(dtype), mode=mode, shape=shape)
            )

    def slot(self, state_id: int) -> int:
        return int(state_id % self.capacity)

    def __len__(self) -> int:
        return int(min(int(self._meta[_NUM_TRANSITIONS]), self.max_size))


class MemmapReplayBufferStorage(_Layout):
    """Ring replay writer. ``reset=False`` reopens an existing buffer when resuming a crashed run."""

    def __init__(
        self, replay_dir, obs_shape, obs_dtype, proprio_shape, action_shape, max_size, frame_stack, nstep, reset=True
    ):
        super().__init__(replay_dir, obs_shape, obs_dtype, proprio_shape, action_shape, max_size, frame_stack, nstep)
        self.replay_dir.mkdir(parents=True, exist_ok=True)
        self._current_state_id: int | None = None
        if reset:
            for name in _FILES.values():
                (self.replay_dir / name).unlink(missing_ok=True)
        self.open_arrays("w+" if reset else "r+")
        if reset:
            for name in ("_state_id", "_transition_id", "_episode_id", "_episode_step"):
                getattr(self, name)[:] = -1
            self._meta[:] = 0
            self._flush_metadata()

    def _flush_metadata(self) -> None:
        for name in ("_state_id", "_transition_id", "_episode_id", "_episode_step", "_meta"):
            getattr(self, name).flush()

    def _write_state(self, obs, proprio, episode_id: int, episode_step: int) -> int:
        state_id = int(self._meta[_NEXT_STATE_ID])
        slot = self.slot(state_id)
        self._obs[slot] = np.asarray(obs, dtype=self.obs_dtype)
        self._proprio[slot] = np.asarray(proprio, dtype=np.float32)
        self._state_id[slot] = state_id
        self._episode_id[slot] = int(episode_id)
        self._episode_step[slot] = int(episode_step)
        self._meta[_NEXT_STATE_ID] = state_id + 1
        return state_id

    def add_initial(self, obs, proprio) -> None:
        episode_id = int(self._meta[_NEXT_EPISODE_ID])
        self._current_state_id = self._write_state(obs, proprio, episode_id=episode_id, episode_step=0)
        self._flush_metadata()

    def add(self, action, reward: float, discount: float, next_obs, next_proprio, done: bool) -> None:
        if self._current_state_id is None:
            raise RuntimeError("add_initial must be called before add.")
        current_id = int(self._current_state_id)
        slot = self.slot(current_id)
        episode_id = int(self._episode_id[slot])
        episode_step = int(self._episode_step[slot])
        self._action[slot] = np.asarray(action, dtype=np.float32)
        self._reward[slot] = np.asarray([reward], dtype=np.float32)
        self._discount[slot] = np.asarray([discount], dtype=np.float32)
        self._done[slot] = np.asarray([done], dtype=np.bool_)
        self._transition_id[slot] = current_id
        next_id = self._write_state(next_obs, next_proprio, episode_id=episode_id, episode_step=episode_step + 1)
        self._meta[_NUM_TRANSITIONS] = int(self._meta[_NUM_TRANSITIONS]) + 1
        if done:
            self._meta[_NEXT_EPISODE_ID] = episode_id + 1
            self._current_state_id = None
        else:
            self._current_state_id = next_id
        self._flush_metadata()

    def resume(self) -> None:
        """Start a fresh episode id after reopening, so samples never splice two run segments."""
        self._meta[_NEXT_EPISODE_ID] = int(self._episode_id.max()) + 1
        self._current_state_id = None
        self._flush_metadata()


class MemmapReplayBuffer(_Layout, IterableDataset):
    """Memmap-backed sampler that reconstructs frame stacks and n-step returns at sample time."""

    def __init__(
        self, replay_dir, obs_shape, obs_dtype, proprio_shape, action_shape, max_size, frame_stack, nstep, discount
    ):
        _Layout.__init__(self, replay_dir, obs_shape, obs_dtype, proprio_shape, action_shape, max_size, frame_stack, nstep)
        IterableDataset.__init__(self)
        self.discount_gamma = float(discount)
        self._opened = False

    def _open(self) -> None:
        if not self._opened:
            self.open_arrays("r")
            self._opened = True

    def __len__(self) -> int:
        return _Layout.__len__(self) if self._opened else 0

    def _has_state(self, sid: int) -> bool:
        return int(self._state_id[self.slot(sid)]) == int(sid)

    def _has_transition(self, sid: int) -> bool:
        return int(self._transition_id[self.slot(sid)]) == int(sid)

    def _episode(self, sid: int) -> int:
        return int(self._episode_id[self.slot(sid)])

    def _valid_start(self, sid: int, next_state_id: int) -> bool:
        if not self._has_state(sid) or not self._has_state(sid + self.nstep):
            return False
        episode = self._episode(sid)
        if self._episode(sid + self.nstep) != episode:
            return False
        for offset in range(self.nstep):
            if not self._has_transition(sid + offset) or self._episode(sid + offset) != episode:
                return False
        return sid + self.nstep < next_state_id

    def _sample_start_id(self) -> int:
        while True:
            next_state_id = int(self._meta[_NEXT_STATE_ID])
            if min(int(self._meta[_NUM_TRANSITIONS]), self.max_size) >= self.nstep:
                low, high = max(0, next_state_id - self.max_size), next_state_id - self.nstep
                if high > low:
                    for _ in range(1024):
                        sid = random.randrange(low, high)
                        if self._valid_start(sid, next_state_id):
                            return sid
            time.sleep(0.05)

    def _read_obs(self, sid: int) -> np.ndarray:
        if not self._has_state(sid):
            raise RuntimeError(f"State {sid} has been overwritten.")
        return np.asarray(self._obs[self.slot(sid)])

    def _stack_state(self, sid: int) -> np.ndarray:
        episode_step = int(self._episode_step[self.slot(sid)])
        obs = [self._read_obs(sid - min(offset, episode_step)) for offset in range(self.frame_stack - 1, -1, -1)]
        if len(self.obs_shape) == 3 and self.obs_shape[0] == 3:
            return np.concatenate(obs, axis=0)
        return np.stack(obs, axis=0)

    def sample(self):
        self._open()
        sid = self._sample_start_id()
        nid = sid + self.nstep
        reward = np.zeros((1,), dtype=np.float32)
        discount = np.ones((1,), dtype=np.float32)
        for offset in range(self.nstep):
            slot = self.slot(sid + offset)
            reward += discount * np.asarray(self._reward[slot])
            discount *= np.asarray(self._discount[slot]) * self.discount_gamma
        return (
            self._stack_state(sid).copy(),
            np.array(self._proprio[self.slot(sid)]),
            np.array(self._action[self.slot(sid)]),
            reward,
            discount,
            self._stack_state(nid).copy(),
            np.array(self._proprio[self.slot(nid)]),
        )

    def __iter__(self):
        while True:
            yield self.sample()


def _seed_worker(_worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def make_replay_loader(storage: MemmapReplayBufferStorage, batch_size: int, num_workers: int, discount: float):
    dataset = MemmapReplayBuffer(
        storage.replay_dir,
        storage.obs_shape,
        storage.obs_dtype,
        storage.proprio_shape,
        storage.action_shape,
        storage.max_size,
        storage.frame_stack,
        storage.nstep,
        discount,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=_seed_worker,
        persistent_workers=int(num_workers) > 0,
    )
    return loader, dataset
