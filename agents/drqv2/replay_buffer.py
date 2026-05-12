from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset


_MEMMAP_OBS_FILE = "obs.memmap"
_MEMMAP_PROPRIO_FILE = "proprio.memmap"
_MEMMAP_ACTION_FILE = "action.memmap"
_MEMMAP_REWARD_FILE = "reward.memmap"
_MEMMAP_DISCOUNT_FILE = "discount.memmap"
_MEMMAP_DONE_FILE = "done.memmap"
_MEMMAP_STATE_ID_FILE = "state_id.memmap"
_MEMMAP_TRANSITION_ID_FILE = "transition_id.memmap"
_MEMMAP_EPISODE_ID_FILE = "episode_id.memmap"
_MEMMAP_EPISODE_STEP_FILE = "episode_step.memmap"
_MEMMAP_META_FILE = "meta.memmap"
_META_NEXT_STATE_ID = 0
_META_NUM_TRANSITIONS = 1
_META_NEXT_EPISODE_ID = 2


def _open_memmap(path: Path, dtype: np.dtype, shape: Tuple[int, ...], mode: str) -> np.memmap:
    return np.memmap(path, dtype=np.dtype(dtype), mode=mode, shape=shape)


class MemmapReplayBufferStorage:
    """
    Disk-backed ring replay storage.

    Each row stores one environment state atom: either one RGB frame (3,H,W),
    one single-frame feature tensor, or one stack-level feature tensor.
    """

    def __init__(
        self,
        replay_dir: Path,
        obs_shape: Tuple[int, ...],
        obs_dtype: np.dtype,
        proprio_shape: Tuple[int, ...],
        action_shape: Tuple[int, ...],
        max_size: int,
        frame_stack: int,
        nstep: int,
        reset: bool = True,
    ) -> None:
        self._replay_dir = Path(replay_dir)
        self._replay_dir.mkdir(parents=True, exist_ok=True)
        self._obs_shape = tuple(obs_shape)
        self._obs_dtype = np.dtype(obs_dtype)
        self._proprio_shape = tuple(proprio_shape)
        self._action_shape = tuple(action_shape)
        self._max_size = int(max_size)
        self._frame_stack = int(frame_stack)
        self._nstep = int(nstep)
        self._capacity = self._max_size + self._frame_stack + self._nstep + 1
        self._current_state_id: int | None = None

        if reset:
            self._reset_files()
        self._open(mode="w+" if reset else "r+")
        if reset:
            self._state_id[:] = -1
            self._transition_id[:] = -1
            self._episode_id[:] = -1
            self._episode_step[:] = -1
            self._meta[:] = 0
            self._flush_metadata()

    def _reset_files(self) -> None:
        for fn in self._replay_dir.glob("*.npz"):
            fn.unlink(missing_ok=True)
        for name in (
            _MEMMAP_OBS_FILE,
            _MEMMAP_PROPRIO_FILE,
            _MEMMAP_ACTION_FILE,
            _MEMMAP_REWARD_FILE,
            _MEMMAP_DISCOUNT_FILE,
            _MEMMAP_DONE_FILE,
            _MEMMAP_STATE_ID_FILE,
            _MEMMAP_TRANSITION_ID_FILE,
            _MEMMAP_EPISODE_ID_FILE,
            _MEMMAP_EPISODE_STEP_FILE,
            _MEMMAP_META_FILE,
        ):
            (self._replay_dir / name).unlink(missing_ok=True)

    def _open(self, mode: str) -> None:
        self._obs = _open_memmap(
            self._replay_dir / _MEMMAP_OBS_FILE,
            self._obs_dtype,
            (self._capacity, *self._obs_shape),
            mode,
        )
        self._proprio = _open_memmap(
            self._replay_dir / _MEMMAP_PROPRIO_FILE,
            np.float32,
            (self._capacity, *self._proprio_shape),
            mode,
        )
        self._action = _open_memmap(
            self._replay_dir / _MEMMAP_ACTION_FILE,
            np.float32,
            (self._capacity, *self._action_shape),
            mode,
        )
        self._reward = _open_memmap(self._replay_dir / _MEMMAP_REWARD_FILE, np.float32, (self._capacity, 1), mode)
        self._discount = _open_memmap(
            self._replay_dir / _MEMMAP_DISCOUNT_FILE,
            np.float32,
            (self._capacity, 1),
            mode,
        )
        self._done = _open_memmap(self._replay_dir / _MEMMAP_DONE_FILE, np.bool_, (self._capacity, 1), mode)
        self._state_id = _open_memmap(self._replay_dir / _MEMMAP_STATE_ID_FILE, np.int64, (self._capacity,), mode)
        self._transition_id = _open_memmap(
            self._replay_dir / _MEMMAP_TRANSITION_ID_FILE,
            np.int64,
            (self._capacity,),
            mode,
        )
        self._episode_id = _open_memmap(self._replay_dir / _MEMMAP_EPISODE_ID_FILE, np.int64, (self._capacity,), mode)
        self._episode_step = _open_memmap(
            self._replay_dir / _MEMMAP_EPISODE_STEP_FILE,
            np.int64,
            (self._capacity,),
            mode,
        )
        self._meta = _open_memmap(self._replay_dir / _MEMMAP_META_FILE, np.int64, (3,), mode)

    def _flush_metadata(self) -> None:
        self._state_id.flush()
        self._transition_id.flush()
        self._episode_id.flush()
        self._episode_step.flush()
        self._meta.flush()

    def __len__(self) -> int:
        return int(min(int(self._meta[_META_NUM_TRANSITIONS]), self._max_size))

    def _slot(self, state_id: int) -> int:
        return int(state_id % self._capacity)

    def _write_state(self, obs: np.ndarray, proprio: np.ndarray, episode_id: int, episode_step: int) -> int:
        state_id = int(self._meta[_META_NEXT_STATE_ID])
        slot = self._slot(state_id)
        self._obs[slot] = np.asarray(obs, dtype=self._obs_dtype)
        self._proprio[slot] = np.asarray(proprio, dtype=np.float32)
        self._state_id[slot] = state_id
        self._episode_id[slot] = int(episode_id)
        self._episode_step[slot] = int(episode_step)
        self._meta[_META_NEXT_STATE_ID] = state_id + 1
        return state_id

    def add_initial(self, obs: np.ndarray, proprio: np.ndarray) -> None:
        episode_id = int(self._meta[_META_NEXT_EPISODE_ID])
        self._current_state_id = self._write_state(obs, proprio, episode_id=episode_id, episode_step=0)
        self._flush_metadata()

    def add(
        self,
        action: np.ndarray,
        reward: float,
        discount: float,
        next_obs: np.ndarray,
        next_proprio: np.ndarray,
        done: bool,
    ) -> None:
        if self._current_state_id is None:
            raise RuntimeError("add_initial must be called before add.")

        current_id = int(self._current_state_id)
        current_slot = self._slot(current_id)
        episode_id = int(self._episode_id[current_slot])
        episode_step = int(self._episode_step[current_slot])

        self._action[current_slot] = np.asarray(action, dtype=np.float32)
        self._reward[current_slot] = np.asarray([reward], dtype=np.float32)
        self._discount[current_slot] = np.asarray([discount], dtype=np.float32)
        self._done[current_slot] = np.asarray([done], dtype=np.bool_)
        self._transition_id[current_slot] = current_id

        next_id = self._write_state(next_obs, next_proprio, episode_id=episode_id, episode_step=episode_step + 1)
        self._meta[_META_NUM_TRANSITIONS] = int(self._meta[_META_NUM_TRANSITIONS]) + 1
        if done:
            self._meta[_META_NEXT_EPISODE_ID] = episode_id + 1
            self._current_state_id = None
        else:
            self._current_state_id = next_id
        self._flush_metadata()


class MemmapReplayBuffer(IterableDataset):
    """Memmap-backed sampler that reconstructs frame stacks at sample time."""

    def __init__(
        self,
        replay_dir: Path,
        obs_shape: Tuple[int, ...],
        obs_dtype: np.dtype,
        proprio_shape: Tuple[int, ...],
        action_shape: Tuple[int, ...],
        max_size: int,
        frame_stack: int,
        nstep: int,
        discount: float,
    ) -> None:
        super().__init__()
        self._replay_dir = Path(replay_dir)
        self._obs_shape = tuple(obs_shape)
        self._obs_dtype = np.dtype(obs_dtype)
        self._proprio_shape = tuple(proprio_shape)
        self._action_shape = tuple(action_shape)
        self._max_size = int(max_size)
        self._frame_stack = int(frame_stack)
        self._nstep = int(nstep)
        self._discount_gamma = float(discount)
        self._capacity = self._max_size + self._frame_stack + self._nstep + 1
        self._opened = False

    def _open(self) -> None:
        if self._opened:
            return
        self._obs = _open_memmap(
            self._replay_dir / _MEMMAP_OBS_FILE,
            self._obs_dtype,
            (self._capacity, *self._obs_shape),
            "r",
        )
        self._proprio = _open_memmap(
            self._replay_dir / _MEMMAP_PROPRIO_FILE,
            np.float32,
            (self._capacity, *self._proprio_shape),
            "r",
        )
        self._action = _open_memmap(
            self._replay_dir / _MEMMAP_ACTION_FILE,
            np.float32,
            (self._capacity, *self._action_shape),
            "r",
        )
        self._reward = _open_memmap(self._replay_dir / _MEMMAP_REWARD_FILE, np.float32, (self._capacity, 1), "r")
        self._discount = _open_memmap(
            self._replay_dir / _MEMMAP_DISCOUNT_FILE,
            np.float32,
            (self._capacity, 1),
            "r",
        )
        self._done = _open_memmap(self._replay_dir / _MEMMAP_DONE_FILE, np.bool_, (self._capacity, 1), "r")
        self._state_id = _open_memmap(self._replay_dir / _MEMMAP_STATE_ID_FILE, np.int64, (self._capacity,), "r")
        self._transition_id = _open_memmap(
            self._replay_dir / _MEMMAP_TRANSITION_ID_FILE,
            np.int64,
            (self._capacity,),
            "r",
        )
        self._episode_id = _open_memmap(self._replay_dir / _MEMMAP_EPISODE_ID_FILE, np.int64, (self._capacity,), "r")
        self._episode_step = _open_memmap(
            self._replay_dir / _MEMMAP_EPISODE_STEP_FILE,
            np.int64,
            (self._capacity,),
            "r",
        )
        self._meta = _open_memmap(self._replay_dir / _MEMMAP_META_FILE, np.int64, (3,), "r")
        self._opened = True

    def __len__(self) -> int:
        if not self._opened:
            return 0
        return int(min(int(self._meta[_META_NUM_TRANSITIONS]), self._max_size))

    def update_nstep(self, nstep: int) -> None:
        self._nstep = int(nstep)

    def _slot(self, state_id: int) -> int:
        return int(state_id % self._capacity)

    def _has_state(self, state_id: int) -> bool:
        return int(self._state_id[self._slot(state_id)]) == int(state_id)

    def _has_transition(self, state_id: int) -> bool:
        return int(self._transition_id[self._slot(state_id)]) == int(state_id)

    def _episode(self, state_id: int) -> int:
        return int(self._episode_id[self._slot(state_id)])

    def _episode_step_at(self, state_id: int) -> int:
        return int(self._episode_step[self._slot(state_id)])

    def _valid_start(self, state_id: int, next_state_id: int) -> bool:
        if not self._has_state(state_id) or not self._has_state(state_id + self._nstep):
            return False
        episode_id = self._episode(state_id)
        if self._episode(state_id + self._nstep) != episode_id:
            return False
        for offset in range(self._nstep):
            sid = state_id + offset
            if not self._has_transition(sid) or self._episode(sid) != episode_id:
                return False
        return state_id + self._nstep < next_state_id

    def _sample_start_id(self) -> int:
        while True:
            next_state_id = int(self._meta[_META_NEXT_STATE_ID])
            num_transitions = int(min(int(self._meta[_META_NUM_TRANSITIONS]), self._max_size))
            if num_transitions >= self._nstep:
                low = max(0, next_state_id - self._max_size)
                high = next_state_id - self._nstep
                if high <= low:
                    time.sleep(0.05)
                    continue
                for _ in range(1024):
                    state_id = random.randrange(low, high)
                    if self._valid_start(state_id, next_state_id):
                        return state_id
            time.sleep(0.05)

    def _read_obs(self, state_id: int) -> np.ndarray:
        if not self._has_state(state_id):
            raise RuntimeError(f"State {state_id} has been overwritten.")
        return np.asarray(self._obs[self._slot(state_id)])

    def _stack_state(self, state_id: int) -> np.ndarray:
        episode_step = self._episode_step_at(state_id)
        obs = []
        for offset in range(self._frame_stack - 1, -1, -1):
            sid = state_id - min(offset, episode_step)
            obs.append(self._read_obs(sid))
        if len(self._obs_shape) == 3 and self._obs_shape[0] == 3:
            return np.concatenate(obs, axis=0)
        return np.stack(obs, axis=0)

    def _sample(self):
        self._open()
        state_id = self._sample_start_id()
        next_state_id = state_id + self._nstep

        obs = self._stack_state(state_id)
        proprio = np.asarray(self._proprio[self._slot(state_id)])
        action = np.asarray(self._action[self._slot(state_id)])
        next_obs = self._stack_state(next_state_id)
        next_proprio = np.asarray(self._proprio[self._slot(next_state_id)])

        reward = np.zeros((1,), dtype=np.float32)
        discount = np.ones((1,), dtype=np.float32)
        for offset in range(self._nstep):
            slot = self._slot(state_id + offset)
            reward += discount * np.asarray(self._reward[slot])
            discount *= np.asarray(self._discount[slot]) * self._discount_gamma
        return obs, proprio, action, reward, discount, next_obs, next_proprio

    def __iter__(self):
        while True:
            yield self._sample()


def _seed_worker(worker_id: int) -> None:
    seed = np.random.randint(0, 2**31 - 1) + worker_id
    np.random.seed(seed)
    random.seed(seed)


def make_memmap_replay_loader(
    replay_dir: Path,
    obs_shape: Tuple[int, ...],
    obs_dtype: np.dtype,
    proprio_shape: Tuple[int, ...],
    action_shape: Tuple[int, ...],
    max_size: int,
    batch_size: int,
    num_workers: int,
    frame_stack: int,
    nstep: int,
    discount: float,
) -> tuple[DataLoader, MemmapReplayBuffer]:
    dataset = MemmapReplayBuffer(
        replay_dir=replay_dir,
        obs_shape=obs_shape,
        obs_dtype=obs_dtype,
        proprio_shape=proprio_shape,
        action_shape=action_shape,
        max_size=max_size,
        frame_stack=frame_stack,
        nstep=nstep,
        discount=discount,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=_seed_worker,
    )
    return loader, dataset
