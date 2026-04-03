from __future__ import annotations

import datetime
import io
import random
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset


def episode_len(episode: Dict[str, np.ndarray]) -> int:
    """Calculate the length of an episode (number of transitions)."""
    return int(episode["obs"].shape[0] - 1)


def save_episode(episode: Dict[str, np.ndarray], fn: Path) -> None:
    """Save an episode to a compressed NPZ file."""
    with io.BytesIO() as bs:
        np.savez_compressed(bs, **episode)
        bs.seek(0)
        with fn.open("wb") as f:
            f.write(bs.read())


def load_episode(fn: Path) -> Dict[str, np.ndarray]:
    """Load an episode from a compressed NPZ file."""
    with fn.open("rb") as f:
        data = np.load(f)
        return {k: data[k] for k in data.keys()}


class ReplayBufferStorage:
    """Storage class for replay buffer, handling episode saving to disk."""

    def __init__(
        self,
        replay_dir: Path,
        obs_shape: Tuple[int, ...],
        obs_dtype: np.dtype,
        proprio_shape: Tuple[int, ...],
        action_shape: Tuple[int, ...]
    ) -> None:
        self._replay_dir = Path(replay_dir)
        self._replay_dir.mkdir(parents=True, exist_ok=True)
        self._obs_dtype = np.dtype(obs_dtype)
        self._action_shape = tuple(action_shape)
        self._current_episode: Dict[str, list[np.ndarray]] = defaultdict(list)
        self._num_episodes = 0
        self._num_transitions = 0
        # Count existing episodes and transitions
        for fn in self._replay_dir.glob("*.npz"):
            try:
                _, _, eps_len = fn.stem.split("_")
                self._num_episodes += 1
                self._num_transitions += int(eps_len)
            except Exception:
                pass

    def __len__(self) -> int:
        """Return the total number of transitions in the buffer."""
        return self._num_transitions

    def add_initial(self, obs: np.ndarray, proprio: np.ndarray) -> None:
        """Add the initial observation and proprioception to the current episode."""
        self._current_episode["obs"].append(np.asarray(obs, dtype=self._obs_dtype))
        self._current_episode["proprio"].append(np.asarray(proprio, dtype=np.float32))
        self._current_episode["action"].append(np.zeros(self._action_shape, dtype=np.float32))
        self._current_episode["reward"].append(np.zeros((1,), dtype=np.float32))
        self._current_episode["discount"].append(np.ones((1,), dtype=np.float32))

    def add(
        self,
        action: np.ndarray,
        reward: float,
        discount: float,
        next_obs: np.ndarray,
        next_proprio: np.ndarray,
        done: bool
    ) -> None:
        """Add a transition to the current episode. If done, store the episode."""
        self._current_episode["obs"].append(np.asarray(next_obs, dtype=self._obs_dtype))
        self._current_episode["proprio"].append(np.asarray(next_proprio, dtype=np.float32))
        self._current_episode["action"].append(np.asarray(action, dtype=np.float32))
        self._current_episode["reward"].append(np.asarray([reward], dtype=np.float32))
        self._current_episode["discount"].append(np.asarray([discount], dtype=np.float32))
        if done:
            episode = {k: np.asarray(v) for k, v in self._current_episode.items()}
            self._current_episode = defaultdict(list)
            self._store_episode(episode)

    def _store_episode(self, episode):
        """Store episode, injecting multi-copy metadata for the sampler."""
        eps_idx = self._num_episodes
        eps_len = episode_len(episode)
        self._num_episodes += 1
        self._num_transitions += eps_len
    
        # Detect multi-copy from obs shape.
        # If obs_shape was set with K prepended, obs.shape = (T+1, K, *feat_shape).
        # We record K so the sampler knows to do random copy selection.
        obs = episode["obs"]
        if hasattr(self, "_num_aug_copies") and self._num_aug_copies > 1:
            episode["obs_num_copies"] = np.array(self._num_aug_copies, dtype=np.int32)
    
        ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        save_episode(episode, self._replay_dir / f"{ts}_{eps_idx}_{eps_len}.npz")


class ReplayBuffer(IterableDataset):
    """Replay buffer dataset for sampling transitions with n-step returns."""

    def __init__(
        self,
        replay_dir: Path,
        max_size: int,
        num_workers: int,
        nstep: int,
        discount: float,
        fetch_every: int,
        save_snapshot: bool,
        num_aug_copies: int = 7
    ) -> None:
        super().__init__()
        self._replay_dir = Path(replay_dir)
        self._max_size = int(max_size)
        self._num_workers = max(1, int(num_workers))
        self._size = 0
        self._episode_fns: list[Path] = []
        self._episodes: Dict[Path, Dict[str, np.ndarray]] = {}
        self._nstep = int(nstep)
        self._discount = float(discount)
        self._fetch_every = int(fetch_every)
        self._samples_since_last_fetch = self._fetch_every
        self._save_snapshot = bool(save_snapshot)
        self._num_aug_copies = int(num_aug_copies)

    def __len__(self) -> int:
        """Return the current number of transitions in the buffer."""
        return self._size

    def update_nstep(self, nstep: int) -> None:
        """Update the n-step parameter for sampling."""
        self._nstep = int(nstep)

    def _store_episode(self, eps_fn: Path) -> bool:
        """Load and store an episode in memory, evicting old ones if necessary."""
        try:
            episode = load_episode(eps_fn)
        except Exception:
            return False
        eps_len = episode_len(episode)
        while self._episode_fns and (eps_len + self._size > self._max_size):
            early_fn = self._episode_fns.pop(0)
            early_eps = self._episodes.pop(early_fn)
            self._size -= episode_len(early_eps)
            early_fn.unlink(missing_ok=True)
        self._episode_fns.append(eps_fn)
        self._episode_fns.sort()
        self._episodes[eps_fn] = episode
        self._size += eps_len
        if not self._save_snapshot:
            eps_fn.unlink(missing_ok=True)
        return True

    def _try_fetch(self) -> None:
        """Fetch new episodes from disk if it's time to do so."""
        if self._samples_since_last_fetch < self._fetch_every:
            return
        self._samples_since_last_fetch = 0
        try:
            worker_info = torch.utils.data.get_worker_info()
            worker_id = worker_info.id if worker_info is not None else 0
        except Exception:
            worker_id = 0

        eps_fns = sorted(self._replay_dir.glob("*.npz"), reverse=True)
        fetched_size = 0
        for eps_fn in eps_fns:
            try:
                eps_idx, eps_len = [int(x) for x in eps_fn.stem.split("_")[1:]]
            except Exception:
                continue
            if eps_idx % self._num_workers != worker_id:
                continue
            if eps_fn in self._episodes or fetched_size + eps_len > self._max_size:
                break
            fetched_size += eps_len
            if not self._store_episode(eps_fn):
                break

    def _sample_episode(self) -> Dict[str, np.ndarray]:
        """Sample a random episode that is long enough for n-step sampling."""
        valid_fns = [fn for fn in self._episode_fns if episode_len(self._episodes[fn]) >= self._nstep]
        if not valid_fns:
            raise RuntimeError("No replay episodes are long enough for current n-step.")
        return self._episodes[random.choice(valid_fns)]

    def _sample(self):
        """Sample a single transition, randomly selecting 1 of K cached copies."""
        try:
            self._try_fetch()
        except Exception:
            traceback.print_exc()
        self._samples_since_last_fetch += 1
    
        while True:
            if self._episode_fns:
                try:
                    episode = self._sample_episode()
                    break
                except RuntimeError:
                    pass
            time.sleep(0.05)
            self._samples_since_last_fetch = self._fetch_every
            self._try_fetch()
    
        idx = np.random.randint(0, episode_len(episode) - self._nstep + 1) + 1
    
        obs_raw = episode["obs"][idx - 1]
        next_obs_raw = episode["obs"][idx + self._nstep - 1]
    
        # Multi-copy select
        if "obs_num_copies" in episode:
            K = int(episode["obs_num_copies"])
            if K > 1:
                k_obs = np.random.randint(0, K)
                k_next = np.random.randint(0, K)
                obs_raw = obs_raw[k_obs]
                next_obs_raw = next_obs_raw[k_next]
    
        proprio = episode["proprio"][idx - 1]
        action = episode["action"][idx]
        next_proprio = episode["proprio"][idx + self._nstep - 1]
        reward = np.zeros((1,), dtype=np.float32)
        discount = np.ones((1,), dtype=np.float32)
        for i in range(self._nstep):
            reward += discount * episode["reward"][idx + i]
            discount *= episode["discount"][idx + i] * self._discount
        return obs_raw, proprio, action, reward, discount, next_obs_raw, next_proprio
 

    def __iter__(self):
        """Iterator for the dataset, yielding samples indefinitely."""
        while True:
            yield self._sample()


def _seed_worker(worker_id: int) -> None:
    """Seed the random number generators for data loading workers."""
    seed = np.random.randint(0, 2**31 - 1) + worker_id
    np.random.seed(seed)
    random.seed(seed)


def make_replay_loader(
    replay_dir: Path,
    max_size: int,
    batch_size: int,
    num_workers: int,
    save_snapshot: bool,
    nstep: int,
    discount: float,
    fetch_every: int = 1000
) -> tuple[DataLoader, ReplayBuffer]:
    """Create a DataLoader and ReplayBuffer for replay sampling."""
    dataset = ReplayBuffer(
        replay_dir,
        int(max_size) // max(1, int(num_workers)),
        num_workers,
        nstep,
        discount,
        fetch_every,
        save_snapshot
    )
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=_seed_worker
    )
    return loader, dataset
