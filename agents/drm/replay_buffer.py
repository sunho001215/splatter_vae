import random
import datetime
import io
import time
import traceback
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Tuple
from collections import defaultdict, deque
from torch.utils.data import DataLoader, IterableDataset

def episode_len(episode: Dict[str, np.ndarray]) -> int:
    """
    Number of valid transitions in an episode.

    We store a dummy first transition (reset observation + zero action/reward),
    exactly like the original DrM code, so:
        true_length = len(observation) - 1
    """
    return int(episode["observation"].shape[0] - 1)


def save_episode(episode: Dict[str, np.ndarray], fn: Path) -> None:
    """Save one complete episode as a compressed .npz file."""
    with io.BytesIO() as bs:
        np.savez_compressed(bs, **episode)
        bs.seek(0)
        with fn.open("wb") as f:
            f.write(bs.read())


def load_episode(fn: Path) -> Dict[str, np.ndarray]:
    """Load one complete episode from disk."""
    with fn.open("rb") as f:
        data = np.load(f)
        return {k: data[k] for k in data.keys()}


class ReplayBufferStorage:
    """
    Episode writer.

    This mirrors the structure of the original DrM code:
    - collect one episode in RAM
    - once the episode ends, save it to disk as compressed .npz
    - keep only metadata counters in memory
    """

    def __init__(self, replay_dir: Path, action_shape: Tuple[int, ...]) -> None:
        self._replay_dir = Path(replay_dir)
        self._replay_dir.mkdir(parents=True, exist_ok=True)

        self._action_shape = tuple(action_shape)
        self._current_episode: Dict[str, list[np.ndarray]] = defaultdict(list)

        self._num_episodes = 0
        self._num_transitions = 0

        # Preload existing files if they already exist.
        for fn in self._replay_dir.glob("*.npz"):
            try:
                _, _, eps_len = fn.stem.split("_")
                self._num_episodes += 1
                self._num_transitions += int(eps_len)
            except Exception:
                pass

    def __len__(self) -> int:
        return self._num_transitions

    def add_initial(self, obs: np.ndarray) -> None:
        """
        Add the dummy first transition.

        This keeps the layout identical to the original DrM replay format:
        - observation[0] is the reset observation
        - action[0], reward[0], discount[0] are dummy values
        """
        self._current_episode["observation"].append(np.asarray(obs, dtype=np.uint8))
        self._current_episode["action"].append(np.zeros(self._action_shape, dtype=np.float32))
        self._current_episode["reward"].append(np.zeros((1,), dtype=np.float32))
        self._current_episode["discount"].append(np.ones((1,), dtype=np.float32))

    def add(
        self,
        action: np.ndarray,
        reward: float,
        discount: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """
        Add one real environment transition.

        We append:
        - next observation
        - action that produced it
        - reward
        - discount
        """
        self._current_episode["observation"].append(np.asarray(next_obs, dtype=np.uint8))
        self._current_episode["action"].append(np.asarray(action, dtype=np.float32))
        self._current_episode["reward"].append(np.asarray([reward], dtype=np.float32))
        self._current_episode["discount"].append(np.asarray([discount], dtype=np.float32))

        if done:
            episode = {
                k: np.asarray(v)
                for k, v in self._current_episode.items()
            }
            self._current_episode = defaultdict(list)
            self._store_episode(episode)

    def _store_episode(self, episode: Dict[str, np.ndarray]) -> None:
        """Write one finished episode to disk."""
        eps_idx = self._num_episodes
        eps_len = episode_len(episode)

        self._num_episodes += 1
        self._num_transitions += eps_len

        ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        eps_fn = self._replay_dir / f"{ts}_{eps_idx}_{eps_len}.npz"
        save_episode(episode, eps_fn)


class ReplayBuffer(IterableDataset):
    """
    DrM-style replay sampler.

    - Loads saved episodes lazily from disk
    - Maintains an in-memory cache of recent episodes
    - Samples n-step transitions
    """

    def __init__(
        self,
        replay_dir: Path,
        max_size: int,
        num_workers: int,
        nstep: int,
        discount: float,
        fetch_every: int,
        save_snapshot: bool,
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

    def __len__(self) -> int:
        return self._size

    def update_nstep(self, nstep: int) -> None:
        """Kept for compatibility with the official DrM pattern."""
        self._nstep = int(nstep)

    def _store_episode(self, eps_fn: Path) -> bool:
        """Load one episode file into the in-memory cache."""
        try:
            episode = load_episode(eps_fn)
        except Exception:
            return False

        eps_len = episode_len(episode)

        # Evict oldest cached episodes if needed.
        while self._episode_fns and (eps_len + self._size > self._max_size):
            early_fn = self._episode_fns.pop(0)
            early_eps = self._episodes.pop(early_fn)
            self._size -= episode_len(early_eps)
            early_fn.unlink(missing_ok=True)

        self._episode_fns.append(eps_fn)
        self._episode_fns.sort()
        self._episodes[eps_fn] = episode
        self._size += eps_len

        # Same behavior as the official code: optionally delete source file
        # after loading, so disk usage stays bounded.
        if not self._save_snapshot:
            eps_fn.unlink(missing_ok=True)

        return True

    def _try_fetch(self) -> None:
        """Periodically scan replay_dir for newly written episode files."""
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

            # Same sharding idea as the original DrM code.
            if eps_idx % self._num_workers != worker_id:
                continue

            if eps_fn in self._episodes:
                break

            if fetched_size + eps_len > self._max_size:
                break

            fetched_size += eps_len
            if not self._store_episode(eps_fn):
                break

    def _sample_episode(self) -> Dict[str, np.ndarray]:
        """
        Sample a cached episode that is long enough for current n-step.
        """
        valid_fns = [
            fn for fn in self._episode_fns
            if episode_len(self._episodes[fn]) >= self._nstep
        ]
        if not valid_fns:
            raise RuntimeError("No replay episodes are long enough for current n-step.")
        eps_fn = random.choice(valid_fns)
        return self._episodes[eps_fn]

    def _sample(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample one n-step transition.

        This matches the official DrM indexing convention:
        - obs      = observation[idx - 1]
        - action   = action[idx]
        - next_obs = observation[idx + nstep - 1]
        """
        try:
            self._try_fetch()
        except Exception:
            traceback.print_exc()

        self._samples_since_last_fetch += 1

        # Wait until at least one episode becomes available.
        while True:
            if self._episode_fns:
                try:
                    episode = self._sample_episode()
                    break
                except RuntimeError:
                    pass

            # While waiting for new episodes, force another directory scan.
            # Without this, a worker can get stuck forever if it started when
            # the replay directory was empty.
            time.sleep(0.05)
            self._samples_since_last_fetch = self._fetch_every
            self._try_fetch()

        idx = np.random.randint(0, episode_len(episode) - self._nstep + 1) + 1

        obs = episode["observation"][idx - 1]
        action = episode["action"][idx]
        next_obs = episode["observation"][idx + self._nstep - 1]

        reward = np.zeros((1,), dtype=np.float32)
        discount = np.ones((1,), dtype=np.float32)

        # Standard n-step return accumulation.
        for i in range(self._nstep):
            reward += discount * episode["reward"][idx + i]
            discount *= episode["discount"][idx + i] * self._discount

        return obs, action, reward, discount, next_obs

    def __iter__(self):
        while True:
            yield self._sample()


def _seed_worker(worker_id: int) -> None:
    """Make replay sampling different across workers."""
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
    fetch_every: int = 1000,
) -> tuple[DataLoader, ReplayBuffer]:
    """
    Build a DataLoader on top of the iterable replay dataset.

    This keeps the same high-level pattern as the original DrM code:
        replay_loader, replay_buffer = make_replay_loader(...)
        replay_iter = iter(replay_loader)
    """
    dataset = ReplayBuffer(
        replay_dir=replay_dir,
        max_size=max_size,
        num_workers=num_workers,
        nstep=nstep,
        discount=discount,
        fetch_every=fetch_every,
        save_snapshot=save_snapshot,
    )

    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=_seed_worker,
    )
    return loader, dataset