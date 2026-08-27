from __future__ import annotations

import numpy as np
import torch

from dataset.droid.dataset import DROIDDatasetConfig, DROIDLogicalDataset, droid_collate
from dataset.droid.rlds import MemoryEpisodeBackend
from dataset.droid.sampling import (
    EpisodeGroupedDistributedSampler,
    MotionCropConfig,
    TemporalSamplingConfig,
)


def _entry() -> dict:
    cameras = []
    for index, x in enumerate((0.4, -0.4), start=1):
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, 3] = (x, -0.3, 0.8)
        cameras.append(
            {
                "logical_id": f"exterior_{index}",
                "serial": str(index) * 3,
                "intrinsics_rlds": [
                    [200.0, 0.0, 160.0],
                    [0.0, 200.0, 90.0],
                    [0.0, 0.0, 1.0],
                ],
                "c2w": c2w.tolist(),
                "w2c": np.linalg.inv(c2w).tolist(),
            }
        )
    return {
        "episode_id": "episode-test",
        "rlds_split": "train",
        "rlds_ordinal": 0,
        "num_steps": 8,
        "dataset_split": "validation",
        "valid": True,
        "exterior_cameras": cameras,
    }


def test_logical_dataset_is_raw_read_only_teacher_input() -> None:
    images = np.zeros((8, 2, 180, 320, 3), dtype=np.uint8)
    for timestep in range(8):
        images[timestep, 0, ..., 0] = timestep * 10
        images[timestep, 1, ..., 1] = timestep * 10
    dataset = DROIDLogicalDataset(
        DROIDDatasetConfig(
            split="validation",
            temporal=TemporalSamplingConfig(validation_stride=3),
            motion_crop=MotionCropConfig(min_size=180, max_size=180),
        ),
        backend=MemoryEpisodeBackend({("train", 0): {"images": images}}),
        manifest_entries=[_entry()],
    )
    item = dataset[0]
    assert item["history_indices"].tolist() == [0, 3, 6]
    assert item["raw_histories"].shape == (2, 3, 3, 180, 320)
    assert item["raw_histories"].dtype == torch.uint8
    assert item["raw_K"].shape == (2, 3, 3)
    assert item["raw_c2w"].shape == item["raw_w2c"].shape == (2, 4, 4)
    assert int(item["sampled_crop_size"]) == 180
    assert "representation_histories" not in item
    assert "target_depth" not in item
    assert "target_flow" not in item
    assert "segmentation" not in item
    assert "semantic_mask" not in item
    batch = droid_collate([item, item])
    assert batch["raw_histories"].shape == (2, 2, 3, 3, 180, 320)
    assert batch["sampled_crop_size"].shape == (2,)
    assert batch["episode_id"] == ["episode-test", "episode-test"]


def test_episode_grouped_distributed_sampler_has_equal_disjoint_rank_shards() -> None:
    class _Dataset:
        episode_index_ranges = ((0, 4), (4, 7), (7, 12))

        def __len__(self):
            return 12

    dataset = _Dataset()
    first = EpisodeGroupedDistributedSampler(
        dataset, num_replicas=2, rank=0, shuffle=True, seed=9, drop_last=True
    )
    second = EpisodeGroupedDistributedSampler(
        dataset, num_replicas=2, rank=1, shuffle=True, seed=9, drop_last=True
    )
    first.set_epoch(3)
    second.set_epoch(3)
    rank_zero = list(first)
    rank_one = list(second)
    assert len(rank_zero) == len(rank_one) == 6
    assert set(rank_zero).isdisjoint(rank_one)
    assert sorted(rank_zero + rank_one) == list(range(12))
