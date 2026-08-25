from __future__ import annotations

import numpy as np

from dataset.droid.cache import (
    CacheProvenance,
    HDF5CacheReader,
    HDF5ShardWriter,
    cache_item_key,
    sequence_cache_key,
)
from dataset.droid.dataset import DROIDDatasetConfig, DROIDLogicalDataset, droid_collate
from dataset.droid.rlds import MemoryEpisodeBackend
from dataset.droid.sampling import (
    EpisodeGroupedDistributedSampler,
    MotionCropConfig,
    TemporalSamplingConfig,
)


def _provenance(name: str) -> CacheProvenance:
    return CacheProvenance(
        teacher_name=name,
        checkpoint="test.ckpt",
        teacher_version="test",
        calibration_version="test",
        preprocessing_version="test",
        resolution=(320, 180),
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


def _write_caches(root):
    depth_root = root / "xlens"
    flow_root = root / "waft"
    with HDF5ShardWriter(
        depth_root, _provenance("xlens"), shard_prefix="depth"
    ) as writer:
        for camera in ("exterior_1", "exterior_2"):
            for frame in (0, 3, 6):
                writer.add(
                    cache_item_key("episode-test", camera, frame),
                    {
                        "metric_depth": np.full(
                            (180, 320), 1.0 + frame / 10, np.float32
                        ),
                        "confidence": np.ones((180, 320), np.float16),
                        "validity": np.ones((180, 320), np.uint8),
                    },
                    {"camera_serial": camera},
                )
    with HDF5ShardWriter(flow_root, _provenance("waft"), shard_prefix="flow") as writer:
        for camera in ("exterior_1", "exterior_2"):
            for frame, gap in ((0, 3), (3, 3), (0, 6)):
                flow = np.zeros((180, 320, 2), np.float16)
                if camera == "exterior_1":
                    flow[60, 100, 0] = 10.0
                else:
                    flow[120, 220, 0] = 10.0
                writer.add(
                    cache_item_key("episode-test", camera, frame, gap=gap),
                    {"forward_flow": flow, "validity": np.ones((180, 320), np.uint8)},
                    {"camera_serial": camera},
                )
    return HDF5CacheReader(depth_root / "depth-index.json"), HDF5CacheReader(
        flow_root / "flow-index.json"
    )


def test_logical_batch_contract_uses_history_and_no_segmentation(tmp_path) -> None:
    depth, flow = _write_caches(tmp_path)
    images = np.zeros((8, 2, 180, 320, 3), dtype=np.uint8)
    for timestep in range(8):
        images[timestep, 0, ..., 0] = timestep * 10
        images[timestep, 1, ..., 1] = timestep * 10
    dataset = DROIDLogicalDataset(
        DROIDDatasetConfig(
            split="validation",
            temporal=TemporalSamplingConfig(validation_stride=3),
            motion_crop=MotionCropConfig(
                min_size=180, max_size=180, flow_smoothing_kernel=1
            ),
        ),
        backend=MemoryEpisodeBackend({("train", 0): {"images": images}}),
        depth_cache=depth,
        flow_cache=flow,
        manifest_entries=[_entry()],
    )
    item = dataset[0]
    assert item["history_indices"].tolist() == [0, 3, 6]
    assert item["representation_histories"].shape == (2, 3, 3, 224, 224)
    assert item["representation_flows"].shape == (2, 2, 2, 224, 224)
    assert item["representation_validity"].shape == (2, 3, 1, 224, 224)
    assert item["target_rgb"].shape == (3, 2, 3, 224, 224)
    assert item["target_depth"].shape == (3, 2, 1, 224, 224)
    assert item["target_flow"].shape == (3, 2, 2, 224, 224)
    assert item["target_flow_confidence"].shape == (3, 2, 1, 224, 224)
    assert item["target_K"].shape == (3, 2, 3, 3)
    assert item["target_w2c"].shape == (3, 2, 4, 4)
    assert item["crop_metadata"]["crop_size"].tolist() == [180, 180]
    centers = set(
        zip(
            item["crop_metadata"]["crop_center_x"].tolist(),
            item["crop_metadata"]["crop_center_y"].tolist(),
            strict=True,
        )
    )
    assert centers == {(100, 130), (220, 190)}
    assert not item["crop_metadata"]["low_motion_fallback_used"].any()
    torch_K = item["target_K"]
    assert (torch_K[0] == torch_K[1]).all() and (torch_K[1] == torch_K[2]).all()
    assert "segmentation" not in item
    assert "semantic_mask" not in item
    assert "local_rgb_b" not in item
    assert "second_view_is_local" not in item
    assert (~item["target_image_validity"]).any()
    batch = droid_collate([item, item])
    assert batch["representation_histories"].shape == (2, 2, 3, 3, 224, 224)
    assert batch["crop_metadata"]["crop_size"].shape == (2, 2)
    assert batch["episode_id"] == ["episode-test", "episode-test"]


def test_optional_see3d_sequence_cache_is_transformed_into_batch(tmp_path) -> None:
    depth, flow = _write_caches(tmp_path)
    see3d_root = tmp_path / "see3d"
    with HDF5ShardWriter(
        see3d_root, _provenance("See3D+X-Lens"), shard_prefix="see3d"
    ) as writer:
        writer.add(
            sequence_cache_key("episode-test", "see3d"),
            {
                "timestep": np.asarray([6], np.int64),
                "generated_rgb": np.full((1, 180, 320, 3), 127, np.uint8),
                "confidence": np.full((1, 180, 320), 0.8, np.float16),
                "metric_depth": np.full((1, 180, 320), 1.2, np.float16),
                "depth_confidence": np.full((1, 180, 320), 0.7, np.float16),
                "depth_validity": np.ones((1, 180, 320), np.uint8),
                "geometry_supported_depth": np.ones((1, 180, 320), np.uint8),
                "virtual_K": np.asarray(
                    [[[200.0, 0.0, 160.0], [0.0, 200.0, 90.0], [0.0, 0.0, 1.0]]],
                    np.float32,
                ),
                "virtual_c2w": np.eye(4, dtype=np.float32)[None],
                "virtual_w2c": np.eye(4, dtype=np.float32)[None],
            },
            {"episode_id": "episode-test"},
        )
    images = np.zeros((8, 2, 180, 320, 3), dtype=np.uint8)
    dataset = DROIDLogicalDataset(
        DROIDDatasetConfig(
            split="validation",
            temporal=TemporalSamplingConfig(validation_stride=3),
            motion_crop=MotionCropConfig(
                min_size=180, max_size=180, flow_smoothing_kernel=1
            ),
        ),
        backend=MemoryEpisodeBackend({("train", 0): {"images": images}}),
        depth_cache=depth,
        flow_cache=flow,
        see3d_cache=HDF5CacheReader(see3d_root / "see3d-index.json"),
        manifest_entries=[_entry()],
    )
    item = dataset[0]
    assert item["synthetic_available"]
    assert item["synthetic_rgb"].shape == (3, 224, 224)
    assert item["synthetic_depth"].shape == (1, 224, 224)
    # Synthetic targets reuse camera A's variable-FOV transform and therefore
    # remain geometrically aligned with that transformed camera matrix.
    np.testing.assert_allclose(item["synthetic_K"], item["target_K"][0, 0])


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
