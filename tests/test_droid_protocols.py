"""Protocol unit tests only; no synthetic result substitutes for native DROID acceptance."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from s4d.config import get, load_config
from s4d.data.contract import collate, validate_batch
from s4d.data.droid.dataset import DroidCacheDataset

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def protocol():
    spec = importlib.util.spec_from_file_location("droid_compare_protocol", REPO / "scripts/droid_compare_step0.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_overfit_configuration_keeps_exact_one_batch_and_original_warmups():
    for overlays in ([], [REPO / "configs/droid/dinov2.yaml"]):
        cfg = load_config([REPO / "configs/droid/pretrain.yaml", *overlays, REPO / "configs/droid/overfit.yaml"])
        assert get(cfg, "data.fixed_window_indices") == [0, 1]
        assert get(cfg, "train.batch_size") == 2
        assert get(cfg, "train.steps") == 2000
        assert get(cfg, "loss.ramp_steps") == 20000
        assert get(cfg, "loss.depth_align_warmup_steps") == 5000
        assert get(cfg, "train.eval_at_start")


def test_parser_partition_and_repository_boundary(protocol, tmp_path):
    args = protocol.argument_parser().parse_args(["--config", "base.yaml", "--output", str(tmp_path / "single.json")])
    assert args.indices == [0, 1, 2, 3] and args.reference is None
    assert protocol.partition_indices(args.indices, 8, 1, 0) == args.indices
    assert protocol.partition_indices(args.indices, 8, 2, 0) == [0, 1]
    assert protocol.partition_indices(args.indices, 8, 2, 1) == [2, 3]
    for indices, length, world, rank in (([0, 0, 1, 2], 4, 2, 0), ([0, 1, 2, 4], 4, 2, 1), ([0, 1], 4, 1, 0)):
        with pytest.raises(ValueError):
            protocol.partition_indices(indices, length, world, rank)
    assert protocol.repository_path(tmp_path / "result.json").parent == tmp_path
    with pytest.raises(ValueError, match="inside"):
        protocol.repository_path("/home/ws/data/comparison.json")


def test_comparison_tolerance_and_mismatched_inputs_fail_closed(protocol):
    common = {
        "config_sha256": "config",
        "model_sha256": "model",
        "batch_sha256": "batch",
        "indices": [0, 1, 2, 3],
        "source_views": [0, 1, 0, 1],
    }
    reference = dict(common, format=protocol.FORMAT, world_size=1, loss=3.0)
    ranks = [dict(common, rank=0, loss=2.0), dict(common, rank=1, loss=4.0)]
    result = protocol.comparison_result(reference, ranks)
    assert result["passed"] and result["absolute_error"] == 0
    ranks[1]["loss"] += 0.001
    assert not protocol.comparison_result(reference, ranks)["passed"]
    ranks[1]["batch_sha256"] = "wrong"
    with pytest.raises(ValueError, match="batch_sha256"):
        protocol.comparison_result(reference, ranks)
    ranks[1]["batch_sha256"] = "batch"
    ranks[1]["loss"] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        protocol.comparison_result(reference, ranks)


def test_comparison_settings_do_not_mutate_training_config(protocol):
    cfg = load_config([REPO / "configs/droid/pretrain.yaml"])
    deterministic = protocol.comparison_config(cfg)
    assert deterministic["train"]["bf16"] is False
    assert deterministic["model"]["encoder"]["drop_path"] == 0
    assert cfg["train"]["bf16"] is True and cfg["model"]["encoder"]["drop_path"] == 0.1
    tensor = torch.arange(8).reshape(2, 4)
    assert protocol.tensor_fingerprint({"tensor": tensor}) == protocol.tensor_fingerprint({"tensor": tensor.clone()})
    assert protocol.tensor_fingerprint({"tensor": tensor}) != protocol.tensor_fingerprint({"tensor": tensor + 1})


def test_fixed_indices_select_same_global_batch_from_synthetic_cache(protocol, tmp_path):
    height, width = 144, 256
    windows = [
        {
            "split": "train",
            "clip": "0:11",
            "canonical_indices": [i, i + 3, i + 6],
            "raw_indices": [2 * i, 2 * i + 6, 2 * i + 12],
        }
        for i in range(4)
    ]
    (tmp_path / "manifest.json").write_text(
        json.dumps({"episode": "synthetic", "camera_serials": ["a", "b"], "windows": windows})
    )
    K = np.array([[100, 0, width / 2], [0, 100, height / 2], [0, 0, 1]], np.float32)
    values = {
        "depth": np.ones((3, 2, 1, height, width), np.float32),
        "motion3d": np.zeros((3, 2, 3, height, width), np.float32),
        "motion_weight": np.ones((3, 2, 1, height, width), np.float32),
        "motion_score": np.zeros((3, 2, 1, height, width), np.float32),
        "K": np.stack([K, K]),
        "w2c": np.stack([np.eye(4, dtype=np.float32)] * 2),
        "c2w": np.stack([np.eye(4, dtype=np.float32)] * 2),
    }
    with h5py.File(tmp_path / "scratch.h5", "w") as cache:
        for index in range(4):
            group = cache.create_group(f"windows/{index:05d}")
            group["images"] = np.full((3, 2, 3, height, width), index, np.uint8)
            for name, value in values.items():
                group[name] = value
    dataset = DroidCacheDataset(tmp_path)
    try:
        indices = [3, 0, 2, 1]
        global_batch = collate([dataset[index] for index in indices])
        batches = [
            collate([dataset[index] for index in protocol.partition_indices(indices, 4, 2, rank)]) for rank in range(2)
        ]
        assert validate_batch(global_batch)["B"] == 4
        torch.testing.assert_close(torch.cat([batch["images"] for batch in batches]), global_batch["images"])
        # A CPU-only fake mean loss tests partition arithmetic, not rasterization.
        synthetic_loss = lambda batch: batch["images"].float().mean()
        assert sum(float(synthetic_loss(batch)) for batch in batches) / 2 == float(synthetic_loss(global_batch))
    finally:
        dataset.close()


def test_native_preflight_failure_precedes_any_cache_read(protocol, tmp_path, monkeypatch):
    def blocked():
        raise RuntimeError("explicit mocked native preflight failure")

    def forbidden_cache(*args, **kwargs):
        raise AssertionError("cache must not be read after blocked native preflight")

    monkeypatch.setattr(protocol, "require_prebuilt_renderer", blocked)
    monkeypatch.setattr(protocol, "DroidCacheDataset", forbidden_cache)
    with pytest.raises(RuntimeError, match="mocked native preflight"):
        protocol.run({}, [0, 1, 2, 3], tmp_path / "blocked.json", None)
    assert not (tmp_path / "blocked.json").exists()


def test_single_record_path_with_explicit_cpu_fake_loss(protocol, batch, tmp_path, monkeypatch):
    from s4d.train.ddp import DistContext

    samples = [
        {
            key: ({name: values[i % 2] for name, values in value.items()} if key == "meta" else value[i % 2].clone())
            for key, value in batch.items()
        }
        for i in range(4)
    ]

    class SyntheticDataset:
        def __init__(self, *args, **kwargs):
            pass

        def __len__(self):
            return len(samples)

        def __getitem__(self, index):
            return samples[index]

        def close(self):
            pass

    class SyntheticModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Linear(2, 2)
            self.decoder = torch.nn.Linear(2, 2)

    def fake_forward(model, values, cfg, step, *, source, mask_ratio):
        assert model.training and not model.encoder.training and not model.decoder.training
        assert step == 0 and mask_ratio == 0
        assert source.tolist() == [0, 1, 0, 1]
        assert values["images"].device.type == "cpu"
        return {"total": torch.tensor(2.5)}  # Explicit fake, no rendering or native acceptance.

    monkeypatch.setattr(protocol, "require_prebuilt_renderer", lambda: None)
    monkeypatch.setattr(protocol.ddp, "init_distributed", lambda: DistContext(0, 0, 1, torch.device("cpu")))
    monkeypatch.setattr(protocol, "DroidCacheDataset", SyntheticDataset)
    monkeypatch.setattr(protocol, "build_model", lambda cfg: SyntheticModel())
    monkeypatch.setattr(protocol, "forward_losses", fake_forward)
    cfg = load_config([REPO / "configs/droid/pretrain.yaml"], ["model.anchor_stats=null"])
    path = tmp_path / "synthetic_single.json"
    protocol.run(cfg, [0, 1, 2, 3], path, None)
    record = json.loads(path.read_text())
    assert record["loss"] == 2.5 and record["world_size"] == 1
    assert record["source_views"] == [0, 1, 0, 1]
    assert len(record["model_sha256"]) == len(record["batch_sha256"]) == 64
