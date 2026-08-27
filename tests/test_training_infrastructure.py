from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
import yaml

from models.gaussian.parameterization import gaussian_params_per_gaussian
from models.splattervae import SplatterVAE, ViTSmallConfig
from models.training.config import TrainConfig
from models.training.distributed import DistributedContext
from models.training.loop import (
    TrainingState,
    _decoder_configurations_match,
    build_optimizer,
    load_checkpoint,
    save_checkpoint,
)
from models.training.schedules import cosine_learning_rate, resolve_warmup_steps
from scripts.train_droid import _validate_fixed_pipeline_contract


def _model() -> SplatterVAE:
    return SplatterVAE(
        vit_config=ViTSmallConfig(),
        gaussian_parameters_per_gaussian=gaussian_params_per_gaussian(1),
        decoder_config={
            "global_center": (0.5, 0.0, 0.5),
            "anchor_initial_spread": 0.45,
            "parent_displacement_scale": 0.25,
            "child_radius": 0.06,
        },
    )


def test_lr_scaling_and_warmup() -> None:
    config = TrainConfig()
    assert config.learning_rates(256) == (1.5e-4, 3.0e-4)
    assert config.learning_rates(128) == (7.5e-5, 1.5e-4)
    assert resolve_warmup_steps(config, 300_000) == 15_000
    first = cosine_learning_rate(
        0, peak_lr=1.5e-4, min_lr=1e-6, total_steps=300_000, warmup_steps=15_000
    )
    assert 0.0 < first < 1.5e-4


def test_unwrapped_checkpoint_round_trip(tmp_path) -> None:
    model = _model()
    config = TrainConfig()
    optimizer, _ = build_optimizer(model, config, effective_global_batch=256)
    state = TrainingState(epoch=2, next_batch_in_epoch=17, global_step=123)
    context = DistributedContext(
        rank=0,
        local_rank=0,
        world_size=1,
        device=torch.device("cpu"),
        process_group_initialized=False,
    )
    original = model.encoder.cls_token.detach().clone()
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, optimizer, state, config, context)
    with torch.no_grad():
        model.encoder.cls_token.add_(10.0)
    loaded = load_checkpoint(path, model, optimizer)
    assert loaded == state
    torch.testing.assert_close(model.encoder.cls_token, original)
    payload = torch.load(path, weights_only=False)
    assert not any(key.startswith("module.") for key in payload["model"])
    assert payload["checkpoint_schema_version"] == 2
    assert payload["world_size"] == 1
    assert len(payload["rng_by_rank"]) == 1


def test_decoder_configuration_accepts_float_round_trip_noise() -> None:
    saved = _model().decoder_configuration()
    runtime = copy.deepcopy(saved)
    runtime["anchor_initial_spread"] = float(saved["anchor_initial_spread"]) + 1e-9
    runtime["global_center"][2] = float(saved["global_center"][2]) - 1e-9
    assert _decoder_configurations_match(saved, runtime)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("anchor_initial_spread", 0.5),
        ("num_groups", 128),
        ("conditioning", "single_token"),
    ),
)
def test_decoder_configuration_rejects_material_mismatch(field, value) -> None:
    saved = _model().decoder_configuration()
    runtime = copy.deepcopy(saved)
    runtime[field] = value
    assert not _decoder_configurations_match(saved, runtime)


def test_fixed_pipeline_requires_native_two_iteration_memfof() -> None:
    path = Path("config/splattervae/droid/pretrain.yaml")
    config = yaml.safe_load(path.read_text())
    _validate_fixed_pipeline_contract(config)
    mismatched = copy.deepcopy(config)
    mismatched["flow"]["iterations"] = 8
    with pytest.raises(ValueError, match="flow.iterations"):
        _validate_fixed_pipeline_contract(mismatched)


def test_lagernvs_configuration_has_no_probability_gate() -> None:
    path = Path("config/splattervae/droid/pretrain.yaml")
    config = yaml.safe_load(path.read_text())
    config["novel_view"]["probability"] = 0.2
    with pytest.raises(ValueError, match="boolean_only"):
        _validate_fixed_pipeline_contract(config)
