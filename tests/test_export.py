"""The real exporter and loader preserve encoder-only CPU policy features."""

from __future__ import annotations

import importlib
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from s4d.config import dump_config
from s4d.model.decoder import DecoderConfig, GroupConfig
from s4d.model.encoder import EncoderConfig, load_encoder
from s4d.train.checkpoint import save_checkpoint
from s4d.train.loop import build_model


def test_real_encoder_export_reload_is_identical_and_contains_no_decoder(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    exporter = importlib.import_module("scripts.export_encoder")
    encoder_cfg = EncoderConfig(
        image_height=32, image_width=32, patch_size=8, width=16, depth=1, heads=4, num_slots=2, slot_dim=12, drop_path=0.0
    )
    decoder_cfg = DecoderConfig(
        slot_dim=12,
        dim=16,
        depth=2,
        heads=4,
        scene=GroupConfig(3, 2, 0.15, 0.06, 0.001, 0.08),
        dynamic=GroupConfig(2, 2, 0.4, 0.04, 0.0005, 0.03),
    )
    decoder_dict = asdict(decoder_cfg)
    decoder_dict.pop("slot_dim")
    config = {"model": {"encoder": asdict(encoder_cfg), "decoder": decoder_dict}, "run": {"name": "export-test"}}
    model = build_model(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    dump_config(config, tmp_path / "config.yaml")
    checkpoint = save_checkpoint(
        tmp_path / "checkpoints" / "step_0000007.pt", 7, model.encoder, model.decoder, optimizer, scheduler, config
    )
    images = torch.randint(0, 256, (2, 3, 3, 32, 32), dtype=torch.uint8)
    model.encoder.requires_grad_(False)  # Match the frozen loader's inference kernel selection.
    expected = model.encoder.policy_state(images)
    export_path = exporter.export(tmp_path, checkpoint, tmp_path / "encoder.pt")
    payload = torch.load(export_path, map_location="cpu", weights_only=True)
    assert "decoder" not in payload
    assert not any("decoder" in key for key in payload["state_dict"])
    assert payload["step"] == 7
    encoder = load_encoder(export_path)
    for key, value in encoder.state_dict().items():
        torch.testing.assert_close(value, model.encoder.state_dict()[key], atol=0, rtol=0)
    assert not encoder.training
    assert all(not parameter.requires_grad and parameter.device.type == "cpu" for parameter in encoder.parameters())
    torch.testing.assert_close(encoder.policy_state(images), expected, atol=0, rtol=0)


def test_encoder_loader_rejects_non_export_payload(tmp_path):
    path = tmp_path / "training_checkpoint.pt"
    torch.save({"encoder": {}}, path)
    with pytest.raises(ValueError, match="encoder-only export"):
        load_encoder(path)


def test_export_waits_for_a_checkpoint_that_appears_later(tmp_path, monkeypatch, capsys):
    import shutil
    import sys

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    exporter = importlib.import_module("scripts.export_encoder")
    encoder_cfg = EncoderConfig(image_height=32, image_width=32, patch_size=8, width=16, depth=1, heads=4, slot_dim=12)
    decoder_cfg = DecoderConfig(
        slot_dim=12,
        dim=16,
        depth=2,
        heads=4,
        scene=GroupConfig(3, 2, 0.15, 0.06, 0.001, 0.08),
        dynamic=GroupConfig(2, 2, 0.4, 0.04, 0.0005, 0.03),
    )
    decoder_dict = asdict(decoder_cfg)
    decoder_dict.pop("slot_dim")
    config = {"model": {"encoder": asdict(encoder_cfg), "decoder": decoder_dict}, "data": {"strides": [2, 4, 6]}}
    model = build_model(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    dump_config(config, tmp_path / "config.yaml")
    staged = save_checkpoint(
        tmp_path / "staged" / "step_0000100.pt", 100, model.encoder, model.decoder, optimizer, scheduler, config
    )
    target = tmp_path / "checkpoints" / "step_0000100.pt"
    target.parent.mkdir()
    sleeps = []

    def fake_sleep(seconds):  # the training run writes the checkpoint while the exporter waits
        sleeps.append(seconds)
        shutil.copyfile(staged, target)

    monkeypatch.setattr(exporter.time, "sleep", fake_sleep)
    out = tmp_path / "encoder.pt"
    monkeypatch.setattr(
        sys, "argv", ["export_encoder.py", "--run", str(tmp_path), "--checkpoint", str(target), "--out", str(out), "--wait"]
    )
    exporter.main()
    assert sleeps == [600] and "waiting for" in capsys.readouterr().out
    payload = torch.load(out, map_location="cpu", weights_only=True)
    assert payload["step"] == 100 and payload["frame_strides"] == [2, 4, 6]
