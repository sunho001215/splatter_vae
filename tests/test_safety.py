from __future__ import annotations

import os

import pytest

from dataset.droid.safety import DEFAULT_DROID_ROOT, validate_derived_root
from preprocessing.common import configure_external_model_caches


def test_derived_outputs_cannot_be_inside_droid_source(tmp_path) -> None:
    source = tmp_path / "droid"
    source.mkdir()
    with pytest.raises(ValueError):
        validate_derived_root(source / "cache", source)
    with pytest.raises(ValueError):
        validate_derived_root(tmp_path, source)
    assert (
        validate_derived_root(tmp_path / "derived", source)
        == (tmp_path / "derived").resolve()
    )


def test_default_read_only_source_is_the_actual_droid_mount() -> None:
    assert str(DEFAULT_DROID_ROOT) == "/home/ws/data/droid"
    with pytest.raises(ValueError):
        validate_derived_root("/home/ws/data/droid/manifests")


def test_foundation_model_caches_are_forced_under_external_derived_root(
    tmp_path, monkeypatch
) -> None:
    names = ("TORCH_HOME", "HF_HOME", "XDG_CACHE_HOME")
    previous = {name: os.environ.get(name) for name in names}
    source = tmp_path / "droid"
    source.mkdir()
    locations = configure_external_model_caches(tmp_path / "derived", source)
    assert all(str(tmp_path / "derived") in value for value in locations.values())
    with pytest.raises(ValueError):
        configure_external_model_caches(source / "cache", source)
    for name, value in previous.items():
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
