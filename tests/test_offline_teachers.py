from __future__ import annotations

import sys
import types

import numpy as np
import torch

from preprocessing.da3.official import two_view_baseline_alignment
from preprocessing.lagernvs.official import _install_xformers_sdpa_fallback


def _w2c_with_second_center(x: float) -> np.ndarray:
    c2w = np.broadcast_to(np.eye(4), (2, 4, 4)).copy()
    c2w[1, 0, 3] = float(x)
    return np.linalg.inv(c2w)


def test_da3_two_view_alignment_uses_identifiable_baseline_scale() -> None:
    predicted = _w2c_with_second_center(2.0)
    supplied = _w2c_with_second_center(0.8)
    rotation, translation, scale, aligned = two_view_baseline_alignment(
        predicted, supplied, return_aligned=True
    )
    np.testing.assert_allclose(rotation, np.eye(3))
    np.testing.assert_allclose(translation, 0.0)
    assert scale == 2.5
    np.testing.assert_allclose(aligned, supplied)


def test_da3_two_view_alignment_accepts_official_3x4_prediction() -> None:
    predicted = _w2c_with_second_center(1.2)[:, :3]
    supplied = _w2c_with_second_center(0.6)
    _rotation, _translation, scale = two_view_baseline_alignment(
        predicted, supplied
    )
    assert scale == 2.0


def test_lagernvs_forces_native_sdpa_even_when_xformers_imports(monkeypatch) -> None:
    installed = types.ModuleType("xformers")
    installed_ops = types.ModuleType("xformers.ops")
    installed.ops = installed_ops
    monkeypatch.setitem(sys.modules, "xformers", installed)
    monkeypatch.setitem(sys.modules, "xformers.ops", installed_ops)

    _install_xformers_sdpa_fallback()
    fallback = sys.modules["xformers.ops"].memory_efficient_attention
    q = torch.randn(2, 5, 3, 8)
    actual = fallback(q, q, q)
    expected = torch.nn.functional.scaled_dot_product_attention(
        q.permute(0, 2, 1, 3),
        q.permute(0, 2, 1, 3),
        q.permute(0, 2, 1, 3),
    ).permute(0, 2, 1, 3)
    torch.testing.assert_close(actual, expected)
