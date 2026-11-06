"""Real-world SNeRL validation (s4d/rwlearn/snerl_eval.py): 3 input cameras; separate validation episodes; input-view
and leave-one-camera-out novel-view reconstruction reported separately. CPU, tiny official NeRF networks, a random
(unpretrained) DINO teacher; 128 px synthetic store with the rig's three cameras."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from s4d.rwlearn import baseline_data as BD
from s4d.rwlearn import configs as C
from s4d.rwlearn import snerl_eval as SE
from tests.test_rwlearn_store_baselines import CAMS, synthetic_store

TINY_NERF = {"netdepth": 2, "netwidth": 32, "netdepth_fine": 2, "netwidth_fine": 32, "N_samples": 8, "N_importance": 8}


def test_protocol_views_for_a_three_camera_rig():
    views = SE.protocol_views(3, 1)
    assert views["input_view"] == ([0, 1, 2], 1)  # all 3 cameras are inputs, render one of them
    assert views["novel_view_loco"] == ([0, 2], 1)  # the rendered camera is held out of the encoder inputs
    with pytest.raises(ValueError):
        SE.protocol_views(1, 0)


def test_realworld_validation_reports_input_and_novel_view_separately(tmp_path, monkeypatch):
    import yaml

    from s4d.baselines.snerl import model as M
    from s4d.baselines.snerl import upstream as U

    store = synthetic_store(episodes=2, frames=6, hw=(128, 128), pause_at=3, drop=None)
    val = BD.RealworldSNeRLScenes(store, store.episodes(), seed=1, num_inputs=3, lookat=C.SNERL_LOOKAT,
                                  scale=C.SNERL_POSE_SCALE)
    assert val.num_cams == 3 and val.num_inputs == 3  # user decision: 3 input cameras
    nerf = yaml.safe_load((C.REPO / "configs/realworld/baselines/snerl/drawer-open.yaml").read_text())["nerf"]
    nerf.update(TINY_NERF)
    torch.manual_seed(0)
    model = M.create(nerf, "cpu")
    with torch.no_grad():  # no empty ray (finite rendered depth), as tests/test_baselines_snerl.py::tiny_model(live)
        for name in ("network_fn", "network_fine"):
            model.render_kwargs_train[name].alpha_linear.bias.add_(1.0)
    extractor = U.dino_extractor("cpu", pretrained=False)
    seen = []
    orig = val.scene

    def spy(ep, t, inputs, target, with_depth=False):
        seen.append((list(inputs), int(target)))
        return orig(ep, t, inputs, target, with_depth)

    monkeypatch.setattr(val, "scene", spy)
    out = SE.realworld_validate(model, extractor, val, 2, 4096, tmp_path / "eval.png")
    assert set(out) == {f"val/{p}/{m}" for p in SE.PROTOCOLS for m in SE.METRICS}
    assert all(np.isfinite(v) for v in out.values())
    assert (tmp_path / "eval.png").is_file()
    for k in range(2):
        _, _, _, target = val.draw(k)
        assert seen[2 * k] == ([0, 1, 2], target) and seen[2 * k + 1] == ([c for c in range(3) if c != target], target)
    assert model.render_kwargs_train["network_fn"].training  # training mode restored
    assert len(CAMS) == 3
