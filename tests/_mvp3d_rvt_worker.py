"""Subprocess worker for tests/test_mvp3d.py: RVT's own ``MVTSingle`` (rvt/mvt/mvt_single.py at the pinned commit).

RVT's packages (``rvt``, ``mvt``, ``point_renderer``) go on ``sys.path`` only in this separate process. The worker
builds ``MVTSingle`` with the arguments of each requested case, loads our encoder's state dict into it
(``strict=False``: our encoder is MVTSingle without RVT's action head), runs it in eval mode on the given virtual
images and returns: the attention-trunk output (input of ``fc_aft_attn``), the per-view max-pool feature (input of
``feat_fc``, only for cases built with ``fc_aft_attn`` replaced by the identity) and ``pixel_loc``.

    python -I tests/_mvp3d_rvt_worker.py <cases.pt> <out.pt>
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from _bootstrap import guard_gpus  # noqa: E402

guard_gpus()


def official_mvt(case: dict):
    from rvt.mvt.mvt_single import MVT as MVTSingle

    cfg = case["cfg"]
    return MVTSingle(
        depth=cfg["depth"],
        img_size=cfg["img_size"],
        add_proprio=False,
        proprio_dim=4,
        add_lang=False,
        lang_dim=512,
        lang_len=77,
        img_feat_dim=3,
        feat_dim=(72 * 3) + 2 + 2,
        im_channels=cfg["im_channels"],
        attn_dim=cfg["attn_dim"],
        attn_heads=cfg["attn_heads"],
        attn_dim_head=cfg["attn_dim_head"],
        activation=cfg["activation"],
        weight_tie_layers=False,
        attn_dropout=cfg["attn_dropout"],
        decoder_dropout=0.0,
        img_patch_size=cfg["img_patch_size"],
        final_dim=cfg["im_channels"],  # RVT: trans_decoder input = final conv output (64 = im_channels)
        self_cross_ver=1,
        add_corr=True,
        norm_corr=False,
        add_pixel_loc=True,
        add_depth=True,
        rend_three_views=False,
        use_point_renderer=True,  # only selects which select_feat_from_hm is imported (PyTorch3D is unavailable)
        pe_fix=True,
        feat_ver=0,
        wpt_img_aug=0.01,
        inp_pre_pro=True,
        inp_pre_con=True,
        cvx_up=False,
        xops=False,
        rot_ver=0,
        num_rot=72,
        renderer=SimpleNamespace(num_img=cfg["num_img"]),
        no_feat=False,
    )


def main(cases_path: str, out_path: str) -> None:
    import torch

    from s4d.third_party import source_tree

    tree = source_tree("rvt")
    sys.path[:0] = [str(tree), str(tree / "rvt"), str(tree / "rvt" / "libs" / "point-renderer")]
    results = []
    for case in torch.load(cases_path, weights_only=False):
        torch.manual_seed(0)
        mvt = official_mvt(case)
        if case["identity_head"]:
            mvt.fc_aft_attn = torch.nn.Identity()
        missing, unexpected = mvt.load_state_dict(case["state_dict"], strict=False)
        mvt.eval()
        captured = {}

        def keep(name, store=captured):
            return lambda module, args: store.__setitem__(name, args[0].detach().clone())

        mvt.fc_aft_attn.register_forward_pre_hook(keep("trunk"))
        mvt.feat_fc.register_forward_pre_hook(keep("feat_fc_in"))
        with torch.no_grad():
            mvt(img=case["img"])
        results.append(
            {
                "trunk": captured["trunk"],
                "feat_fc_in": captured["feat_fc_in"],
                "pixel_loc": mvt.pixel_loc.clone(),
                "missing": sorted(missing),
                "unexpected": sorted(unexpected),
                "attn_file": sys.modules["rvt.mvt.attn"].__file__,
            }
        )
    torch.save(results, out_path)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
