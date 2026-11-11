"""Torch-free observation definitions shared by FQL training (``fql_data`` re-exports them unchanged) and deployment."""

from dataclasses import dataclass

FRAME_STACK = 3
FRAME_SPACING = 1  # transitions.history_indices: consecutive camera frames / synchronized steps


@dataclass(frozen=True)
class Condition:
    encoder_type: str | None  # None: no frozen encoder (end-to-end CNN trained jointly with FQL)
    views: int
    depth: bool
    cameras: bool
    label: str
    extra_inputs: str = "none"  # observation inputs beyond the single-view reference (1 RGB view + proprio)

    @property
    def multiview(self) -> bool:
        return self.views > 1

    @property
    def inputs(self) -> str:
        v = "3 synchronised RGB views" if self.multiview else "1 RGB view (random camera per sample)"
        return v + (" + depth" if self.depth else "") + (" + K, c2w" if self.cameras else "") + \
            " + 10-d proprio, 3-step history"


SINGLE_VIEW_REFERENCE = "1 RGB view (one random camera per sample, fixed for the history and next obs) + 10-d proprio"
CONDITIONS = {
    "dynamo": Condition("dynamo", 1, False, False, "DynaMo, single RGB"),
    "reviwo": Condition("reviwo", 1, False, False, "ReViWo, single RGB"),
    "sincro": Condition("sincro", 1, False, False, "SinCro, single RGB"),
    "snerl": Condition("snerl", 3, False, True, "SNeRL, native multi-view RGB + camera poses (privileged inputs)",
                       "+2 RGB views (all 3 synchronised cameras) and the 3 camera poses (calibrated extrinsics, "
                       "normalised c2w): SNeRL's native multi-view condition"),
    "mvp3d_multiview": Condition("mvp3d", 3, True, True, "3D-MVP A, multi-view RGB-D + extrinsics (privileged)",
                                 "+2 RGB views, depth of all 3, intrinsics + extrinsics (fused point cloud)"),
    "mvp3d_single": Condition("mvp3d", 1, True, True, "3D-MVP B, single RGB-D + extrinsics (privileged)",
                              "+depth of the view, its intrinsics + extrinsics"),
    "e2e_cnn": Condition(None, 1, False, False, "End-to-end IMPALA-small CNN, random init, trained jointly with FQL "
                                                "(official FQL pixel handling), single RGB"),
    # Ours (FQLNQ 2026-11-11): the real-world-pretrained Original Splatter4D export (s4d/rwlearn/ours.py ->
    # scripts/export_encoder.py), read exactly as in the simulation downstream protocol (s4d.rl.encoders
    # Splatter4DStateEncoder): ONE camera's 3-frame RGB history -> one joint state (a sequence feature, like SinCro's);
    # frame spacing 1 camera frame lies inside the pretraining strides 1/2/3 (realworld-ops/ours/DEVIATIONS.md D7).
    "ours": Condition("splatter4d", 1, False, False, "Ours (Splatter4D Original, real-world pretrained), single RGB"),
}
