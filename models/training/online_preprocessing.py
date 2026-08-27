from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import torch

from dataset.droid.dataset import IMAGENET_MEAN, IMAGENET_STD, normalize_encoder_rgb
from dataset.droid.sampling import (
    MotionCropConfig,
    build_motion_maps,
    select_motion_crop_from_maps,
)
from dataset.droid.transforms import (
    IMAGENET_NEUTRAL_RGB,
    image_validity_mask,
    transform_confidence,
    transform_depth,
    transform_flow,
    transform_intrinsics,
    transform_rgb,
    transform_validity,
)
from preprocessing.lagernvs.camera import canonical_intrinsics
from preprocessing.lagernvs.official import LagerNVSDROIDTeacher
from preprocessing.lagernvs.pose import LagerTargetPoseConfig, sample_safe_target_poses
from preprocessing.memfof import MEMFOFDROIDTeacher
from preprocessing.xlens.official import XLensDROIDTeacher


@dataclass(frozen=True)
class OnlinePreprocessingConfig:
    motion_crop: MotionCropConfig
    normalize_mean: tuple[float, float, float] = IMAGENET_MEAN
    normalize_std: tuple[float, float, float] = IMAGENET_STD
    rgb_padding_value: tuple[float, float, float] = IMAGENET_NEUTRAL_RGB


class OnlineTeacherPipeline:
    """Run frozen teachers once, then build the aligned 224x224 training batch."""

    def __init__(
        self,
        memfof: MEMFOFDROIDTeacher,
        xlens: XLensDROIDTeacher,
        config: OnlinePreprocessingConfig,
        *,
        lagernvs: LagerNVSDROIDTeacher | None = None,
        novel_pose_config: LagerTargetPoseConfig | None = None,
        scene_center: tuple[float, float, float] | None = None,
    ) -> None:
        self.memfof = memfof
        self.xlens = xlens
        self.config = config
        self.lagernvs = lagernvs
        self.novel_pose_config = novel_pose_config
        self.scene_center = scene_center

    @torch.inference_mode()
    def infer_memfof(self, raw_batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        return self.memfof(raw_batch["raw_histories"])

    @torch.inference_mode()
    def infer_xlens(self, raw_batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        return self.xlens(
            raw_batch["raw_histories"],
            raw_batch["raw_K"],
            raw_batch["raw_c2w"],
        )

    @torch.no_grad()
    def select_motion_crops(
        self,
        raw_batch: dict[str, Any],
        memfof_output: dict[str, torch.Tensor],
        *,
        motion_maps: list[list[tuple[torch.Tensor, torch.Tensor]]] | None = None,
    ) -> list[list[Any]]:
        """Select one shared temporal crop per logical sample and camera."""

        histories = raw_batch["raw_histories"]
        if histories.dim() != 6 or histories.shape[1:4] != (2, 3, 3):
            raise ValueError(
                "Online DROID preprocessing expects (B,2,3,3,180,320) histories."
            )
        crop_sizes = raw_batch["sampled_crop_size"].long()
        if crop_sizes.shape != (histories.shape[0],):
            raise ValueError("Each logical DROID sample must provide one crop size.")
        motion_maps = motion_maps or self.compute_motion_maps(
            raw_batch, memfof_output
        )
        return [
            [
                select_motion_crop_from_maps(
                    motion_maps[batch_index][camera_index][0],
                    motion_maps[batch_index][camera_index][1],
                    int(crop_sizes[batch_index].item()),
                    self.config.motion_crop,
                )
                for camera_index in range(histories.shape[1])
            ]
            for batch_index in range(histories.shape[0])
        ]

    @torch.no_grad()
    def compute_motion_maps(
        self,
        raw_batch: dict[str, Any],
        memfof_output: dict[str, torch.Tensor],
    ) -> list[list[tuple[torch.Tensor, torch.Tensor]]]:
        """Aggregate and smooth native MEMFOF flow for each sample/camera."""

        histories = raw_batch["raw_histories"]
        if histories.dim() != 6 or histories.shape[1:4] != (2, 3, 3):
            raise ValueError(
                "Online DROID preprocessing expects (B,2,3,3,180,320) histories."
            )
        return [
            [
                build_motion_maps(
                    memfof_output["flow"][batch_index, camera_index],
                    memfof_output["validity"][batch_index, camera_index],
                    self.config.motion_crop,
                )
                for camera_index in range(histories.shape[1])
            ]
            for batch_index in range(histories.shape[0])
        ]

    @torch.no_grad()
    def prepare_real_batch(
        self,
        raw_batch: dict[str, Any],
        *,
        memfof_output: dict[str, torch.Tensor] | None = None,
        xlens_output: dict[str, torch.Tensor] | None = None,
        selections: list[list[Any]] | None = None,
    ) -> dict[str, Any]:
        memfof_output = memfof_output or self.infer_memfof(raw_batch)
        xlens_output = xlens_output or self.infer_xlens(raw_batch)
        histories = raw_batch["raw_histories"]
        if histories.dim() != 6 or histories.shape[1:4] != (2, 3, 3):
            raise ValueError(
                "Online DROID preprocessing expects (B,2,3,3,180,320) histories."
            )
        batch_size, camera_count, timesteps = histories.shape[:3]
        crop_sizes = raw_batch["sampled_crop_size"].long()
        if crop_sizes.shape != (batch_size,):
            raise ValueError("Each logical DROID sample must provide one crop size.")

        transformed_rgb: list[torch.Tensor] = []
        transformed_validity: list[torch.Tensor] = []
        transformed_depth: list[torch.Tensor] = []
        transformed_depth_confidence: list[torch.Tensor] = []
        transformed_depth_validity: list[torch.Tensor] = []
        transformed_flow: list[torch.Tensor] = []
        transformed_flow_confidence: list[torch.Tensor] = []
        transformed_flow_validity: list[torch.Tensor] = []
        transformed_K: list[torch.Tensor] = []
        selections = selections or self.select_motion_crops(raw_batch, memfof_output)

        for batch_index in range(batch_size):
            rgb_by_camera = []
            validity_by_camera = []
            depth_by_camera = []
            depth_confidence_by_camera = []
            depth_validity_by_camera = []
            flow_by_camera = []
            flow_confidence_by_camera = []
            flow_validity_by_camera = []
            K_by_camera = []
            for camera_index in range(camera_count):
                selection = selections[batch_index][camera_index]
                transform = selection.transform
                camera_rgb = transform_rgb(
                    histories[batch_index, camera_index],
                    transform,
                    padding_value=self.config.rgb_padding_value,
                )
                image_validity = image_validity_mask(
                    transform,
                    leading_shape=(timesteps,),
                    device=histories.device,
                )
                depth = transform_depth(
                    xlens_output["metric_depth"][batch_index, :, camera_index],
                    transform,
                )
                depth_confidence = transform_confidence(
                    xlens_output["confidence"][batch_index, :, camera_index],
                    transform,
                )
                depth_validity = transform_validity(
                    xlens_output["validity"][batch_index, :, camera_index],
                    transform,
                ) & image_validity
                flow = transform_flow(
                    memfof_output["flow"][batch_index, camera_index], transform
                )
                flow_confidence = transform_confidence(
                    memfof_output["confidence"][batch_index, camera_index],
                    transform,
                )
                flow_validity = transform_validity(
                    memfof_output["validity"][batch_index, camera_index],
                    transform,
                ) & image_validity[1:2].expand(2, -1, -1, -1)
                rgb_by_camera.append(camera_rgb)
                validity_by_camera.append(image_validity)
                depth_by_camera.append(depth)
                depth_confidence_by_camera.append(depth_confidence)
                depth_validity_by_camera.append(depth_validity)
                flow_by_camera.append(flow)
                flow_confidence_by_camera.append(flow_confidence)
                flow_validity_by_camera.append(flow_validity)
                K_by_camera.append(
                    transform_intrinsics(
                        raw_batch["raw_K"][batch_index, camera_index], transform
                    )
                )
            transformed_rgb.append(torch.stack(rgb_by_camera))
            transformed_validity.append(torch.stack(validity_by_camera))
            transformed_depth.append(torch.stack(depth_by_camera))
            transformed_depth_confidence.append(
                torch.stack(depth_confidence_by_camera)
            )
            transformed_depth_validity.append(torch.stack(depth_validity_by_camera))
            transformed_flow.append(torch.stack(flow_by_camera))
            transformed_flow_confidence.append(torch.stack(flow_confidence_by_camera))
            transformed_flow_validity.append(torch.stack(flow_validity_by_camera))
            transformed_K.append(torch.stack(K_by_camera))

        # RGB/validity remain camera-major for the encoder, then are permuted to
        # time-major for rendering against the same transformed camera geometry.
        rgb = torch.stack(transformed_rgb)
        image_validity = torch.stack(transformed_validity)
        depth = torch.stack(transformed_depth)
        depth_confidence = torch.stack(transformed_depth_confidence)
        depth_validity = torch.stack(transformed_depth_validity)
        flow = torch.stack(transformed_flow)
        flow_confidence = torch.stack(transformed_flow_confidence)
        flow_validity = torch.stack(transformed_flow_validity)
        camera_K = torch.stack(transformed_K)

        metadata_fields = tuple(asdict(selections[0][0].metadata))
        crop_metadata: dict[str, torch.Tensor] = {}
        integer_fields = {
            "crop_size",
            "crop_x0",
            "crop_y0",
            "crop_center_x",
            "crop_center_y",
        }
        for field_name in metadata_fields:
            values = [
                [asdict(selection.metadata)[field_name] for selection in per_camera]
                for per_camera in selections
            ]
            dtype = (
                torch.bool
                if field_name == "low_motion_fallback_used"
                else torch.long
                if field_name in integer_fields
                else torch.float32
            )
            crop_metadata[field_name] = torch.tensor(
                values, dtype=dtype, device=histories.device
            )

        time_major_K = camera_K[:, None].expand(-1, timesteps, -1, -1, -1)
        c2w = raw_batch["raw_c2w"].float()
        w2c = raw_batch["raw_w2c"].float()
        output = dict(raw_batch)
        output.update(
            {
                "representation_histories": normalize_encoder_rgb(
                    rgb,
                    self.config.normalize_mean,
                    self.config.normalize_std,
                ),
                "representation_flows": flow,
                "representation_validity": image_validity,
                "representation_K": camera_K[:, :, None]
                .expand(-1, -1, timesteps, -1, -1)
                .contiguous(),
                "target_rgb": rgb.permute(0, 2, 1, 3, 4, 5).float() / 255.0,
                "target_image_validity": image_validity.permute(
                    0, 2, 1, 3, 4, 5
                ).contiguous(),
                "target_depth": depth.permute(0, 2, 1, 3, 4, 5).contiguous(),
                "target_depth_confidence": depth_confidence.permute(
                    0, 2, 1, 3, 4, 5
                ).contiguous(),
                "target_depth_validity": depth_validity.permute(
                    0, 2, 1, 3, 4, 5
                ).contiguous(),
                "target_flow": flow.permute(0, 2, 1, 3, 4, 5).contiguous(),
                "target_flow_confidence": flow_confidence.permute(
                    0, 2, 1, 3, 4, 5
                ).contiguous(),
                "target_flow_validity": flow_validity.permute(
                    0, 2, 1, 3, 4, 5
                ).contiguous(),
                "target_K": time_major_K.contiguous(),
                "target_c2w": c2w[:, None]
                .expand(-1, timesteps, -1, -1, -1)
                .contiguous(),
                "target_w2c": w2c[:, None]
                .expand(-1, timesteps, -1, -1, -1)
                .contiguous(),
                "crop_metadata": crop_metadata,
                "native_memfof_flow": memfof_output["flow"],
                "native_memfof_confidence": memfof_output["confidence"],
                "native_memfof_validity": memfof_output["validity"],
                "native_xlens_depth": xlens_output["metric_depth"],
                "native_xlens_confidence": xlens_output["confidence"],
                "native_xlens_validity": xlens_output["validity"],
                "aggregate_motion": torch.stack(
                    [
                        torch.stack(
                            [selection.aggregate_motion_map for selection in per_camera]
                        )
                        for per_camera in selections
                    ]
                ),
                "smoothed_motion": torch.stack(
                    [
                        torch.stack(
                            [selection.smoothed_motion_map for selection in per_camera]
                        )
                        for per_camera in selections
                    ]
                ),
            }
        )
        return output

    @torch.no_grad()
    def sample_novel_target(
        self,
        batch: dict[str, Any],
        *,
        seed: int,
        stage_runner: Callable[[str, Callable[[], Any]], Any] | None = None,
    ) -> dict[str, Any]:
        if self.novel_pose_config is None or self.scene_center is None:
            raise RuntimeError("LagerNVS target-pose configuration is incomplete.")
        logical_batch = batch["raw_histories"].shape[0]
        focal_px = (
            self.lagernvs.canonical_focal_px
            if self.lagernvs is not None
            else 186.5
        )
        target_K = canonical_intrinsics(
            (logical_batch,),
            focal_px=focal_px,
            device=batch["raw_histories"].device,
        )
        return sample_safe_target_poses(
            batch["raw_c2w"],
            batch["raw_K"],
            target_K,
            batch["native_xlens_depth"][:, 2],
            batch["native_xlens_confidence"][:, 2],
            batch["native_xlens_validity"][:, 2],
            batch["raw_histories"].new_tensor(
                self.scene_center, dtype=torch.float32
            ),
            self.novel_pose_config,
            seed=int(seed),
            stage_runner=stage_runner,
        )

    @torch.no_grad()
    def prepare_lagernvs_inputs(
        self,
        batch: dict[str, Any],
        target: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        if self.lagernvs is None:
            raise RuntimeError("novel_view.enabled=true requires the LagerNVS teacher.")
        return self.lagernvs.prepare_inputs(
            batch["raw_histories"][:, :, 2],
            batch["raw_K"],
            batch["raw_c2w"],
            target["target_c2w"],
        )

    @torch.no_grad()
    def merge_novel_view(
        self,
        batch: dict[str, Any],
        target: dict[str, Any],
        teacher: dict[str, torch.Tensor],
    ) -> dict[str, Any]:
        output = dict(batch)
        # Materialize normal (non-inference) tensors before they enter gsplat's
        # differentiable render path.  Neural teachers themselves still execute
        # under inference_mode in their adapters.
        output.update(
            {
                "novel_rgb": teacher["generated_rgb"].clone(),
                "novel_K": teacher["canonical_K"].clone(),
                "novel_c2w": target["target_c2w"].clone(),
                "novel_w2c": target["target_w2c"].clone(),
                "novel_support_mask": target["support_mask"].clone(),
                "novel_source_rgb": teacher["canonical_source_rgb"].clone(),
                "novel_source_validity": teacher[
                    "canonical_source_validity"
                ].clone(),
                "novel_pose_metadata": {
                    key: value
                    for key, value in target.items()
                    if key
                    not in {
                        "target_c2w",
                        "target_w2c",
                        "support_mask",
                    }
                },
                "novel_base_c2w": target["base_c2w"].clone(),
            }
        )
        return output

    @torch.no_grad()
    def add_novel_view(
        self,
        batch: dict[str, Any],
        *,
        seed: int,
    ) -> dict[str, Any]:
        if self.lagernvs is None:
            raise RuntimeError("novel_view.enabled=true requires the LagerNVS teacher.")
        target = self.sample_novel_target(batch, seed=seed)
        prepared = self.prepare_lagernvs_inputs(batch, target)
        teacher = self.lagernvs.infer_prepared(prepared)
        return self.merge_novel_view(batch, target, teacher)

    def __call__(
        self,
        raw_batch: dict[str, Any],
        *,
        novel_enabled: bool = False,
        seed: int = 0,
    ) -> dict[str, Any]:
        memfof_output = self.infer_memfof(raw_batch)
        xlens_output = self.infer_xlens(raw_batch)
        output = self.prepare_real_batch(
            raw_batch, memfof_output=memfof_output, xlens_output=xlens_output
        )
        return self.add_novel_view(output, seed=seed) if novel_enabled else output
