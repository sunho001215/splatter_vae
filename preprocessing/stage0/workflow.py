from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dataset.droid.codecs import (
    JPEGCodecConfig,
    NumericCodecConfig,
    decode_numeric_array,
    depth_meters_to_u16,
    depth_u16_to_meters,
    encode_jpeg,
    encode_numeric_array,
    flow_pixels_to_i16,
)
from dataset.droid.preprocessed_manifest import (
    DEFAULT_LAGER_SCENE_CENTER,
    RETAINED_RAW_STRIDE,
    load_stage0_manifest,
    sample_key,
    shard_path,
)
from dataset.droid.records import (
    lager_record_contract,
    pack_lager_record,
    pack_real_rgb_record,
    pack_timestep_metadata,
)
from dataset.droid.rlds import TFDSRLDSBackend
from dataset.droid.safety import (
    assert_no_preprocessing_quality_hold,
    validate_derived_root,
)
from dataset.droid.shards import (
    IndexedTarReader,
    IndexedTarWriter,
    quarantine_incomplete_shard,
    shard_is_complete,
    validate_indexed_shard,
    write_json_atomic,
)
from preprocessing.lagernvs.camera import canonical_intrinsics
from preprocessing.lagernvs.pose import pose_sampler_contract

STAGES = ("rgb", "da3", "megaflow", "lagernvs", "compose")
STAGE_SUFFIXES = {
    "rgb": ("rgb",),
    "da3": ("depth",),
    "megaflow": ("flow",),
    "lagernvs": ("lager",),
    "compose": ("meta", "rgb", "depth", "flow", "lager"),
}


def deterministic_item_seed(
    schema_signature: str, episode_id: str, raw_timestep: int
) -> int:
    payload = f"{schema_signature}\0{episode_id}\0{int(raw_timestep)}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & 0x7FFF_FFFF


def _package_versions(names: Iterable[str]) -> dict[str, str | None]:
    output: dict[str, str | None] = {}
    for name in names:
        try:
            output[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            output[name] = None
    return output


def _resolved_package_versions() -> dict[str, str]:
    packages: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            packages[str(name).lower().replace("_", "-")] = distribution.version
    return dict(sorted(packages.items()))


def environment_metadata() -> dict[str, Any]:
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": _package_versions(
            (
                "torch",
                "torchvision",
                "numpy",
                "Pillow",
                "numcodecs",
                "huggingface-hub",
                "tensorflow",
                "tensorflow-datasets",
                "depth-anything-3",
                "megaflow",
            )
        ),
        "resolved_packages": _resolved_package_versions(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_cuda": torch.version.cuda,
        "cuda_device": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
    }


@dataclass(frozen=True)
class Stage0WorkerConfig:
    root: str
    stage: str
    worker_id: int = 0
    worker_count: int = 1
    droid_root: str = "/home/ws/data/droid"
    jpeg: JPEGCodecConfig = field(default_factory=JPEGCodecConfig)
    numeric: NumericCodecConfig = field(default_factory=NumericCodecConfig)
    model_cache: str | None = None
    da3_repository: str = "/home/ws/ws/droid_training/third_party/Depth-Anything-3"
    megaflow_repository: str = "/home/ws/ws/droid_training/third_party/MegaFlow"
    lagernvs_repository: str = "/home/ws/ws/droid_training/third_party/LagerNVS"
    scene_center: tuple[float, float, float] = DEFAULT_LAGER_SCENE_CENTER
    codec_audit_images: int = 32

    def __post_init__(self) -> None:
        if self.stage not in STAGES:
            raise ValueError(f"Unknown Stage-0 preprocessing stage {self.stage!r}.")
        if self.worker_count <= 0 or not 0 <= self.worker_id < self.worker_count:
            raise ValueError("Worker id/count are inconsistent.")
        # Workers are public entry points too.  Enforce the source-tree boundary
        # here rather than relying only on the top-level manifest/orchestrator.
        validate_derived_root(self.root, self.droid_root)
        if self.model_cache is not None:
            validate_derived_root(self.model_cache, self.droid_root)
        if self.codec_audit_images < 0:
            raise ValueError("codec_audit_images must be nonnegative.")

    @property
    def cache_root(self) -> Path:
        if self.model_cache is not None:
            return Path(self.model_cache).expanduser().resolve()
        return Path(self.root).expanduser().resolve() / "metadata" / "model_cache"


class _Progress:
    def __init__(self, config: Stage0WorkerConfig, shard_ids: list[int]) -> None:
        self.config = config
        self.root = Path(config.root).expanduser().resolve()
        self.path = (
            self.root
            / "progress"
            / config.stage
            / f"worker-{config.worker_id:02d}.json"
        )
        self.failed_path = (
            self.root
            / "progress"
            / config.stage
            / f"worker-{config.worker_id:02d}-failed.jsonl"
        )
        self.started = time.time()
        self.shard_ids = shard_ids
        self.completed: list[int] = []
        self.skipped: list[int] = []
        self.current: int | None = None
        self._write("starting")

    def _write(self, status: str) -> None:
        write_json_atomic(
            self.path,
            {
                "stage": self.config.stage,
                "worker_id": self.config.worker_id,
                "worker_count": self.config.worker_count,
                "pid": os.getpid(),
                "started_unix": self.started,
                "updated_unix": time.time(),
                "status": status,
                "assigned_shards": self.shard_ids,
                "current_shard": self.current,
                "completed_shards": self.completed,
                "skipped_verified_shards": self.skipped,
            },
        )

    def begin(self, shard_id: int) -> None:
        self.current = int(shard_id)
        self._write("running")

    def skip(self, shard_id: int) -> None:
        self.skipped.append(int(shard_id))
        self.current = None
        self._write("running")

    def finish(self, shard_id: int) -> None:
        self.completed.append(int(shard_id))
        self.current = None
        self._write("running")

    def fail(self, shard_id: int, error: BaseException) -> None:
        self.failed_path.parent.mkdir(parents=True, exist_ok=True)
        with self.failed_path.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "time_unix": time.time(),
                        "stage": self.config.stage,
                        "worker_id": self.config.worker_id,
                        "shard_id": int(shard_id),
                        "error_type": type(error).__name__,
                        "error": str(error),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
        self._write("failed")

    def done(self) -> None:
        self._write("complete")


class _CodecReferenceCapture:
    """Keep a deterministic bounded set of pre-JPEG pixels for Q95/Q97 audit."""

    def __init__(self, config: Stage0WorkerConfig) -> None:
        self.config = config
        self._values: list[tuple[int, str, np.ndarray]] = []

    def consider(self, identifier: str, image_hwc_u8: np.ndarray) -> None:
        maximum = int(self.config.codec_audit_images)
        if maximum == 0:
            return
        image = np.asarray(image_hwc_u8, dtype=np.uint8)
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError("Codec reference RGB must be HxWx3 uint8.")
        priority = int.from_bytes(
            hashlib.sha256(str(identifier).encode("utf-8")).digest()[:8], "little"
        )
        self._values.append((priority, str(identifier), np.ascontiguousarray(image)))
        self._values.sort(key=lambda item: (item[0], item[1]))
        if len(self._values) > maximum:
            self._values.pop()

    def write(self) -> str | None:
        if not self._values:
            return None
        path = (
            Path(self.config.root).expanduser().resolve()
            / "metadata"
            / "codec_references"
            / f"{self.config.stage}-worker-{self.config.worker_id:02d}.npz"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".partial")
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                identifiers=np.asarray(
                    [item[1] for item in self._values], dtype=np.str_
                ),
                images=np.stack([item[2] for item in self._values]),
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        return str(path)


def _episodes_for_shard(
    manifest: Mapping[str, Any], shard: Mapping[str, Any]
) -> list[dict[str, Any]]:
    return [
        dict(value)
        for value in manifest["episodes"][
            int(shard["episode_start"]) : int(shard["episode_stop"])
        ]
    ]


def _episode_geometry(
    entry: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cameras = entry["exterior_cameras"]
    K = np.asarray([camera["intrinsics_rlds"] for camera in cameras], dtype=np.float32)
    c2w = np.asarray([camera["c2w"] for camera in cameras], dtype=np.float32)
    w2c = np.asarray([camera["w2c"] for camera in cameras], dtype=np.float32)
    if K.shape != (2, 3, 3) or c2w.shape != (2, 4, 4) or w2c.shape != (2, 4, 4):
        raise ValueError(f"Episode {entry['episode_id']} has invalid camera geometry.")
    if not np.allclose(c2w @ w2c, np.eye(4), atol=2.0e-3):
        raise ValueError(f"Episode {entry['episode_id']} c2w/w2c are inconsistent.")
    return K, c2w, w2c


def _raw_retained_images(
    backend: TFDSRLDSBackend, entry: Mapping[str, Any]
) -> np.ndarray:
    episode = backend.get_episode(str(entry["rlds_split"]), int(entry["rlds_ordinal"]))
    images = np.asarray(episode["images"], dtype=np.uint8)
    expected = (int(entry["num_steps"]), 2, 180, 320, 3)
    if images.shape != expected:
        raise ValueError(
            f"Episode {entry['episode_id']} RGB shape {images.shape} != {expected}."
        )
    retained = np.ascontiguousarray(images[::RETAINED_RAW_STRIDE])
    if retained.shape[0] != int(entry["retained_count"]):
        raise RuntimeError(
            "Retained RGB count disagrees with the deterministic manifest."
        )
    return retained


def _global_key(entry: Mapping[str, Any], retained_index: int) -> str:
    return sample_key(int(entry["global_retained_start"]) + int(retained_index))


def _prepare_writer(
    config: Stage0WorkerConfig, manifest: Mapping[str, Any], shard_id: int
) -> IndexedTarWriter:
    output_stage = "final" if config.stage == "compose" else config.stage
    path = shard_path(config.root, output_stage, shard_id)
    if any(
        candidate.exists()
        for candidate in (
            path,
            path.with_suffix(".idx.json"),
            path.with_suffix(".complete.json"),
            path.with_suffix(".tar.partial"),
            path.with_suffix(".idx.json.partial"),
        )
    ):
        quarantine_incomplete_shard(
            path,
            recovery_root=Path(config.root)
            / "recovery"
            / config.stage
            / f"worker-{config.worker_id:02d}",
            schema_signature=str(manifest["schema_signature"]),
        )
    return IndexedTarWriter(
        path,
        stage=output_stage,
        shard_id=shard_id,
        schema_signature=str(manifest["schema_signature"]),
    )


def _process_rgb_shard(
    config: Stage0WorkerConfig,
    manifest: Mapping[str, Any],
    shard: Mapping[str, Any],
    backend: TFDSRLDSBackend,
    codec_capture: _CodecReferenceCapture,
) -> None:
    with _prepare_writer(config, manifest, int(shard["shard_id"])) as writer:
        for entry in _episodes_for_shard(manifest, shard):
            images = _raw_retained_images(backend, entry)
            for retained_index, synchronized in enumerate(images):
                for camera, image in enumerate(synchronized):
                    codec_capture.consider(
                        f"{entry['episode_id']}:{retained_index * RETAINED_RAW_STRIDE}:cam{camera}",
                        image,
                    )
                jpegs = tuple(encode_jpeg(image, config.jpeg) for image in synchronized)
                writer.add(
                    _global_key(entry, retained_index),
                    "rgb",
                    pack_real_rgb_record(jpegs),
                )


def _process_da3_shard(
    config: Stage0WorkerConfig,
    manifest: Mapping[str, Any],
    shard: Mapping[str, Any],
    backend: TFDSRLDSBackend,
    teacher: Any,
) -> None:
    with _prepare_writer(config, manifest, int(shard["shard_id"])) as writer:
        for entry in _episodes_for_shard(manifest, shard):
            images = _raw_retained_images(backend, entry)
            K, _c2w, w2c = _episode_geometry(entry)
            for retained_index, synchronized in enumerate(images):
                output = teacher(synchronized, K, w2c)
                depth = depth_meters_to_u16(output["metric_depth"])
                if depth.shape != (2, 180, 320):
                    raise RuntimeError(
                        "DA3 native depth encoding has an invalid shape."
                    )
                writer.add(
                    _global_key(entry, retained_index),
                    "depth",
                    encode_numeric_array(depth, config.numeric),
                )


def _process_megaflow_shard(
    config: Stage0WorkerConfig,
    manifest: Mapping[str, Any],
    shard: Mapping[str, Any],
    backend: TFDSRLDSBackend,
    teacher: Any,
) -> None:
    with _prepare_writer(config, manifest, int(shard["shard_id"])) as writer:
        for entry in _episodes_for_shard(manifest, shard):
            images = _raw_retained_images(backend, entry)
            flow = teacher.infer_retained_gap6(images)
            expected = (max(0, len(images) - 2), 2, 2, 180, 320)
            if flow.shape != expected:
                raise RuntimeError(
                    f"MegaFlow episode output {flow.shape} != {expected}."
                )
            for retained_index, value in enumerate(flow):
                writer.add(
                    _global_key(entry, retained_index),
                    "flow",
                    encode_numeric_array(flow_pixels_to_i16(value), config.numeric),
                )


def _process_lagernvs_shard(
    config: Stage0WorkerConfig,
    manifest: Mapping[str, Any],
    shard: Mapping[str, Any],
    backend: TFDSRLDSBackend,
    teacher: Any,
    pose_config: Any,
    codec_capture: _CodecReferenceCapture,
) -> None:
    depth_reader = IndexedTarReader(
        shard_path(config.root, "da3", int(shard["shard_id"]))
    )
    target_K = canonical_intrinsics(
        (1,), focal_px=float(teacher.canonical_focal_px), device=teacher.device
    )
    scene_center = torch.tensor(config.scene_center, device=teacher.device)
    try:
        with _prepare_writer(config, manifest, int(shard["shard_id"])) as writer:
            for entry in _episodes_for_shard(manifest, shard):
                images = _raw_retained_images(backend, entry)
                K_np, c2w_np, _w2c = _episode_geometry(entry)
                K = torch.from_numpy(K_np)[None].to(teacher.device)
                c2w = torch.from_numpy(c2w_np)[None].to(teacher.device)
                for retained_index, synchronized in enumerate(images):
                    key = _global_key(entry, retained_index)
                    depth_u16 = decode_numeric_array(
                        depth_reader.read(key, "depth"), config.numeric
                    )
                    depth_m, depth_valid = depth_u16_to_meters(depth_u16)
                    depth = torch.from_numpy(depth_m)[:, None][None].to(teacher.device)
                    validity = torch.from_numpy(depth_valid)[:, None][None].to(
                        teacher.device
                    )
                    from preprocessing.lagernvs.pose import sample_safe_target_poses

                    raw_timestep = retained_index * RETAINED_RAW_STRIDE
                    try:
                        poses = sample_safe_target_poses(
                            c2w,
                            K,
                            target_K,
                            depth,
                            validity,
                            scene_center,
                            pose_config,
                            seed=deterministic_item_seed(
                                str(manifest["schema_signature"]),
                                str(entry["episode_id"]),
                                raw_timestep,
                            ),
                        )
                    except RuntimeError as error:
                        raise RuntimeError(
                            "LagerNVS pose generation failed for "
                            f"episode={entry['episode_id']}, key={key}, "
                            f"retained_index={retained_index}, "
                            f"raw_timestep={raw_timestep}: {error}"
                        ) from error
                    source = torch.from_numpy(synchronized).permute(0, 3, 1, 2)[None]
                    rendered = teacher(source, K, c2w, poses["target_c2w"])
                    generated = rendered["generated_rgb"]
                    if generated.shape != (1, 4, 3, 256, 256):
                        raise RuntimeError(
                            f"LagerNVS returned {tuple(generated.shape)}."
                        )
                    generated_u8 = (
                        generated[0].mul(255).round().clamp(0, 255).byte().cpu().numpy()
                    )
                    for view, image in enumerate(generated_u8):
                        codec_capture.consider(
                            f"{entry['episode_id']}:{retained_index * RETAINED_RAW_STRIDE}:lager{view}",
                            np.moveaxis(image, 0, -1),
                        )
                    jpegs = tuple(
                        encode_jpeg(image, config.jpeg) for image in generated_u8
                    )
                    metadata = {
                        name: value[0].detach().cpu().numpy()
                        for name, value in poses.items()
                        if torch.is_tensor(value) and name != "support_mask"
                    }
                    writer.add(
                        key,
                        "lager",
                        pack_lager_record(
                            jpegs,
                            metadata,
                            poses["support_mask"][0].detach().cpu().numpy(),
                        ),
                    )
    finally:
        depth_reader.close()


def _process_compose_shard(
    config: Stage0WorkerConfig,
    manifest: Mapping[str, Any],
    shard: Mapping[str, Any],
) -> None:
    shard_id = int(shard["shard_id"])
    readers = {
        stage: IndexedTarReader(shard_path(config.root, stage, shard_id))
        for stage in ("rgb", "da3", "megaflow", "lagernvs")
    }
    try:
        with _prepare_writer(config, manifest, shard_id) as writer:
            for entry in _episodes_for_shard(manifest, shard):
                count = int(entry["retained_count"])
                for retained_index in range(count):
                    key = _global_key(entry, retained_index)
                    writer.add(
                        key,
                        "meta",
                        pack_timestep_metadata(
                            global_retained_index=int(entry["global_retained_start"])
                            + retained_index,
                            episode_index=int(entry["manifest_episode_index"]),
                            episode_id=str(entry["episode_id"]),
                            retained_index=retained_index,
                            raw_timestep=retained_index * RETAINED_RAW_STRIDE,
                        ),
                    )
                    writer.add(key, "rgb", readers["rgb"].read(key, "rgb", verify=True))
                    writer.add(
                        key, "depth", readers["da3"].read(key, "depth", verify=True)
                    )
                    if retained_index + 2 < count:
                        writer.add(
                            key,
                            "flow",
                            readers["megaflow"].read(key, "flow", verify=True),
                        )
                    writer.add(
                        key,
                        "lager",
                        readers["lagernvs"].read(key, "lager", verify=True),
                    )
    finally:
        for reader in readers.values():
            reader.close()


def _construct_teacher(config: Stage0WorkerConfig) -> tuple[Any | None, Any | None]:
    config.cache_root.mkdir(parents=True, exist_ok=True)
    if config.stage == "da3":
        from preprocessing.da3 import DA3DROIDTeacher

        teacher = DA3DROIDTeacher(
            config.da3_repository,
            cache_dir=config.cache_root,
            device="cuda:0",
        )
        return teacher, None
    if config.stage == "megaflow":
        from preprocessing.megaflow import MegaFlowDROIDTeacher

        teacher = MegaFlowDROIDTeacher(
            config.megaflow_repository,
            cache_dir=config.cache_root,
            device="cuda:0",
        )
        return teacher, None
    if config.stage == "lagernvs":
        from preprocessing.lagernvs.official import LagerNVSDROIDTeacher
        from preprocessing.lagernvs.pose import LagerTargetPoseConfig

        teacher = LagerNVSDROIDTeacher(
            config.lagernvs_repository,
            cache_dir=config.cache_root,
            device="cuda:0",
            dtype=torch.bfloat16,
            microbatch_size=1,
        )
        return teacher, LagerTargetPoseConfig()
    return None, None


def _stage_provenance(
    config: Stage0WorkerConfig,
    teacher: Any | None,
    pose_config: Any | None = None,
) -> None:
    root = Path(config.root).expanduser().resolve()
    payload = {
        "stage": config.stage,
        "configuration": {
            **asdict(config),
            "jpeg": config.jpeg.metadata(),
            "numeric": config.numeric.metadata(),
        },
        "environment": environment_metadata(),
        "teacher": teacher.metadata() if teacher is not None else None,
        "pose_configuration": (
            asdict(pose_config) if pose_config is not None else None
        ),
        "pose_sampler_contract": (
            pose_sampler_contract(pose_config) if pose_config is not None else None
        ),
        "lager_record_contract": (
            lager_record_contract() if config.stage == "lagernvs" else None
        ),
    }
    path = (
        root
        / "metadata"
        / "stages"
        / f"{config.stage}-worker-{config.worker_id:02d}.json"
    )
    write_json_atomic(path, payload)


def run_stage_worker(config: Stage0WorkerConfig) -> dict[str, Any]:
    assert_no_preprocessing_quality_hold(config.root)
    manifest = load_stage0_manifest(config.root)
    expected_encoding = manifest["encoding"]
    configured_encoding = {
        "quality": int(config.jpeg.quality),
        "chroma_subsampling": str(config.jpeg.subsampling),
        "compression_level": int(config.numeric.compression_level),
    }
    manifest_encoding = {
        "quality": int(expected_encoding["real_rgb"]["quality"]),
        "chroma_subsampling": str(expected_encoding["real_rgb"]["chroma_subsampling"]),
        "compression_level": int(
            expected_encoding["numeric_compression"]["compression_level"]
        ),
    }
    if configured_encoding != manifest_encoding:
        raise ValueError(
            "Worker codecs differ from the signed dataset manifest: "
            f"configured={configured_encoding}, manifest={manifest_encoding}."
        )
    expected_scene_center = tuple(
        float(value)
        for value in manifest["teacher_processing"]["lagernvs"][
            "configured_scene_center"
        ]
    )
    if tuple(float(value) for value in config.scene_center) != expected_scene_center:
        raise ValueError(
            "Worker Lager scene center differs from the signed dataset manifest: "
            f"configured={config.scene_center}, manifest={expected_scene_center}."
        )
    shards = [
        shard
        for shard in manifest["shards"]
        if int(shard["shard_id"]) % int(config.worker_count) == int(config.worker_id)
    ]
    progress = _Progress(config, [int(shard["shard_id"]) for shard in shards])
    codec_capture = _CodecReferenceCapture(config)
    teacher, pose_config = _construct_teacher(config)
    _stage_provenance(config, teacher, pose_config)
    backend = (
        TFDSRLDSBackend(config.droid_root, cache_size=1)
        if config.stage in {"rgb", "da3", "megaflow", "lagernvs"}
        else None
    )
    for shard in shards:
        shard_id = int(shard["shard_id"])
        output_stage = "final" if config.stage == "compose" else config.stage
        output_path = shard_path(config.root, output_stage, shard_id)
        if shard_is_complete(
            output_path,
            schema_signature=str(manifest["schema_signature"]),
            verify_checksums=True,
        ):
            progress.skip(shard_id)
            continue
        progress.begin(shard_id)
        try:
            if config.stage == "rgb":
                assert backend is not None
                _process_rgb_shard(config, manifest, shard, backend, codec_capture)
            elif config.stage == "da3":
                assert backend is not None and teacher is not None
                _process_da3_shard(config, manifest, shard, backend, teacher)
            elif config.stage == "megaflow":
                assert backend is not None and teacher is not None
                _process_megaflow_shard(config, manifest, shard, backend, teacher)
            elif config.stage == "lagernvs":
                assert (
                    backend is not None
                    and teacher is not None
                    and pose_config is not None
                )
                _process_lagernvs_shard(
                    config,
                    manifest,
                    shard,
                    backend,
                    teacher,
                    pose_config,
                    codec_capture,
                )
            elif config.stage == "compose":
                _process_compose_shard(config, manifest, shard)
            validate_indexed_shard(output_path, deep=False)
            progress.finish(shard_id)
        except BaseException as error:
            progress.fail(shard_id, error)
            raise
    progress.done()
    codec_reference = codec_capture.write()
    return {
        "stage": config.stage,
        "worker_id": config.worker_id,
        "assigned": len(shards),
        "completed": len(progress.completed),
        "skipped": len(progress.skipped),
        "codec_reference": codec_reference,
    }
