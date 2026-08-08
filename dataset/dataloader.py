import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, get_worker_info

from utils.general_utils import invert_4x4

DatasetPathInput = Union[str, Sequence[str]]
DemoRef = Tuple[int, str]
SampleRef = Tuple[int, str, int, Tuple[int, ...]]
TEMPORAL_WINDOW = 3
TRAIN_TEMPORAL_STRIDES = (3, 6, 9)
FLOW_TEMPORAL_GAPS = (3, 6, 9, 12, 18)
OPTICAL_FLOW_GROUP = "optical_flow"
SegSelector = Tuple[Optional[int], int]
SegIdValue = Union[int, str, Sequence[Union[int, str]]]
SegIdSpec = Optional[Union[int, str, Sequence[Union[int, str]], Dict[str, SegIdValue]]]


def _normalize_dataset_paths(dataset_path: DatasetPathInput) -> List[str]:
    if isinstance(dataset_path, (str, bytes)):
        paths = [str(dataset_path)]
    else:
        paths = [str(path) for path in dataset_path]
    if not paths:
        raise ValueError("At least one HDF5 dataset path is required.")
    return paths


def _demo_label(file_idx: int, demo_key: str, num_files: int) -> str:
    return demo_key if num_files == 1 else f"file{file_idx}:{demo_key}"


SEG_TYPE_ALIASES = {
    "background": -1,
    "body": 1,
    "joint": 3,
    "geom": 5,
    "site": 6,
    "camera": 7,
    "light": 8,
}
SEG_TYPE_NAMES = {value: key for key, value in SEG_TYPE_ALIASES.items()}


def _parse_seg_selector(value: Union[int, str]) -> SegSelector:
    if isinstance(value, (int, np.integer)):
        return (None, int(value))
    text = str(value).strip()
    if not text:
        raise ValueError("Empty segmentation selector is not allowed.")
    if ":" not in text:
        return (None, int(text))
    type_text, id_text = [part.strip() for part in text.split(":", 1)]
    if not type_text or not id_text:
        raise ValueError(f"Invalid segmentation selector {value!r}; expected 'geom:43' or '5:43'.")
    obj_type = SEG_TYPE_ALIASES.get(type_text.lower(), None)
    if obj_type is None:
        obj_type = int(type_text)
    return (int(obj_type), int(id_text))


def _coerce_seg_ids(value: SegIdValue | None) -> Optional[List[SegSelector]]:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        return [_parse_seg_selector(part.strip()) for part in text.split(",") if part.strip()]
    if isinstance(value, (int, np.integer)):
        return [_parse_seg_selector(value)]
    return [_parse_seg_selector(item) for item in value]


def _normalize_seg_id_spec(spec: SegIdSpec):
    if spec is None:
        return None
    if isinstance(spec, dict):
        return {str(key): _coerce_seg_ids(value) for key, value in spec.items()}
    return _coerce_seg_ids(spec)


def _format_seg_selector(selector: SegSelector) -> str:
    obj_type, obj_id = selector
    if obj_type is None:
        return str(int(obj_id))
    return f"{SEG_TYPE_NAMES.get(int(obj_type), str(int(obj_type)))}:{int(obj_id)}"


def _format_seg_selectors(selectors: Sequence[SegSelector]) -> str:
    return "[" + ", ".join(_format_seg_selector(selector) for selector in selectors) + "]"


def _segmentation_mask_from_selectors(
    seg_ids: np.ndarray,
    seg_types: Optional[np.ndarray],
    selectors: Sequence[SegSelector],
) -> np.ndarray:
    mask = np.zeros(seg_ids.shape, dtype=bool)
    for obj_type, obj_id in selectors:
        if obj_type is None:
            mask |= seg_ids == int(obj_id)
        else:
            if seg_types is None:
                raise ValueError(
                    f"Typed segmentation selector {_format_seg_selector((obj_type, obj_id))!r} requires *_seg_type datasets. "
                    "Collect with segmentation.save_objtype=true or use plain integer IDs."
                )
            mask |= (seg_ids == int(obj_id)) & (seg_types == int(obj_type))
    return mask


class metaworldMultiViewTemporalHDF5Dataset(Dataset):
    """metaworld/Meta-World HDF5 dataset returning every selected camera.

    ``dataset_path`` may be either one HDF5 file or a list of HDF5 files.  When
    multiple files are supplied, the dataset builds one global sample list over
    all ``(file, demo, timestep)`` entries, so a shuffled DataLoader naturally
    samples across environments during training.

    Returned tensors:
        images: (3, N_cam, 3, H, W) uint8 in [0, 255]
        K:      (N_cam, 3, 3) camera intrinsics for static cameras
        c2w:    (N_cam, 4, 4) OpenCV camera-to-world transforms
        w2c:    (N_cam, 4, 4) OpenCV world-to-camera transforms
        depths: (3, N_cam, 1, H, W) float32 metric camera-z depth
        masks:  (3, N_cam, 1, H, W) bool selected segmentation mask
        optical_flows: (3, N_cam, 2, H, W) float16 pixel displacement
    """

    def __init__(
        self,
        dataset_path: DatasetPathInput,
        demo_keys: List[Union[str, DemoRef]],
        views: List[str],
        max_frames_per_demo: Optional[int] = None,
        seed: int = 0,
        temporal_strides: Sequence[int] = TRAIN_TEMPORAL_STRIDES,
        fixed_temporal_stride: Optional[int] = None,
        selected_seg_ids: SegIdSpec = None,
    ):
        super().__init__()
        self.dataset_paths = _normalize_dataset_paths(dataset_path)
        self.dataset_path = self.dataset_paths[0]
        self.demo_refs: List[DemoRef] = [
            (int(item[0]), str(item[1])) if isinstance(item, tuple) else (0, str(item))
            for item in demo_keys
        ]
        self.max_frames_per_demo = max_frames_per_demo
        self.temporal_strides = tuple(sorted({int(stride) for stride in temporal_strides}))
        if self.temporal_strides != TRAIN_TEMPORAL_STRIDES:
            raise ValueError(
                f"Training temporal strides are fixed to {list(TRAIN_TEMPORAL_STRIDES)}, "
                f"got {list(self.temporal_strides)}."
            )
        self.fixed_temporal_stride = (
            None if fixed_temporal_stride is None else max(1, int(fixed_temporal_stride))
        )
        if (
            self.fixed_temporal_stride is not None
            and self.fixed_temporal_stride not in self.temporal_strides
        ):
            raise ValueError(
                f"fixed_temporal_stride={self.fixed_temporal_stride} must be included in "
                f"temporal_strides={self.temporal_strides}."
            )
        self.selected_seg_ids = _normalize_seg_id_spec(selected_seg_ids)
        self.rng = random.Random(seed)
        self.views = list(views)
        if not self.views:
            raise ValueError("The fixed dataset contract requires an explicit non-empty views list.")
        self.num_views = len(self.views)

        self._h5_handles: Dict[int, h5py.File] = {}
        self.demo_lengths: Dict[DemoRef, int] = {}
        self.demo_mask_selectors: Dict[DemoRef, List[SegSelector]] = {}
        self.cam_cache: Dict[DemoRef, Dict[str, Dict[str, np.ndarray]]] = {}
        self.cam_tensor_cache: Dict[DemoRef, Dict[str, torch.Tensor]] = {}
        self.samples: List[SampleRef] = []

        self._index_files_and_build_samples()


    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5_handles"] = {}
        return state

    def __del__(self):
        try:
            for handle in self._h5_handles.values():
                handle.close()
        except Exception:
            pass

    def _get_h5(self, file_idx: int) -> h5py.File:
        if file_idx not in self._h5_handles:
            self._h5_handles[file_idx] = h5py.File(self.dataset_paths[file_idx], "r")
        return self._h5_handles[file_idx]

    def _selected_ids_for_demo(self, dataset_path: str, demo_key: str, env_name: str) -> List[SegSelector]:
        if self.selected_seg_ids is None:
            raise ValueError(
                "The fixed dataset contract requires dataset.selected_seg_ids. "
                "Use dataset/metaworld/tools/inspect_segmentation_ids.py to list available IDs."
            )
        if isinstance(self.selected_seg_ids, list):
            ids = self.selected_seg_ids
        else:
            path = Path(dataset_path)
            keys = (env_name, demo_key, path.stem, path.name, "default", "*")
            ids = None
            for key in keys:
                if key in self.selected_seg_ids:
                    ids = self.selected_seg_ids[key]
                    break
            if ids is None:
                raise ValueError(
                    f"No selected_seg_ids entry matched env={env_name!r}, demo={demo_key!r}, path={path.name!r}. "
                    "Add one of those keys or a 'default' entry."
                )
        if ids is None or len(ids) == 0:
            raise ValueError(f"Selected segmentation ID list is empty for env={env_name!r}, demo={demo_key!r}.")
        return list(ids)

    def _index_files_and_build_samples(self) -> None:
        refs_by_file: Dict[int, List[str]] = {}
        for file_idx, demo_key in self.demo_refs:
            refs_by_file.setdefault(file_idx, []).append(demo_key)

        for file_idx, demo_keys in refs_by_file.items():
            dataset_path = self.dataset_paths[file_idx]
            with h5py.File(dataset_path, "r") as f:
                if "data" not in f:
                    raise ValueError(f'Invalid dataset "{dataset_path}": missing top-level group "data".')

                data_grp = f["data"]
                for demo_key in demo_keys:
                    if demo_key not in data_grp:
                        raise ValueError(f'Demo key "{demo_key}" not found under /data in "{dataset_path}".')

                    demo_grp = data_grp[demo_key]
                    env_name = str(demo_grp.attrs.get("env_name", "default"))
                    if "camera_names" not in demo_grp.attrs:
                        raise ValueError(f'"/data/{demo_key}" in "{dataset_path}" has no "camera_names" attribute.')
                    camera_names = json.loads(demo_grp.attrs["camera_names"])

                    for view in self.views:
                        if view not in camera_names:
                            raise ValueError(
                                f'View "{view}" not found in "{dataset_path}" demo "{demo_key}" '
                                f"camera_names={camera_names}."
                            )

                    if "obs" not in demo_grp:
                        raise ValueError(f'"/data/{demo_key}" in "{dataset_path}" has no "obs" group.')
                    obs_grp = demo_grp["obs"]

                    ref_view = self.views[0]
                    ref_name = f"{ref_view}_rgb"
                    if ref_name not in obs_grp:
                        raise ValueError(
                            f'Missing dataset "{dataset_path}:/data/{demo_key}/obs/{ref_name}". '
                            f"Available keys: {list(obs_grp.keys())[:20]} ..."
                        )
                    for view in self.views:
                        rgb_name = f"{view}_rgb"
                        if rgb_name not in obs_grp:
                            raise ValueError(f'Missing dataset "{dataset_path}:/data/{demo_key}/obs/{rgb_name}".')
                        depth_name = f"{view}_depth"
                        if depth_name not in obs_grp:
                            raise ValueError(
                                f'Missing depth dataset "{dataset_path}:/data/{demo_key}/obs/{depth_name}". '
                                "The fixed training contract requires depth."
                            )
                        seg_name = f"{view}_seg"
                        if seg_name not in obs_grp:
                            raise ValueError(
                                f'Missing segmentation dataset "{dataset_path}:/data/{demo_key}/obs/{seg_name}". '
                                "The fixed training contract requires segmentation IDs."
                            )

                    demo_ref = (file_idx, demo_key)
                    self.demo_mask_selectors[demo_ref] = self._selected_ids_for_demo(
                        dataset_path, demo_key, env_name
                    )

                    timesteps = int(obs_grp[ref_name].shape[0])
                    if self.max_frames_per_demo is not None:
                        timesteps = min(timesteps, int(self.max_frames_per_demo))
                    self.demo_lengths[demo_ref] = timesteps

                    if OPTICAL_FLOW_GROUP not in demo_grp:
                        raise ValueError(
                            f'Missing flow group "{dataset_path}:/data/{demo_key}/{OPTICAL_FLOW_GROUP}". '
                            "Run the WAFT preprocessing utility first."
                        )
                    flow_grp = demo_grp[OPTICAL_FLOW_GROUP]
                    if not bool(flow_grp.attrs.get("preprocessing_complete", False)):
                        raise ValueError(
                            f'Flow preprocessing is incomplete for "{dataset_path}:/data/{demo_key}".'
                        )
                    available_gaps = tuple(
                        int(value)
                        for value in np.asarray(
                            flow_grp.attrs.get("available_temporal_gaps", ()),
                            dtype=np.int64,
                        ).tolist()
                    )
                    if available_gaps != FLOW_TEMPORAL_GAPS:
                        raise ValueError(
                            f"Invalid available_temporal_gaps for {demo_key}: {available_gaps}."
                        )
                    flow_direction = flow_grp.attrs.get("flow_direction", "")
                    flow_units = flow_grp.attrs.get("flow_units", "")
                    if isinstance(flow_direction, bytes):
                        flow_direction = flow_direction.decode("utf-8")
                    if isinstance(flow_units, bytes):
                        flow_units = flow_units.decode("utf-8")
                    if str(flow_direction) != "forward" or str(flow_units) != "pixel_displacement":
                        raise ValueError(
                            f"Unsupported optical-flow convention for {demo_key}: "
                            f"direction={flow_direction!r}, units={flow_units!r}."
                        )
                    for view in self.views:
                        if view not in flow_grp:
                            raise ValueError(f'Missing optical-flow camera group for {demo_key}/{view}.')
                        for gap in FLOW_TEMPORAL_GAPS:
                            name = f"gap_{gap}"
                            if name not in flow_grp[view]:
                                raise ValueError(f'Missing optical-flow dataset {demo_key}/{view}/{name}.')
                            rgb_shape = obs_grp[f"{view}_rgb"].shape
                            expected_shape = (
                                max(0, int(rgb_shape[0]) - gap),
                                int(rgb_shape[1]),
                                int(rgb_shape[2]),
                                2,
                            )
                            flow_ds = flow_grp[view][name]
                            if tuple(flow_ds.shape) != expected_shape or flow_ds.dtype != np.dtype(np.float16):
                                raise ValueError(
                                    f"Invalid shape for {demo_key}/{view}/{name}: {flow_ds.shape}; "
                                    f"expected float16 {expected_shape}."
                                )

                    if "camera_params" not in demo_grp:
                        raise ValueError(f'"/data/{demo_key}" in "{dataset_path}" has no "camera_params" group.')
                    cam_params_grp = demo_grp["camera_params"]
                    if "intrinsics" not in cam_params_grp or "extrinsics_world_T_cam" not in cam_params_grp:
                        raise ValueError(
                            f'"{dataset_path}:/data/{demo_key}/camera_params" must contain "intrinsics" '
                            'and "extrinsics_world_T_cam".'
                        )

                    intrinsics_ds = cam_params_grp["intrinsics"]
                    world_T_cam_ds = cam_params_grp["extrinsics_world_T_cam"]
                    camera_names_list = list(camera_names)
                    self.cam_cache[demo_ref] = {}

                    for view in self.views:
                        cam_idx = camera_names_list.index(view)
                        K = np.array(intrinsics_ds[cam_idx], dtype=np.float32)
                        world_T_cam = np.array(world_T_cam_ds[cam_idx], dtype=np.float32)

                        w2c_gl = invert_4x4(world_T_cam)
                        gl_to_cv = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)
                        w2c = gl_to_cv @ w2c_gl
                        c2w = invert_4x4(w2c).astype(np.float32)

                        self.cam_cache[demo_ref][view] = {"K": K, "w2c": w2c, "c2w": c2w}

                    self.cam_tensor_cache[demo_ref] = {
                        key: torch.from_numpy(
                            np.stack([self.cam_cache[demo_ref][view][key] for view in self.views], axis=0)
                        )
                        for key in ("K", "c2w", "w2c")
                    }

                    for t_idx in range(timesteps):
                        valid_strides = tuple(
                            stride
                            for stride in self.temporal_strides
                            if t_idx + (TEMPORAL_WINDOW - 1) * stride < timesteps
                        )
                        if valid_strides and (
                            self.fixed_temporal_stride is None
                            or self.fixed_temporal_stride in valid_strides
                        ):
                            self.samples.append((file_idx, demo_key, t_idx, valid_strides))

        unique_specs = sorted(
            {_format_seg_selectors(selectors) for selectors in self.demo_mask_selectors.values()}
        )
        print(
            f"[metaworldMultiViewTemporalHDF5Dataset] Indexed {len(self.demo_refs)} demos "
            f"from {len(self.dataset_paths)} file(s), {len(self.samples)} samples, "
            f"temporal_window={TEMPORAL_WINDOW}, temporal_strides={self.temporal_strides}, "
            f"fixed_temporal_stride={self.fixed_temporal_stride}, views={self.views}, "
            f"segmentation_selectors={unique_specs}"
        )
        if not self.samples:
            raise ValueError(
                "No temporal windows were found. Check max_frames_per_demo and the fixed "
                f"temporal strides {self.temporal_strides}."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.get_item(idx)

    def get_item(
        self,
        idx: int,
        temporal_stride_override: Optional[int] = None,
    ) -> Dict[str, Any]:
        file_idx, demo_key, t, valid_strides = self.samples[idx]
        demo_ref = (file_idx, demo_key)
        demo_grp = self._get_h5(file_idx)["data"][demo_key]
        obs_grp = demo_grp["obs"]

        selected_selectors = self.demo_mask_selectors.get(demo_ref, [])
        requires_seg_type = any(obj_type is not None for obj_type, _obj_id in selected_selectors)
        if temporal_stride_override is None:
            temporal_stride = (
                int(self.rng.choice(valid_strides))
                if self.fixed_temporal_stride is None
                else self.fixed_temporal_stride
            )
        else:
            temporal_stride = int(temporal_stride_override)
            if temporal_stride not in valid_strides:
                raise ValueError(
                    f"Stride {temporal_stride} is invalid for {demo_key} start {t}; "
                    f"valid strides are {valid_strides}."
                )
        t_indices = [int(t + offset * temporal_stride) for offset in range(TEMPORAL_WINDOW)]

        images_by_view = []
        depths_by_view = []
        masks_by_view = []
        flows_by_view = []
        for view in self.views:
            # Each modality is read once per camera for the complete temporal
            # window. HDF5 fancy indexing supports these strictly increasing IDs.
            rgb_block = np.asarray(obs_grp[f"{view}_rgb"][t_indices], dtype=np.uint8)
            images_by_view.append(torch.from_numpy(rgb_block).permute(0, 3, 1, 2))

            depth_block = np.asarray(obs_grp[f"{view}_depth"][t_indices], dtype=np.float32)
            if depth_block.ndim == 4 and depth_block.shape[-1] == 1:
                depth_block = depth_block[..., 0]
            depths_by_view.append(torch.from_numpy(depth_block).unsqueeze(1))

            seg_block = np.asarray(obs_grp[f"{view}_seg"][t_indices], dtype=np.int32)
            seg_type_block = None
            if requires_seg_type:
                seg_type_name = f"{view}_seg_type"
                if seg_type_name not in obs_grp:
                    raise ValueError(
                        f"Typed selected_seg_ids require dataset '{seg_type_name}'. "
                        "Recollect with segmentation.save_objtype=true or use plain integer IDs."
                    )
                seg_type_block = np.asarray(obs_grp[seg_type_name][t_indices], dtype=np.int32)
            mask_block = _segmentation_mask_from_selectors(
                seg_block, seg_type_block, selected_selectors
            )
            masks_by_view.append(torch.from_numpy(mask_block).unsqueeze(1))

            flow_view_grp = demo_grp[OPTICAL_FLOW_GROUP][view]
            flow_01 = np.asarray(flow_view_grp[f"gap_{temporal_stride}"][t], dtype=np.float16)
            flow_12 = np.asarray(
                flow_view_grp[f"gap_{temporal_stride}"][t + temporal_stride],
                dtype=np.float16,
            )
            flow_02 = np.asarray(
                flow_view_grp[f"gap_{2 * temporal_stride}"][t], dtype=np.float16
            )
            flows_by_view.append(
                torch.from_numpy(np.stack((flow_01, flow_12, flow_02), axis=0)).permute(0, 3, 1, 2)
            )

        cameras = self.cam_tensor_cache[demo_ref]
        sample = {
            "images": torch.stack(images_by_view, dim=1),
            "K": cameras["K"],
            "c2w": cameras["c2w"],
            "w2c": cameras["w2c"],
            "demo_key": _demo_label(file_idx, demo_key, len(self.dataset_paths)),
            "hdf5_path": self.dataset_paths[file_idx],
            "file_idx": int(file_idx),
            "t": int(t),
            "t_indices": torch.tensor(t_indices, dtype=torch.long),
            "temporal_stride": int(temporal_stride),
        }
        sample["depths"] = torch.stack(depths_by_view, dim=1)
        sample["masks"] = torch.stack(masks_by_view, dim=1)
        sample["optical_flows"] = torch.stack(flows_by_view, dim=1)
        return sample



# -------------------------------------------------------------------------
# Helpers for building DataLoaders (train / valid)
# -------------------------------------------------------------------------


def _list_demo_keys_metaworld(dataset_path: str) -> List[str]:
    with h5py.File(dataset_path, "r") as f:
        if "data" not in f:
            raise ValueError(f'Invalid dataset "{dataset_path}": missing top-level group "data".')
        demos = list(f["data"].keys())

    def _demo_index(key: str) -> int:
        try:
            return int(key.replace("demo", ""))
        except Exception:
            return 10**18

    return sorted(demos, key=_demo_index)


def _list_demo_refs_metaworld(dataset_paths: Sequence[str]) -> List[DemoRef]:
    refs: List[DemoRef] = []
    for file_idx, path in enumerate(dataset_paths):
        refs.extend((file_idx, demo_key) for demo_key in _list_demo_keys_metaworld(path))
    return refs


def _random_demo_split(
    demo_refs: Sequence[DemoRef],
    train_ratio: float,
    seed: int,
) -> tuple[List[DemoRef], List[DemoRef]]:
    refs = list(demo_refs)
    if len(refs) < 2:
        raise ValueError(
            "At least two demonstrations are required for disjoint train and validation splits."
        )
    if not 0.0 < float(train_ratio) < 1.0:
        raise ValueError(f"train_ratio must be strictly between zero and one, got {train_ratio}.")
    random.Random(int(seed)).shuffle(refs)
    num_train = min(len(refs) - 1, max(1, int(len(refs) * float(train_ratio))))
    train_refs = refs[:num_train]
    valid_refs = refs[num_train:]
    if set(train_refs) & set(valid_refs):
        raise RuntimeError("Train and validation demonstration splits overlap.")
    return train_refs, valid_refs


def _save_split_manifest(
    path: str,
    train_refs: Sequence[DemoRef],
    valid_refs: Sequence[DemoRef],
    seed: int,
) -> None:
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    def encode(ref: DemoRef) -> dict[str, int | str]:
        return {"file_idx": int(ref[0]), "demo_key": str(ref[1])}

    payload = {
        "seed": int(seed),
        "train": [encode(ref) for ref in train_refs],
        "validation": [encode(ref) for ref in valid_refs],
    }
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _worker_init_fn(worker_id: int) -> None:
    info = get_worker_info()
    if info is None:
        return
    info.dataset.rng = random.Random(info.seed)


def build_train_valid_loaders_metaworld(
    dataset_path: DatasetPathInput,
    views: List[str],
    selected_seg_ids: SegIdSpec,
    batch_size: int = 128,
    num_workers: int = 4,
    pin_memory: bool = True,
    train_ratio: float = 0.9,
    seed: int = 0,
    num_episodes: Optional[int] = None,
    max_frames_per_demo: Optional[int] = None,
    train_temporal_strides: Sequence[int] = TRAIN_TEMPORAL_STRIDES,
    validation_temporal_stride: int = 9,
    split_manifest_path: Optional[str] = None,
    drop_last_train: bool = True,
    shuffle_train: bool = True,
):
    """Build fixed-contract train/validation loaders from a seeded demo split."""
    dataset_paths = _normalize_dataset_paths(dataset_path)
    demo_refs = _list_demo_refs_metaworld(dataset_paths)
    if num_episodes is not None:
        demo_refs = demo_refs[: int(num_episodes)]
    train_refs, valid_refs = _random_demo_split(demo_refs, train_ratio, seed)

    if split_manifest_path is None:
        first_path = Path(dataset_paths[0])
        split_manifest_path = str(
            first_path.parent / "splits" / f"{first_path.stem}_seed{seed}.json"
        )
    _save_split_manifest(split_manifest_path, train_refs, valid_refs, seed)
    print(
        f"[Dataset split] train={len(train_refs)}, validation={len(valid_refs)}, "
        f"manifest={split_manifest_path}"
    )

    explicit_views = list(views)
    if not explicit_views:
        raise ValueError("dataset.views must be an explicit non-empty camera list.")
    common = dict(
        dataset_path=dataset_paths,
        views=explicit_views,
        max_frames_per_demo=max_frames_per_demo,
        temporal_strides=train_temporal_strides,
        selected_seg_ids=selected_seg_ids,
    )
    train_dataset = metaworldMultiViewTemporalHDF5Dataset(
        demo_keys=train_refs, seed=seed, **common
    )
    valid_dataset = metaworldMultiViewTemporalHDF5Dataset(
        demo_keys=valid_refs,
        seed=seed + 999,
        fixed_temporal_stride=int(validation_temporal_stride),
        **common,
    )

    train_worker_options = (
        {"persistent_workers": True, "prefetch_factor": 2} if num_workers > 0 else {}
    )
    valid_workers = max(1, num_workers // 2) if num_workers > 0 else 0
    valid_worker_options = (
        {"persistent_workers": True, "prefetch_factor": 2}
        if valid_workers > 0
        else {}
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last_train,
        worker_init_fn=_worker_init_fn,
        **train_worker_options,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=valid_workers,
        pin_memory=pin_memory,
        drop_last=False,
        worker_init_fn=_worker_init_fn,
        **valid_worker_options,
    )
    return train_loader, valid_loader
