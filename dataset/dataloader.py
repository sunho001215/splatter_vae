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
SampleRef = Tuple[int, str, int]
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
        images: (T, N_cam, 3, H, W) float32 in [-1, 1]
        K:      (N_cam, 3, 3) camera intrinsics for static cameras
        c2w:    (N_cam, 4, 4) OpenCV camera-to-world transforms
        w2c:    (N_cam, 4, 4) OpenCV world-to-camera transforms
        depths: optional (T, N_cam, 1, H, W) float32 metric camera-z depth
        masks:  optional (T, N_cam, 1, H, W) float32 selected segmentation mask
    """

    def __init__(
        self,
        dataset_path: DatasetPathInput,
        demo_keys: List[Union[str, DemoRef]],
        views: Optional[List[str]] = None,
        camera_num: Optional[int] = None,
        max_frames_per_demo: Optional[int] = None,
        seed: int = 0,
        min_time_gap: int = 10,
        temporal_window: int = 3,
        temporal_stride: int = 1,
        temporal_stride_list: Optional[Sequence[int]] = None,
        temporal_min_state_change: float = 0.0,
        temporal_gripper_change_weight: float = 0.05,
        use_depth: bool = False,
        use_segmentation_mask: bool = False,
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
        self.camera_num = None if camera_num is None else int(camera_num)
        if self.camera_num is not None and self.camera_num <= 0:
            raise ValueError(f"camera_num must be positive when set, got {camera_num}.")

        # ``min_time_gap`` remains for backward-compatible config loading, but
        # it is no longer used because this dataset returns strided windows.
        self.min_time_gap = int(min_time_gap)
        self.temporal_window = max(1, int(temporal_window))
        self.temporal_stride = max(1, int(temporal_stride))
        if temporal_stride_list is None:
            self.temporal_stride_list = [self.temporal_stride]
        else:
            self.temporal_stride_list = sorted({max(1, int(stride)) for stride in temporal_stride_list})
            if len(self.temporal_stride_list) == 0:
                raise ValueError("temporal_stride_list must contain at least one positive stride.")
            self.temporal_stride = self.temporal_stride_list[0]
        self.temporal_min_state_change = max(0.0, float(temporal_min_state_change))
        self.temporal_gripper_change_weight = max(0.0, float(temporal_gripper_change_weight))
        self.use_depth = bool(use_depth)
        self.use_segmentation_mask = bool(use_segmentation_mask)
        self.selected_seg_ids = _normalize_seg_id_spec(selected_seg_ids)
        self.rng = random.Random(seed)

        self.views: Optional[List[str]] = None if views is None else list(views)
        if self.views is not None and self.camera_num is not None:
            self.views = self.views[: self.camera_num]

        self._h5_handles: Dict[int, h5py.File] = {}
        self.demo_lengths: Dict[DemoRef, int] = {}
        self.demo_mask_selectors: Dict[DemoRef, List[SegSelector]] = {}
        self.cam_cache: Dict[DemoRef, Dict[str, Dict[str, np.ndarray]]] = {}
        self.cam_tensor_cache: Dict[DemoRef, Dict[str, torch.Tensor]] = {}
        self.demo_motion_features: Dict[DemoRef, Optional[np.ndarray]] = {}
        self.demo_motion_feature_kinds: Dict[DemoRef, str] = {}
        self.samples: List[SampleRef] = []

        self._index_files_and_build_samples()

        if self.views is None or len(self.views) < 1:
            raise ValueError("You need at least one camera view.")

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
        if not self.use_segmentation_mask:
            return []
        if self.selected_seg_ids is None:
            raise ValueError(
                "dataset.use_segmentation_mask=true requires dataset.selected_seg_ids. "
                "Use dataset/metaworld_demo_collect/inspect_segmentation_ids.py to list available IDs."
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

                    if self.views is None:
                        inferred_views = list(camera_names)
                        if self.camera_num is not None:
                            inferred_views = inferred_views[: self.camera_num]
                        self.views = inferred_views
                    else:
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
                        if self.use_depth and depth_name not in obs_grp:
                            raise ValueError(
                                f'Missing depth dataset "{dataset_path}:/data/{demo_key}/obs/{depth_name}". '
                                "Collect depth images or set dataset.use_depth=false."
                            )
                        seg_name = f"{view}_seg"
                        if self.use_segmentation_mask and seg_name not in obs_grp:
                            raise ValueError(
                                f'Missing segmentation dataset "{dataset_path}:/data/{demo_key}/obs/{seg_name}". '
                                "Collect segmentation masks or set dataset.use_segmentation_mask=false."
                            )

                    demo_ref = (file_idx, demo_key)
                    if self.use_segmentation_mask:
                        self.demo_mask_selectors[demo_ref] = self._selected_ids_for_demo(dataset_path, demo_key, env_name)

                    timesteps = int(obs_grp[ref_name].shape[0])
                    if self.max_frames_per_demo is not None:
                        timesteps = min(timesteps, int(self.max_frames_per_demo))
                    self.demo_lengths[demo_ref] = timesteps

                    motion_features = None
                    motion_kind = ""
                    obs_env_grp = demo_grp.get("obs_env", None)
                    if obs_env_grp is not None and "obs" in obs_env_grp:
                        motion_features = np.asarray(obs_env_grp["obs"][:timesteps], dtype=np.float32)
                        motion_kind = "obs"
                    elif "states" in demo_grp:
                        motion_features = np.asarray(demo_grp["states"][:timesteps], dtype=np.float32)
                        motion_kind = "state"
                    if motion_features is not None and motion_features.ndim == 1:
                        motion_features = motion_features[:, None]
                    self.demo_motion_features[demo_ref] = motion_features
                    self.demo_motion_feature_kinds[demo_ref] = motion_kind

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

                    max_start = timesteps - self.temporal_window + 1
                    if max_start <= 0:
                        continue
                    for t_idx in range(max_start):
                        self.samples.append((file_idx, demo_key, t_idx))

        mask_msg = ""
        if self.use_segmentation_mask:
            unique_specs = sorted({_format_seg_selectors(selectors) for selectors in self.demo_mask_selectors.values()})
            mask_msg = f", segmentation_selectors={unique_specs}"
        print(
            f"[metaworldMultiViewTemporalHDF5Dataset] Indexed {len(self.demo_refs)} demos "
            f"from {len(self.dataset_paths)} file(s), {len(self.samples)} samples, "
            f"temporal_window={self.temporal_window}, temporal_stride_list={self.temporal_stride_list}, "
            f"temporal_min_state_change={self.temporal_min_state_change:g}, "
            f"views={self.views}{mask_msg}"
        )
        if len(self.samples) == 0:
            raise ValueError(
                f"No temporal windows were found. Check max_frames_per_demo, "
                f"temporal_window={self.temporal_window}, and temporal_stride_list={self.temporal_stride_list}."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def _largest_interval_indices(self, start: int, timesteps: int) -> List[int]:
        if self.temporal_window <= 1:
            return [int(start)]
        end = int(timesteps - 1)
        indices = [int(start)]
        span = max(0, end - int(start))
        for step in range(1, self.temporal_window):
            remaining_slots = self.temporal_window - 1 - step
            proposed = int(round(int(start) + span * step / max(1, self.temporal_window - 1)))
            min_allowed = indices[-1] + 1
            max_allowed = end - remaining_slots
            indices.append(max(min_allowed, min(proposed, max_allowed)))
        return indices

    def _state_change(self, demo_ref: DemoRef, t0: int, t1: int) -> float:
        features = self.demo_motion_features.get(demo_ref, None)
        if features is None or t0 >= features.shape[0] or t1 >= features.shape[0]:
            return float("inf")
        a = features[int(t0)]
        b = features[int(t1)]
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            return float("inf")
        if self.demo_motion_feature_kinds.get(demo_ref, "") == "obs" and a.shape[0] >= 4:
            ee_change = float(np.linalg.norm(b[:3] - a[:3]))
            gripper_change = abs(float(b[3] - a[3]))
            return ee_change + self.temporal_gripper_change_weight * gripper_change
        return float(np.linalg.norm(b - a))

    def _sample_time_indices(self, demo_ref: DemoRef, start: int, timesteps: int, temporal_stride: int) -> List[int]:
        if self.temporal_window <= 1:
            return [int(start)]
        if int(start) + (self.temporal_window - 1) * int(temporal_stride) >= int(timesteps):
            return self._largest_interval_indices(start, timesteps)

        if self.temporal_min_state_change <= 0.0 or self.demo_motion_features.get(demo_ref, None) is None:
            return [int(start + offset * temporal_stride) for offset in range(self.temporal_window)]

        indices = [int(start)]
        prev_t = int(start)
        for _offset in range(1, self.temporal_window):
            candidate = prev_t + int(temporal_stride)
            while candidate < int(timesteps) and self._state_change(demo_ref, prev_t, candidate) < self.temporal_min_state_change:
                candidate += 1
            if candidate >= int(timesteps):
                return self._largest_interval_indices(start, timesteps)
            indices.append(candidate)
            prev_t = candidate
        return indices

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        file_idx, demo_key, t = self.samples[idx]
        demo_ref = (file_idx, demo_key)
        obs_grp = self._get_h5(file_idx)["data"][demo_key]["obs"]

        selected_selectors = self.demo_mask_selectors.get(demo_ref, [])
        requires_seg_type = any(obj_type is not None for obj_type, _obj_id in selected_selectors)
        temporal_stride = self.rng.choice(self.temporal_stride_list)
        t_indices = self._sample_time_indices(
            demo_ref=demo_ref,
            start=int(t),
            timesteps=self.demo_lengths[demo_ref],
            temporal_stride=int(temporal_stride),
        )

        images_by_view = []
        depths_by_view = []
        masks_by_view = []
        for view in self.views:
            # Each modality is read once per camera for the complete temporal
            # window. HDF5 fancy indexing supports these strictly increasing IDs.
            rgb_block = np.asarray(obs_grp[f"{view}_rgb"][t_indices], dtype=np.uint8)
            image_block = torch.from_numpy(rgb_block).permute(0, 3, 1, 2).to(torch.float32)
            image_block = image_block.div(255.0).mul(2.0).sub(1.0)
            images_by_view.append(image_block)

            if self.use_depth:
                depth_block = np.asarray(obs_grp[f"{view}_depth"][t_indices], dtype=np.float32)
                if depth_block.ndim == 4 and depth_block.shape[-1] == 1:
                    depth_block = depth_block[..., 0]
                depths_by_view.append(torch.from_numpy(depth_block).unsqueeze(1))

            if self.use_segmentation_mask:
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
                    seg_block,
                    seg_type_block,
                    selected_selectors,
                )
                masks_by_view.append(
                    torch.from_numpy(mask_block.astype(np.float32)).unsqueeze(1)
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
        if self.use_depth:
            sample["depths"] = torch.stack(depths_by_view, dim=1)
        if self.use_segmentation_mask:
            sample["masks"] = torch.stack(masks_by_view, dim=1)
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


def _worker_init_fn(worker_id: int) -> None:
    info = get_worker_info()
    if info is None:
        return
    info.dataset.rng = random.Random(info.seed)


def build_train_valid_loaders_metaworld(
    dataset_path: DatasetPathInput,
    batch_size: int = 128,
    num_workers: int = 4,
    pin_memory: bool = True,
    train_ratio: float = 0.9,
    seed: int = 0,
    num_episodes: Optional[int] = None,
    max_frames_per_demo: Optional[int] = None,
    views: Optional[List[str]] = None,
    camera_num: Optional[int] = None,
    min_time_gap: int = 25,
    temporal_window: int = 3,
    temporal_stride: int = 1,
    temporal_stride_list: Optional[Sequence[int]] = None,
    temporal_min_state_change: float = 0.0,
    temporal_gripper_change_weight: float = 0.05,
    drop_last_train: bool = True,
    shuffle_train: bool = True,
    shuffle_valid: bool = True,
    use_depth: bool = False,
    use_segmentation_mask: bool = False,
    selected_seg_ids: SegIdSpec = None,
):
    """Build train/validation loaders with all selected cameras per sample."""
    dataset_paths = _normalize_dataset_paths(dataset_path)
    demo_refs = _list_demo_refs_metaworld(dataset_paths)
    if num_episodes is not None:
        demo_refs = demo_refs[: int(num_episodes)]

    rng = random.Random(seed)
    rng.shuffle(demo_refs)

    n_total = len(demo_refs)
    n_train = max(1, int(n_total * train_ratio))
    n_train = min(n_train, n_total - 1) if n_total > 1 else n_train
    train_refs = demo_refs[:n_train]
    valid_refs = demo_refs[n_train:] if n_total > 1 else demo_refs[:]

    if views is None:
        first_file_idx, first_demo = train_refs[0]
        with h5py.File(dataset_paths[first_file_idx], "r") as f:
            views = list(json.loads(f["data"][first_demo].attrs["camera_names"]))
    else:
        views = list(views)

    if camera_num is not None:
        views = views[: int(camera_num)]

    train_dataset = metaworldMultiViewTemporalHDF5Dataset(
        dataset_path=dataset_paths,
        demo_keys=train_refs,
        views=views,
        camera_num=camera_num,
        max_frames_per_demo=max_frames_per_demo,
        seed=seed,
        min_time_gap=min_time_gap,
        temporal_window=temporal_window,
        temporal_stride=temporal_stride,
        temporal_stride_list=temporal_stride_list,
        temporal_min_state_change=temporal_min_state_change,
        temporal_gripper_change_weight=temporal_gripper_change_weight,
        use_depth=use_depth,
        use_segmentation_mask=use_segmentation_mask,
        selected_seg_ids=selected_seg_ids,
    )
    valid_dataset = metaworldMultiViewTemporalHDF5Dataset(
        dataset_path=dataset_paths,
        demo_keys=valid_refs,
        views=views,
        camera_num=camera_num,
        max_frames_per_demo=max_frames_per_demo,
        seed=seed + 999,
        min_time_gap=min_time_gap,
        temporal_window=temporal_window,
        temporal_stride=temporal_stride,
        temporal_stride_list=temporal_stride_list,
        temporal_min_state_change=temporal_min_state_change,
        temporal_gripper_change_weight=temporal_gripper_change_weight,
        use_depth=use_depth,
        use_segmentation_mask=use_segmentation_mask,
        selected_seg_ids=selected_seg_ids,
    )

    train_worker_options = {}
    if num_workers > 0:
        train_worker_options = {"persistent_workers": True, "prefetch_factor": 2}
    valid_workers = max(1, num_workers // 2)
    valid_worker_options = {"persistent_workers": True, "prefetch_factor": 2}

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
        shuffle=shuffle_valid,
        num_workers=valid_workers,
        pin_memory=pin_memory,
        drop_last=True,
        worker_init_fn=_worker_init_fn,
        **valid_worker_options,
    )

    return train_loader, valid_loader
