"""Random-access real-sample cache loader implementing the shared batch contract."""

from __future__ import annotations

import json
import os
from pathlib import Path

import h5py
import torch
from torch.utils.data import Dataset

from s4d.data.contract import validate_batch
from s4d.data.droid.convert import RESOLUTIONS


class DroidCacheDataset(Dataset):
    """Each item is a complete three-time, two-external-camera window.

    with_eval is accepted for shared training integration. This sample has no
    held-out static camera, so optional evaluation camera fields stay absent.
    Validation is a temporally disjoint clip of the same episode.
    """

    def __init__(self, root, *, split="train", with_eval=False, image_height=144, image_width=256):
        self.root = Path(root).resolve(strict=True)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        shape = (int(image_height), int(image_width))
        backbones = [key for key, value in RESOLUTIONS.items() if value == shape]
        if len(backbones) != 1:
            raise ValueError(f"supported DROID sample image sizes: {RESOLUTIONS}, got {shape}")
        self.path = self.root / f"{backbones[0]}.h5"
        if split not in ("train", "validation"):
            raise ValueError("sample split must be train or validation")
        self.indices = [i for i, row in enumerate(self.manifest["windows"]) if row["split"] == split]
        if not self.indices:
            raise ValueError(f"empty sample split {split}")
        self._file = None
        self._pid = None

    def __len__(self):
        return len(self.indices)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file"] = state["_pid"] = None
        return state

    def __getitem__(self, index):
        if self._file is None or self._pid != os.getpid():
            if self._file is not None:
                self._file.close()
            self._file = h5py.File(self.path, "r")
            self._pid = os.getpid()
        global_index = self.indices[index]
        group = self._file[f"windows/{global_index:05d}"]
        sample = {name: torch.from_numpy(group[name][:]) for name in group}
        window = self.manifest["windows"][global_index]
        sample["meta"] = {
            "task": "droid_sample",
            "episode": self.manifest["episode"],
            "clip": window["clip"],
            "t_indices": tuple(window["canonical_indices"]),
            "raw_indices": tuple(window["raw_indices"]),
            "stride": 3,
            "camera_serials": tuple(self.manifest["camera_serials"]),
        }
        # Fail at the loader boundary; shared collate validates again after stacking.
        validate_batch({name: value if name == "meta" else value.unsqueeze(0) for name, value in sample.items()})
        return sample

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None
