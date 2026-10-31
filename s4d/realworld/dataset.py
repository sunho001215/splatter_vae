"""Random-access loader for exported real-world windows; implements the shared batch contract (track form)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import h5py
import torch
from torch.utils.data import Dataset

from s4d.data.contract import validate_batch
from s4d.realworld.export import FORMAT


class RealworldCacheDataset(Dataset):
    def __init__(self, root, *, split="train", image_height=144, image_width=256, with_eval=False):
        self.root = Path(root).resolve(strict=True)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if self.manifest.get("format") != FORMAT:
            raise ValueError(f"real-world cache must be {FORMAT!r}, got {self.manifest.get('format')!r}")
        if self.manifest.get("MOCK_MODEL_OUTPUTS"):
            import warnings

            warnings.warn("this cache was built from MOCK model outputs (tests only)", stacklevel=2)
        names = [k for k, v in self.manifest["resolutions"].items() if tuple(v) == (image_height, image_width)]
        if len(names) != 1:
            raise ValueError(f"available sizes {self.manifest['resolutions']}")
        self.path = self.root / f"{names[0]}.h5"
        self.indices = [i for i, w in enumerate(self.manifest["windows"]) if w["split"] == split]
        if not self.indices:
            raise ValueError(f"empty split {split}")
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
            self._file = h5py.File(self.path, "r")
            self._pid = os.getpid()
        gi = self.indices[index]
        group = self._file[f"windows/{gi:05d}"]
        sample = {name: torch.from_numpy(group[name][:]) for name in group}
        w = self.manifest["windows"][gi]
        sample["meta"] = {"task": "realworld", "episode": self.manifest["episode"],
                          "t_indices": tuple(w["canonical_indices"]), "stride": w["stride"],
                          "camera_ids": tuple(self.manifest["cameras"])}
        validate_batch({k: v if k == "meta" else v.unsqueeze(0) for k, v in sample.items()})
        return sample
