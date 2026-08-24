from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path
from typing import Any

import torch

from dataset.droid.safety import validate_derived_root


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def path_identity(path: str | os.PathLike[str]) -> str:
    value = Path(path).expanduser().resolve()
    if value.is_file():
        return f"{value.name}:{sha256_file(value)}"
    if not value.is_dir():
        raise FileNotFoundError(value)
    digest = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    for child in sorted((item for item in value.rglob("*") if item.is_file()), key=str):
        relative = child.relative_to(value).as_posix()
        size = child.stat().st_size
        digest.update(f"{relative}\0{size}\n".encode())
        file_count += 1
        total_bytes += size
    return f"{value.name}:tree:{file_count}:{total_bytes}:{digest.hexdigest()}"


def git_revision(repository: str | os.PathLike[str]) -> str:
    root = Path(repository).expanduser().resolve()
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def configure_external_model_caches(
    derived_root: str | os.PathLike[str],
    droid_root: str | os.PathLike[str],
) -> dict[str, str]:
    """Keep framework and model caches outside source data and the code tree."""
    root = validate_derived_root(
        Path(derived_root) / "metadata" / "model_cache", droid_root
    )
    locations = {
        "TORCH_HOME": root / "torch",
        "HF_HOME": root / "huggingface",
        "XDG_CACHE_HOME": root / "xdg",
    }
    for name, path in locations.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    return {name: str(path) for name, path in locations.items()}


def verify_cuda_device() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "This teacher requires CUDA; set CUDA_VISIBLE_DEVICES before launch."
        )
    torch.cuda.set_device(0)
    return {
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        "torch_cuda_current_device": torch.cuda.current_device(),
        "torch_cuda_device_name": torch.cuda.get_device_name(0),
        "mapped_device": "cuda:0",
    }
