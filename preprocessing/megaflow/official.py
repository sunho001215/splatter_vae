from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

MEGAFLOW_REPOSITORY_REVISION = "ee5b61813db0a76ac0db9034899aade72a0d230c"
MEGAFLOW_CHECKPOINT_ID = "Kristen-Z/MegaFlow"
MEGAFLOW_CHECKPOINT_REVISION = "b4c5c33800b8fa88e047d2eb70ae74b0feca606d"
MEGAFLOW_CHECKPOINT_FILENAME = "megaflow-flow.safetensors"


@dataclass(frozen=True)
class MegaFlowInferenceConfig:
    refinement_iterations: int = 8
    maximum_sequence_frames: int = 12
    amp_dtype: str = "bf16"
    attention_backend: str = "pytorch_sdpa"

    def __post_init__(self) -> None:
        if self.refinement_iterations <= 0:
            raise ValueError("MegaFlow refinement iterations must be positive.")
        if self.maximum_sequence_frames < 2:
            raise ValueError("MegaFlow chunks require at least two frames.")
        if self.amp_dtype != "bf16":
            raise ValueError("The validated MegaFlow offline path uses BF16 autocast.")
        if self.attention_backend != "pytorch_sdpa":
            raise ValueError("The validated Blackwell MegaFlow path uses PyTorch SDPA.")


class MegaFlowDROIDTeacher:
    """Pinned official MegaFlow forward-flow adapter at native 180x320."""

    def __init__(
        self,
        repository: str | Path,
        *,
        cache_dir: str | Path,
        device: torch.device | str = "cuda:0",
        config: MegaFlowInferenceConfig | None = None,
    ) -> None:
        repository_path = Path(repository).expanduser().resolve()
        if not (repository_path / "megaflow" / "model" / "megaflow.py").is_file():
            raise FileNotFoundError(
                f"Official MegaFlow source is missing: {repository_path}"
            )
        from huggingface_hub import hf_hub_download

        checkpoint = hf_hub_download(
            repo_id=MEGAFLOW_CHECKPOINT_ID,
            filename=MEGAFLOW_CHECKPOINT_FILENAME,
            revision=MEGAFLOW_CHECKPOINT_REVISION,
            cache_dir=str(Path(cache_dir).expanduser().resolve()),
        )
        if str(repository_path) not in sys.path:
            sys.path.insert(0, str(repository_path))
        if "megaflow.model.layers.attention" in sys.modules:
            raise RuntimeError(
                "MegaFlow attention was imported before selecting its audited backend."
            )
        # The optional xFormers wheel resolves successfully on this host but its
        # memory-efficient kernel raises CUDA invalid-argument on Blackwell.
        # MegaFlow's official implementation has a native PyTorch SDPA fallback.
        os.environ["XFORMERS_DISABLED"] = "1"
        from megaflow import MegaFlow

        self.device = torch.device(device)
        # Preserve the official recommended constructor while forcing its
        # internal hub lookup to the audited immutable revision above.
        with patch("huggingface_hub.hf_hub_download", return_value=str(checkpoint)):
            model = MegaFlow.from_pretrained("megaflow-flow", device=str(self.device))
        self.model = model.eval()
        self.model.requires_grad_(False)
        self.config = config or MegaFlowInferenceConfig()
        self.repository_path = str(repository_path)
        self.checkpoint_path = str(Path(checkpoint).resolve())

    @torch.inference_mode()
    def infer_sequence(self, frames: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Infer every consecutive forward field, using one-frame chunk overlap."""

        value = torch.as_tensor(frames)
        if value.dim() == 4:
            value = value.unsqueeze(0)
        if value.dim() != 5 or value.shape[2:] != (3, 180, 320):
            raise ValueError(
                f"MegaFlow expects (B,T,3,180,320) RGB in [0,255], got {tuple(value.shape)}."
            )
        if value.shape[1] < 2:
            return torch.empty(value.shape[0], 0, 2, 180, 320, dtype=torch.float32)
        value = value.to(self.device, dtype=torch.float32, non_blocking=True)
        outputs = []
        start = 0
        maximum = int(self.config.maximum_sequence_frames)
        while start < value.shape[1] - 1:
            stop = min(start + maximum, int(value.shape[1]))
            chunk = value[:, start:stop]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                result = self.model(
                    chunk,
                    num_reg_refine=int(self.config.refinement_iterations),
                )["flow_preds"][-1]
            if result.shape != (value.shape[0], stop - start - 1, 2, 180, 320):
                raise RuntimeError(
                    f"MegaFlow returned unexpected shape {tuple(result.shape)}."
                )
            outputs.append(result.float().cpu())
            start = stop - 1
        output = torch.cat(outputs, dim=1)
        if output.shape[1] != value.shape[1] - 1:
            raise RuntimeError(
                "MegaFlow chunk overlap lost or duplicated a temporal pair."
            )
        return output

    @torch.inference_mode()
    def infer_retained_gap6(
        self, retained_rgb: np.ndarray | torch.Tensor
    ) -> np.ndarray:
        """Infer both retained phases and return sources 0..R-3 in timeline order.

        Input is (R,2,180,320,3) or (R,2,3,180,320). Output is
        (R-2,2,2,180,320): source retained index, camera, component, y, x.
        """

        value = torch.as_tensor(retained_rgb)
        if value.shape[-1] == 3:
            value = value.permute(0, 1, 4, 2, 3)
        if value.dim() != 5 or value.shape[1:] != (2, 3, 180, 320):
            raise ValueError(f"Unexpected retained RGB shape {tuple(value.shape)}.")
        retained = int(value.shape[0])
        output = torch.empty(max(0, retained - 2), 2, 2, 180, 320)
        for phase in (0, 1):
            phase_frames = value[phase::2].permute(1, 0, 2, 3, 4).contiguous()
            if phase_frames.shape[1] < 2:
                continue
            phase_flow = self.infer_sequence(phase_frames)
            source_ids = torch.arange(phase, retained - 2, 2)
            if source_ids.numel() != phase_flow.shape[1]:
                raise RuntimeError("MegaFlow phase reconstruction is inconsistent.")
            output[source_ids] = phase_flow.permute(1, 0, 2, 3, 4)
        return output.numpy()

    def metadata(self) -> dict[str, object]:
        return {
            "repository": "cvg/megaflow",
            "repository_revision": MEGAFLOW_REPOSITORY_REVISION,
            "model_id": "megaflow-flow",
            "checkpoint_repository": MEGAFLOW_CHECKPOINT_ID,
            "checkpoint_revision": MEGAFLOW_CHECKPOINT_REVISION,
            "checkpoint_filename": MEGAFLOW_CHECKPOINT_FILENAME,
            "checkpoint_path": self.checkpoint_path,
            "flow_direction": "forward_t_to_tplus6",
            "native_resolution": [180, 320],
            **asdict(self.config),
        }
