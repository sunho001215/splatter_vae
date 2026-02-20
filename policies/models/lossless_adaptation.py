import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple

# -----------------------------
# Lossless Adaptation: Bottleneck Adapters for Swin Transformer
# -----------------------------

class TransformerBottleneckAdapter(nn.Module):
    """
    Bottleneck adapter: x + Up(Act(Down(LN(x))))
    - "Lossless" at init: Up is zero-initialized => adapter outputs ~0 => identity mapping.
    - Works on token tensors shaped (B, N, C).
    """
    def __init__(
        self,
        dim: int,
        bottleneck_dim: int,
        dropout: float = 0.0,
        use_layernorm: bool = True,
        init_up_zero: bool = True,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim) if use_layernorm else nn.Identity()
        self.down = nn.Linear(dim, bottleneck_dim, bias=True)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.up = nn.Linear(bottleneck_dim, dim, bias=True)

        # "Lossless" initialization: adapter starts as near-identity function
        if init_up_zero:
            nn.init.zeros_(self.up.weight)
            nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.down(h)
        h = self.act(h)
        h = self.drop(h)
        h = self.up(h)
        return x + h


class _SwinBlockWithAdapter(nn.Module):
    """
    Wrap an existing SwinTransformerBlock without modifying its source file:
      x -> block(x) -> adapter(block(x))
    """
    def __init__(self, block: nn.Module, adapter: TransformerBottleneckAdapter):
        super().__init__()
        self.block = block
        self.adapter = adapter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        x = self.adapter(x)
        return x


def inject_swin_middle_adapters(
    swin_encoder: nn.Module,
    adapter_cfg: Dict[str, Any],
) -> None:
    """
    Inject 6 adapters into Swin stages/blocks:
      - stage1: block0
      - stage2: block0
      - stage3: block0, block1
      - stage4: block0, block1

    NOTE: This function mutates `swin_encoder.layers[s].blocks[b]` in-place,
    so we do NOT need to edit `models/swin_transformer.py`.
    """
    enabled = bool(adapter_cfg.get("enabled", False))
    if not enabled:
        return

    # Default positions required by your spec (6 adapters total)
    positions: List[Tuple[int, int]] = adapter_cfg.get(
        "positions",
        [(0, 0), (1, 0), (2, 0), (2, 1), (3, 0), (3, 1)],
    )

    bottleneck_ratio = int(adapter_cfg.get("bottleneck_ratio", 16))  # D -> D/ratio -> D
    dropout = float(adapter_cfg.get("dropout", 0.0))
    use_layernorm = bool(adapter_cfg.get("use_layernorm", True))

    # SwinTransformerV2 has `.layers`, and each layer has `.blocks`
    if not hasattr(swin_encoder, "layers"):
        raise ValueError("Expected a Swin encoder with attribute `.layers` (SwinTransformerV2).")

    if len(swin_encoder.layers) < 4:
        raise ValueError(f"Expected >=4 Swin stages, got {len(swin_encoder.layers)}.")

    for (stage_idx, block_idx) in positions:
        layer = swin_encoder.layers[stage_idx]
        if not hasattr(layer, "blocks"):
            raise ValueError(f"Swin stage {stage_idx} has no `.blocks`.")

        if block_idx >= len(layer.blocks):
            raise ValueError(
                f"Swin stage {stage_idx} has only {len(layer.blocks)} blocks; "
                f"cannot place adapter at block {block_idx}."
            )

        # Stage embedding dim is stored on the BasicLayer as `dim`
        dim = int(getattr(layer, "dim", None) or 0)
        if dim <= 0:
            raise ValueError(f"Could not infer embedding dim for stage {stage_idx}.")

        bottleneck_dim = max(1, dim // bottleneck_ratio)

        adapter = TransformerBottleneckAdapter(
            dim=dim,
            bottleneck_dim=bottleneck_dim,
            dropout=dropout,
            use_layernorm=use_layernorm,
            init_up_zero=True,  # "lossless" start
        )

        # Wrap the existing block in-place (no edits to Swin source)
        layer.blocks[block_idx] = _SwinBlockWithAdapter(layer.blocks[block_idx], adapter)