"""Configuration dataclasses of the ReViWo model.

Copied verbatim from the reference repository (sunho001215/splatter_vae @ c0abf56, ``baselines/ReViWo/transformer.py``):
the upstream ``MultiViewBetaVAE`` calls ``config.update(...)`` and uses the returned config, which these dataclasses
provide.
"""

from dataclasses import dataclass

@dataclass
class CodebookConfig:
    n_embed: int = 512
    embed_dim: int = 64
    beta: float = 0.25


@dataclass
class STTransConfig:
    block_size: int = 4*4
    vocab_size: int = 0
    n_tokens_per_frame: int = 4*4
    n_layer: int = 8
    n_head: int = 8
    n_embed: int = 256
    dropout: float = 0.1
    bias: bool = False 
    mask_rate: float = None

    def update(self, overrides: dict) -> "STTransConfig":
        """Return a new config with `overrides` applied."""
        d = self.__dict__.copy()
        d.update(overrides)
        return STTransConfig(**d)

