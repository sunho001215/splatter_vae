"""Real-world (Franka Panda + Robotiq + three RealSense rigs) preprocessing pipeline.

Self-contained: nothing outside ``s4d.realworld`` imports this package, and this package never imports upstream
model code (FoundationStereo, TrackEverything, SAM 3, DINO). Those run in separate, pinned environments behind
``s4d.realworld.workers``. See docs/realworld_preprocessing.md and docs/realworld_preprocessing_STATUS.md.

Everything that depends on a recording's on-disk layout lives in an adapter (``s4d.realworld.adapters``); the
numerical stages are tested on synthetic fixtures and validated on recorded episodes (see the status document).
"""

from s4d.realworld.contract import SCHEMA_VERSION

__all__ = ["SCHEMA_VERSION"]
