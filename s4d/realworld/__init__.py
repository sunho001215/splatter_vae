"""Real-world (Franka Panda + Robotiq + three RealSense rigs) preprocessing pipeline.

Self-contained: nothing outside ``s4d.realworld`` imports this package, and this package never imports upstream
model code (FoundationStereo, AllTracker, SAM 3, DINO). Those run in separate, pinned environments behind
``s4d.realworld.workers``. See docs/realworld_preprocessing.md and docs/realworld_preprocessing_STATUS.md.

No real sample data existed when this was written: everything that depends on the recorded on-disk layout lives
in an adapter (``s4d.realworld.adapters``), and every numerical stage is validated on synthetic fixtures only.
"""

from s4d.realworld.contract import SCHEMA_VERSION

__all__ = ["SCHEMA_VERSION"]
