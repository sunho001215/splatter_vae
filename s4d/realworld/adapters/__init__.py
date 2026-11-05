"""Ingestion adapters: the only code that knows a recording's on-disk layout.

``EpisodeAdapter`` (base.py) is the interface; ``PandaTeleopH5Adapter`` (panda_teleop.py) reads the own rig's
recordings (Franka Panda + Robotiq 2F-85 + 3x RealSense RGB-D with the device IR pair); ``SyntheticAdapter``
(synthetic_adapter.py) is a fully synthetic fixture for tests. Select one by name with ``get_adapter``.
"""

from s4d.realworld.adapters.base import EpisodeAdapter, get_adapter, register_adapter

__all__ = ["EpisodeAdapter", "get_adapter", "register_adapter"]
