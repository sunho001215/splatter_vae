"""Ingestion adapters: the only code that knows a recording's on-disk layout.

``EpisodeAdapter`` (base.py) is the interface; ``TemplateRealSenseAdapter`` (template.py) documents what a real
Franka + Robotiq + 3x RealSense adapter must provide once sample data exists; ``SyntheticAdapter``
(synthetic_adapter.py) is a fully synthetic fixture for tests. Select one by name with ``get_adapter``.
"""

from s4d.realworld.adapters.base import EpisodeAdapter, get_adapter, register_adapter

__all__ = ["EpisodeAdapter", "get_adapter", "register_adapter"]
