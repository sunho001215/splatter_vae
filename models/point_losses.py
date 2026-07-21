"""Compatibility exports for validation-only point-cloud utilities."""

from models.pointcloud_utils import depths_to_world_point_cloud, sample_points

__all__ = ["depths_to_world_point_cloud", "sample_points"]
