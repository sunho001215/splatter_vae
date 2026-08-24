from __future__ import annotations

import numpy as np

from preprocessing.workspace import (
    backproject_z_depth_to_world,
    propose_gaussian_workspace_parameters,
)


def test_backproject_xlens_z_depth_into_robot_base() -> None:
    depth = np.ones((2, 2), np.float32) * 2.0
    K = np.array([[2.0, 0.0, 0.5], [0.0, 2.0, 0.5], [0.0, 0.0, 1.0]])
    c2w = np.eye(4)
    c2w[:3, 3] = (1.0, 2.0, 3.0)
    points = backproject_z_depth_to_world(depth, K, c2w)
    np.testing.assert_allclose(points[0, 0], (1.0, 2.0, 5.0))
    np.testing.assert_allclose(points[0, 1], (2.0, 2.0, 5.0))


def test_workspace_proposal_uses_robust_statistics() -> None:
    rng = np.random.default_rng(3)
    points = rng.normal((0.5, 0.0, 0.4), (0.3, 0.2, 0.2), size=(10_000, 3))
    points = np.concatenate((points, [[1000.0, -1000.0, 1000.0]]))
    depths = rng.uniform(0.2, 2.0, size=10_000)
    proposal = propose_gaussian_workspace_parameters(points, depths)
    assert np.linalg.norm(np.asarray(proposal.global_center) - (0.5, 0.0, 0.4)) < 0.05
    assert proposal.zfar < 3.0
    assert proposal.child_radius < proposal.anchor_initial_spread
