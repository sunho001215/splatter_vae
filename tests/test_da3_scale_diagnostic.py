import numpy as np

from scripts.diagnose_droid_da3_scale import triangulate_matches


def _scene():
    K = np.repeat(
        np.array([[[130.0, 0.0, 160.0], [0.0, 130.0, 90.0], [0.0, 0.0, 1.0]]]),
        2,
        axis=0,
    )
    w2c = np.repeat(np.eye(4)[None], 2, axis=0)
    w2c[1, 0, 3] = -0.5
    world = np.array([[0.0, 0.0, 1.0], [0.1, 0.1, 2.0], [0.2, -0.1, 3.0]])
    camera = world[None] + w2c[:, None, :3, 3]
    pixels = np.einsum("vij,vnj->vni", K, camera)
    return K, w2c, pixels[..., :2] / pixels[..., 2:], camera[..., 2]


def test_calibrated_triangulation_recovers_metric_depth():
    K, w2c, points, expected = _scene()
    actual, accepted, error = triangulate_matches(points, K, w2c)
    np.testing.assert_allclose(actual, expected, atol=1.0e-10)
    np.testing.assert_allclose(accepted, points)
    assert error.max() < 1.0e-10


def test_calibrated_triangulation_rejects_inconsistent_correspondences():
    K, w2c, points, _ = _scene()
    points[1, :, 1] += 20.0
    depth, accepted, _ = triangulate_matches(points, K, w2c)
    assert depth.shape == (2, 0)
    assert accepted.shape == (2, 0, 2)


def test_calibrated_triangulation_rejects_negative_depth():
    K, w2c, points, _ = _scene()
    points = points[::-1].copy()
    depth, _, _ = triangulate_matches(points, K, w2c)
    assert depth.shape == (2, 0)
