"""Worker-env tests of the board detectors (OpenCV with aruco is a worker-image dependency, absent from the release-gate
image): ChArUco detection on a rendered board and Zivid CB01 detection/ordering on synthetic rasters. Not collected by
the default suite (outside ``testpaths``); run in the s4d-rwprep worker image, e.g.
    python -m pytest -q -p no:cacheprovider tests_worker_env
Every test hard-requires cv2.aruco: nothing is skipped."""

import numpy as np
import pytest

from s4d.realworld.calibration_board import BoardSpec

SPEC = BoardSpec(squares_x=8, squares_y=6, square_m=0.05, marker_m=0.037)


def test_detect_charuco_on_a_rendered_board():
    import cv2

    from s4d.realworld.calibration_board import detect_charuco

    aruco = cv2.aruco
    board = aruco.CharucoBoard((SPEC.squares_x, SPEC.squares_y), SPEC.square_m, SPEC.marker_m,
                               aruco.getPredefinedDictionary(getattr(aruco, SPEC.dictionary)))
    img = board.generateImage((800, 600), marginSize=40)
    ids, uv = detect_charuco(img, SPEC)
    assert sorted(ids) == list(range((SPEC.squares_x - 1) * (SPEC.squares_y - 1)))
    # geometry: the first id row is evenly spaced along x (row-major id order), at the generated square size
    order = np.argsort(ids)
    d = np.diff(uv[order][: SPEC.squares_x - 1, 0])
    px = min((800 - 80) / SPEC.squares_x, (600 - 80) / SPEC.squares_y)
    assert np.allclose(np.abs(d), px, atol=1.0) and np.allclose(np.diff(uv[order][: SPEC.squares_x - 1, 1]), 0, atol=1.0)


def _render_zivid(*, rotation=0, marker_id=1, marker=True):
    """Raster synthetic board with known physical corner ids, including an oblique projective view."""
    import cv2

    pitch, left, top = 50, 100, 100
    image = np.full((650, 700), 155, np.uint8)
    image[60:560, 60:620] = 40
    for row in range(8):
        for col in range(7):
            image[top + row * pitch:top + (row + 1) * pitch,
                  left + col * pitch:left + (col + 1) * pitch] = 240 if (row + col) % 2 == 0 else 10
    if marker:
        cx, cy = left + 8.5 * pitch, top + 7.17 * pitch
        x, y = int(round(cx - 30)), int(round(cy - 30))
        image[y - 8:y + 68, x - 8:x + 68] = 255
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        image[y:y + 60, x:x + 60] = cv2.aruco.generateImageMarker(dictionary, marker_id, 60)
    # Coordinates represent pixel boundaries (continuous convention); raster corner estimator gives -0.5 in OpenCV.
    truth = BoardSpec.zivid_cb01().object_points()[:, :2] / 0.03 * pitch + [left, top]
    src = np.float32([[0, 0], [699, 0], [699, 649], [0, 649]])
    dst = np.float32([[35, 50], [650, 10], [680, 600], [10, 630]])
    H = cv2.getPerspectiveTransform(src, dst)
    image = cv2.warpPerspective(image, H, (700, 650), borderValue=155)
    truth = cv2.perspectiveTransform((truth - 0.5).reshape(-1, 1, 2), H).reshape(-1, 2) + 0.5
    if rotation == 2:
        image = np.rot90(image, 2).copy()
        truth = np.array([700, 650]) - truth
    return image, truth


@pytest.mark.parametrize("rotation", [0, 2])
def test_detect_zivid_rendered_both_orientations(rotation):
    from s4d.realworld.calibration_board import detect_board

    image, truth = _render_zivid(rotation=rotation)
    ids, uv = detect_board(image, BoardSpec.zivid_cb01())
    assert np.array_equal(ids, np.arange(42))
    assert np.max(np.linalg.norm(uv - truth, axis=1)) < 0.4
    # RGB dispatch has exactly the same geometry and pixel convention.
    ids_rgb, uv_rgb = detect_board(np.repeat(image[..., None], 3, -1), BoardSpec.zivid_cb01())
    assert np.array_equal(ids_rgb, ids) and np.allclose(uv_rgb, uv)


@pytest.mark.parametrize("marker,marker_id", [(False, 1), (True, 2)])
def test_zivid_missing_or_wrong_marker_does_not_assign_corner_ids(marker, marker_id):
    from s4d.realworld.calibration_board import detect_board

    image, _ = _render_zivid(marker=marker, marker_id=marker_id)
    ids, uv = detect_board(image, BoardSpec.zivid_cb01())
    assert ids.shape == (0,) and uv.shape == (0, 2)


def test_zivid_orders_all_checker_row_column_directions_and_rejects_misplaced_marker():
    from s4d.realworld.calibration_board import _zivid_corner_order

    grid = np.stack(np.meshgrid(np.arange(6), np.arange(7)), -1).astype(float)
    for candidate in (grid, grid[::-1], grid[:, ::-1], grid[::-1, ::-1]):
        ordered = _zivid_corner_order(candidate.reshape(-1, 2), np.array([7.5, 6.17]))
        assert np.allclose(ordered, grid.reshape(-1, 2))
    assert _zivid_corner_order(grid.reshape(-1, 2), np.array([2.5, 3.0])) is None
