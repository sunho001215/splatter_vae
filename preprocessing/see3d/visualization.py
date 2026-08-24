from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def colorize_scalar(
    value: np.ndarray, *, minimum: float | None = None, maximum: float | None = None
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    finite = np.isfinite(array)
    if minimum is None:
        minimum = float(np.percentile(array[finite], 2.0)) if finite.any() else 0.0
    if maximum is None:
        maximum = float(np.percentile(array[finite], 98.0)) if finite.any() else 1.0
    normalized = np.clip((array - minimum) / max(maximum - minimum, 1.0e-6), 0.0, 1.0)
    # Compact blue-cyan-yellow-red diagnostic palette.
    red = np.clip(1.5 * normalized - 0.25, 0.0, 1.0)
    green = np.clip(1.5 - np.abs(2.0 * normalized - 1.0) * 1.5, 0.0, 1.0)
    blue = np.clip(1.25 - 1.5 * normalized, 0.0, 1.0)
    color = np.stack((red, green, blue), axis=-1)
    color[~finite] = 0.0
    return (color * 255.0).round().astype(np.uint8)


def save_see3d_validation_grid(
    path: str | Path,
    panels: Sequence[tuple[str, np.ndarray]],
    *,
    columns: int = 5,
) -> None:
    if not panels:
        raise ValueError("At least one See3D panel is required.")
    width, height = 320, 180
    label_height = 24
    rows = (len(panels) + columns - 1) // columns
    canvas = Image.new(
        "RGB", (columns * width, rows * (height + label_height)), "white"
    )
    draw = ImageDraw.Draw(canvas)
    for index, (label, array) in enumerate(panels):
        value = np.asarray(array)
        if value.ndim == 2:
            value = np.repeat(value[..., None], 3, axis=-1)
        image = Image.fromarray(value.astype(np.uint8)).resize(
            (width, height), Image.Resampling.BILINEAR
        )
        x = (index % columns) * width
        y = (index // columns) * (height + label_height)
        canvas.paste(image, (x, y + label_height))
        draw.text((x + 5, y + 5), label, fill="black")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)
