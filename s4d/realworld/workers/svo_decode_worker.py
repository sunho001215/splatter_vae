"""ZED SVO decoder worker (runs in an isolated environment with PyAV; no ZED SDK needed).

task ``decode_svo``: params ``svo_path``; output ``frames`` (N, H, 2W, 3) uint8 RGB side-by-side left|right raw
(unrectified) images exactly as encoded, and ``timestamps_ns`` (N,) from the SVO frame records. Frames are decoded in
record order with one HEVC decoder; the frame count must equal the record count.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "adapters"))

from _protocol import main, sha256  # noqa: E402


def handle(request, inputs):
    import av
    import numpy as np
    from droid_svo import index_svo  # standard-library parser shared with the adapter

    path = request["params"]["svo_path"]
    index = index_svo(path)
    data = Path(path).read_bytes()
    codec = av.CodecContext.create("hevc", "r")
    frames = []
    for rec in index.records:
        for frame in codec.decode(av.Packet(data[rec.offset:rec.offset + rec.size])):
            frames.append(frame.to_ndarray(format="rgb24"))
    for frame in codec.decode(None):
        frames.append(frame.to_ndarray(format="rgb24"))
    if len(frames) != len(index.records):
        raise RuntimeError(f"decoded {len(frames)} frames from {len(index.records)} records")
    out = np.stack(frames)
    if out.shape[1:] != (index.height, 2 * index.width, 3):
        raise RuntimeError(f"decoded shape {out.shape[1:]} != ({index.height}, {2 * index.width}, 3)")
    identity = {"name": "PyAV HEVC decoder", "repo": "PyAV-Org/PyAV", "commit": f"av {av.__version__}",
                "checkpoint": None, "checkpoint_sha256": None, "is_mock": False,
                "libavcodec": ".".join(map(str, av.library_versions.get("libavcodec", ())))}
    meta = {"svo_sha256": sha256(path), "serial": index.serial, "records": len(index.records),
            "key_frames": sum(r.key for r in index.records), "eye_width": index.width, "height": index.height}
    return identity, {"frames": out, "timestamps_ns": np.array([r.timestamp_ns for r in index.records], np.int64)}, meta


if __name__ == "__main__":
    main(handle)
