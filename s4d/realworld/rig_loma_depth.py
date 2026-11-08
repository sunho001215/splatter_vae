"""Depth by stage NAME for ``rig_loma``: samples a depth stage at the static RoMa match ends and at the robot 2D-3D
correspondence pixels, so that every depth source (FoundationStereo, native sensor, S2M2, ...) can drive the same
solver independently. A stage file is an HDF5 with datasets ``depth`` (n_frames, H, W) metres and ``valid`` (bool;
optional, depth > 0 otherwise), on the RGB grid of the camera (the layout of the processed depth stages and of the
S2M2 stream's ``s2m2_depth``), located by ``--template`` (fields: root, ep, stage, key).

Per match end / correspondence: nearest-pixel depth (the convention of ``romav2_match_worker.sample_at``), validity
and the local 5x5 depth range (``local_range``) -> ``<out>/<episode>.npz`` with d_a, d_b, valid_a, valid_b, range_a,
range_b (ext-ext) or d, valid, range (correspondences), row-aligned with the input file."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DEFAULT_TEMPLATE = "{root}/{ep}/episodes/{ep}/derived/{stage}/{key}.h5"


def local_range(depth: np.ndarray, r: int = 2) -> np.ndarray:
    from scipy.ndimage import maximum_filter, minimum_filter

    big = np.where(depth > 0, depth, np.inf)
    small = np.where(depth > 0, depth, -np.inf)
    rng = maximum_filter(small, size=2 * r + 1) - minimum_filter(big, size=2 * r + 1)
    return np.where(np.isfinite(rng), rng, np.inf).astype(np.float32)


def _sample(img, uv):
    h, w = img.shape[:2]
    j = np.clip(np.floor(uv[:, 0]).astype(int), 0, w - 1)
    i = np.clip(np.floor(uv[:, 1]).astype(int), 0, h - 1)
    return img[i, j]


class StageReader:
    def __init__(self, template: str, root: str, ep: str, stage: str):
        self.template, self.root, self.ep, self.stage = template, root, ep, stage
        self.cache = {}
        self.missing = 0  # frames not computed by the stage (subset stages, e.g. s2m2_depth)

    def frame(self, cam: str, idx: int):
        import h5py

        key = (cam, int(idx))
        if key not in self.cache:
            p = self.template.format(root=self.root, ep=self.ep, stage=self.stage, key=cam.replace("/", "__"))
            with h5py.File(p, "r") as f:
                d = f["depth"][int(idx)].astype(np.float32)
                v = f["valid"][int(idx)].astype(bool) if "valid" in f else d > 0
                if "computed" in f and not bool(f["computed"][int(idx)]):  # stage computed on a frame subset only
                    v = np.zeros_like(v)
                    self.missing += 1
            d = np.where(v & np.isfinite(d), d, 0.0).astype(np.float32)
            self.cache[key] = (d, local_range(d))
        return self.cache[key]


def sample_matches(match_path: Path, reader: StageReader) -> dict:
    with np.load(match_path) as z:
        meta = json.loads(str(z["meta"]))
        m = {k: z[k] for k in ("pair", "frame_a", "frame_b", "uv_a", "uv_b")}
    cams = meta["cameras"]
    n = len(m["pair"])
    out = {k: np.zeros(n, np.float32) for k in ("d_a", "d_b", "range_a", "range_b")}
    for p in np.unique(m["pair"]):
        pa, pb = divmod(int(p), 10)
        for end, ci, fr in (("a", pa, m["frame_a"]), ("b", pb, m["frame_b"])):
            sel = np.flatnonzero(m["pair"] == p)
            for f in np.unique(fr[sel]):
                s = sel[fr[sel] == f]
                d, rg = reader.frame(cams[ci], int(f))
                out[f"d_{end}"][s] = _sample(d, m[f"uv_{end}"][s])
                out[f"range_{end}"][s] = _sample(rg, m[f"uv_{end}"][s])
    out["valid_a"], out["valid_b"] = out["d_a"] > 0, out["d_b"] > 0
    return out


def sample_corr(corr_path: Path, reader: StageReader) -> dict:
    with np.load(corr_path) as z:
        meta = json.loads(str(z["meta"]))
        c = {k: z[k] for k in ("cam", "t_index", "uv_obs")} if len(z["cam"]) else None
    if c is None:
        return {"d": np.zeros(0, np.float32), "range": np.zeros(0, np.float32), "valid": np.zeros(0, bool)}
    n = len(c["cam"])
    d_out, r_out = np.zeros(n, np.float32), np.zeros(n, np.float32)
    for ci in np.unique(c["cam"]):
        for t in np.unique(c["t_index"][c["cam"] == ci]):
            s = np.flatnonzero((c["cam"] == ci) & (c["t_index"] == t))
            d, rg = reader.frame(meta["cameras"][int(ci)], int(t))
            d_out[s] = _sample(d, c["uv_obs"][s])
            r_out[s] = _sample(rg, c["uv_obs"][s])
    return {"d": d_out, "range": r_out, "valid": d_out > 0}


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--root", required=True, help="task root of the stage files (template field {root})")
    ap.add_argument("--template", default=DEFAULT_TEMPLATE)
    ap.add_argument("--matches", default=None)
    ap.add_argument("--corr", default=None)
    ap.add_argument("--out-ee", default=None)
    ap.add_argument("--out-corr", default=None)
    ap.add_argument("--episodes", nargs="*", default=None)
    a = ap.parse_args(argv)
    for src, dst, fn in ((a.matches, a.out_ee, sample_matches), (a.corr, a.out_corr, sample_corr)):
        if not src:
            continue
        files = sorted(Path(src).glob("*.npz"))
        if a.episodes:
            files = [f for f in files if f.stem in set(a.episodes)]
        for f in files:
            out = Path(dst) / f.name
            if out.exists():
                continue
            res = fn(f, StageReader(a.template, a.root, f.stem, a.stage))
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, stage=a.stage, template=a.template, **res)
            tmp.replace(out)
            print(json.dumps({"file": f.name, "n": int(len(res.get("d", res.get("d_a")))),
                              "valid": float(np.mean(res.get("valid", res.get("valid_a"))))}), flush=True)


if __name__ == "__main__":
    main()
