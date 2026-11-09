"""Latency benchmark of the policy server: sustained 15 Hz paced and max-rate runs, measured on the CLIENT monotonic clock.

    .venv/bin/python -I ops/policy_serving/bench.py --host 127.0.0.1 --port 47100 --policy dynamo-drawer-open \
        --mode paced,max --steps 300 --native-hw 480x640 --out bench.json

Per step: client preprocessing (exact training crop + resize of the native frame, ``preprocess_frame``), serialization,
network = round trip minus the server-reported duration, server prep (frame push + history assembly) / queue / encoder /
FQL / total, end-to-end = start of the step's first preprocessing -> decoded action. Payload bytes of the first step
(history fill) and of steady-state steps (only new frames). Inputs are SYNTHETIC native frames (random RGB, random
metric depth with holes), a synthetic regular 15 Hz capture clock and, where the condition needs them, a synthetic
calibration / identity native depth correction / zero M5 motion: they exercise the exact transport and compute path,
never a learned behaviour. numpy + stdlib (+ the torch-free deploy preprocessing); never talks to a robot.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from ops.policy_serving.client import PolicyClient  # noqa: E402

TIMINGS = ("e2e_ms", "preprocess_ms", "serialize_ms", "send_ms", "rtt_ms", "network_ms", "decode_ms")
SERVER = ("prep_ms", "queue_ms", "validation_ms", "encoder_ms", "fql_ms", "policy_ms", "infer_ms", "total_ms")


def synthetic_choices(native_hw=(480, 640), camera: str | None = None):
    """Session choices for a bench run (labelled synthetic calibration / identity correction where required)."""
    h, w = (int(x) for x in native_hw)

    def choose(spec: dict) -> dict:
        orders = spec["trained_camera_orders"]
        cams = list(orders[0]) if spec["views"] > 1 else [camera or orders[0][0]]
        out = {"native_hw": {c: [h, w] for c in cams}}
        if spec["views"] == 1:
            out["camera_id"] = cams[0]
        if spec["needs_calibration"]:
            f = float(max(h, w))
            out["calibration"] = {c: {"K_native": [[f, 0.0, w / 2], [0.0, f, h / 2], [0.0, 0.0, 1.0]],
                                      "c2w": np.eye(4).tolist()} for c in cams}
        if "depth" in spec["modalities"]:
            out["native_depth_correction"] = {c: {"stage": "sensor_depth", "factor": 1.0, "validated": False,
                                                  "evidence": "bench: synthetic input, no correction"} for c in cams}
        return out

    return choose


def _percentiles(values) -> dict | None:
    v = np.asarray([x for x in values if x is not None], np.float64)
    if not len(v):
        return None
    return {"n": int(len(v)), "p50": float(np.percentile(v, 50)), "p95": float(np.percentile(v, 95)),
            "p99": float(np.percentile(v, 99)), "max": float(v.max()), "mean": float(v.mean())}


def run(client: PolicyClient, *, mode: str = "paced", steps: int = 300, warmup: int = 10, hz: float | None = None,
        native_hw=(480, 640), seed: int = 0, ttl_ms: float | None = None, timeout_s: float = 1.0,
        noise_seed: int | None = None, pool: int = 8) -> dict:
    """One run on a connected client; returns the summary (+ per-step rows under ``steps``)."""
    if mode not in ("paced", "max"):
        raise ValueError("mode must be paced or max")
    spec = client.spec
    cams, depth = list(spec["camera_ids"]), "depth" in spec["modalities"]
    period_s = float(spec["frame_period_s"]) * int(spec["frame_spacing"])
    hz = hz or 1.0 / float(spec["frame_period_s"])
    h, w = (int(x) for x in native_hw)
    rng = np.random.default_rng(seed)
    frames = {}
    for c in cams:
        frames[c] = []
        for _ in range(pool):
            rgb = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
            d = rng.uniform(0.3, 1.5, (h, w)).astype(np.float32) if depth else None
            if d is not None:
                d[rng.random((h, w)) < 0.05] = 0
            frames[c].append((rgb, d, None if d is None else d > 0))
    corr = (client.choices.get("native_depth_correction") or {}) if depth else {}
    base = time.monotonic_ns()
    next_id = dict.fromkeys(cams, 0)
    rows, t_starts = [], []
    total = warmup + steps
    start = time.monotonic()
    for i in range(total):
        if mode == "paced":
            tick = start + i / hz
            delay = tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        t_starts.append(time.monotonic())
        need = int(spec["history_length"]) if i == 0 else 1
        for _ in range(need):
            for v, c in enumerate(cams):
                k = next_id[c]
                rgb, d, valid = frames[c][k % pool]
                client.add_frame_native(c, k, base + int(round(k * period_s * 1e9)) + v * 1_000_000, rgb,
                                        pose7=[0.4, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0], gripper_register=3.0, depth=d,
                                        valid=valid, native_correction=corr.get(c))
                next_id[c] = k + 1
            if spec["views"] > 1:
                client.add_sync_set({c: next_id[c] - 1 for c in cams}, 0.0)
        end = {c: next_id[c] - 1 for c in cams}
        obs_t = max(base + int(round(end[c] * period_s * 1e9)) + v * 1_000_000 for v, c in enumerate(cams))
        r = client.step(end, f"bench-{i}", [0.4, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0], 3.0, pose_time_ns=obs_t - 3_000_000,
                        gripper_time_ns=obs_t - 40_000_000, ttl_ms=ttl_ms, noise_seed=noise_seed, timeout_s=timeout_s)
        rows.append({"i": i, "warmup": i < warmup, "status": r.status, "code": r.code, "frames_sent": r.frames_sent,
                     "request_bytes": r.request_bytes, "response_bytes": r.response_bytes, "timings_ms": r.timings_ms,
                     "server_ms": r.server_ms})
    measured = [r for r in rows if not r["warmup"]]
    ok = [r for r in measured if r["status"] == "ok"]
    elapsed = t_starts[-1] - t_starts[warmup] if steps > 1 else float("nan")
    budget_ms = 1000.0 / hz
    steady = [r["request_bytes"] for r in rows[1:] if r["status"] == "ok"]
    return {
        "mode": mode, "target_hz": hz if mode == "paced" else None, "steps": steps, "warmup": warmup,
        "achieved_hz": (steps - 1) / elapsed if steps > 1 and elapsed > 0 else None,
        "ok": len(ok), "errors": dict(collections.Counter(r["code"] or r["status"] for r in measured
                                                           if r["status"] != "ok")),
        "budget_ms": budget_ms, "e2e_over_budget": sum(r["timings_ms"]["e2e_ms"] > budget_ms for r in ok),
        "client_ms": {k: _percentiles(r["timings_ms"].get(k) for r in ok) for k in TIMINGS},
        "server_ms": {k: _percentiles(r["server_ms"].get(k) for r in ok) for k in SERVER},
        "payload_bytes": {"first_request": rows[0]["request_bytes"], "first_frames": rows[0]["frames_sent"],
                          "steady_request_median": float(np.median(steady)) if steady else None,
                          "steady_frames": rows[1]["frames_sent"] if len(rows) > 1 else None,
                          "response_median": float(np.median([r["response_bytes"] for r in ok])) if ok else None},
        "inputs": "SYNTHETIC native frames / capture clock / calibration (bench only)",
        "rows": rows,
    }


def main(argv=None) -> int:
    from ops.policy_serving.config import load_config

    ap = argparse.ArgumentParser(prog="ops/policy_serving/bench.py", description=__doc__.split("\n")[0])
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--token-file", default=None)
    ap.add_argument("--camera", default=None, help="single-view camera id (default: first trained camera)")
    ap.add_argument("--mode", default="paced,max")
    ap.add_argument("--hz", type=float, default=None, help="paced rate (default: the trained frame rate)")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--native-hw", default="480x640")
    ap.add_argument("--timeout-s", type=float, default=1.0)
    ap.add_argument("--ttl-ms", type=float, default=None)
    ap.add_argument("--noise-seed", type=int, default=None, help="validation-mode FQL noise (default: server RNG)")
    ap.add_argument("--takeover", action="store_true")
    ap.add_argument("--rows", action="store_true", help="keep per-step rows in the output")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    try:
        cc = load_config(args.config)["client"]
    except (ModuleNotFoundError, FileNotFoundError):  # robot PC without PyYAML / config: CLI values + defaults
        from ops.policy_serving.config import CLIENT_DEFAULTS as cc
    native_hw = tuple(int(x) for x in args.native_hw.lower().split("x"))
    token = None
    if args.token_file:
        token = Path(args.token_file).expanduser().read_bytes().strip()
    client = PolicyClient(args.host or cc["host"], args.port or cc["port"], policy_id=args.policy,
                          choices=synthetic_choices(native_hw, args.camera), token=token,
                          connect_timeout_s=cc["connect_timeout_s"], io_timeout_s=cc["io_timeout_s"],
                          step_timeout_s=args.timeout_s, reconnect_backoff_s=cc["reconnect_backoff_s"],
                          ttl_ms=args.ttl_ms or cc["ttl_ms"], takeover=args.takeover, name="bench")
    out = {"route": f"{client.host}:{client.port}", "policy": args.policy, "native_hw": list(native_hw), "runs": []}
    try:
        client.connect()
        out["input_spec"] = {k: client.spec[k] for k in ("condition", "camera_ids", "views", "modalities",
                                                          "image_hw", "history_length", "frame_spacing",
                                                          "frame_period_s", "fixture")}
        out["ping_ms"] = [client.ping() for _ in range(20)]
        for mode in args.mode.split(","):
            client.reset()
            res = run(client, mode=mode, steps=args.steps, warmup=args.warmup, hz=args.hz, native_hw=native_hw,
                      ttl_ms=args.ttl_ms, timeout_s=args.timeout_s, noise_seed=args.noise_seed)
            if not args.rows:
                res.pop("rows")
            out["runs"].append(res)
        out["client_stats"] = dict(client.stats)
    finally:
        client.close()
    text = json.dumps(out, indent=1, allow_nan=False)
    if args.out:
        args.out.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
