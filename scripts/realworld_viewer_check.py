"""Verify recorded viewer playback in a real (headless) Chromium: page load, recording load, rendering, real-time
playback, looping, camera interaction that is not reset by playback, and the index page.

    python scripts/realworld_viewer_check.py OUT_DIR [--recording compare] [--chromium PATH]

Needs ``playwright`` and a Chromium with WebGL (``--use-angle=gl-egl`` on Mesa's surfaceless EGL works without a
display). Serves ``OUT_DIR`` on a local port, drives the player and writes ``OUT_DIR/verification/<recording>.json``
plus screenshots. Exit code 1 if any check fails.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def shot(page, path: Path) -> np.ndarray:
    from PIL import Image

    page.screenshot(path=str(path))
    return np.asarray(Image.open(path).convert("RGB"), np.float32)


def slider(page):
    el = page.query_selector("[role=slider]")
    if el is None:
        return None, None
    return float(el.get_attribute("aria-valuenow")), float(el.get_attribute("aria-valuemax"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--recording", default="compare")
    ap.add_argument("--chromium", default=None)
    ap.add_argument("--viewport", default="1400x900")
    args = ap.parse_args(argv)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from playwright.sync_api import sync_playwright
    from realworld_viewer import player_url

    meta = json.loads((args.out / "viewer.json").read_text())
    rec = meta["recordings"][args.recording]
    vdir = args.out / "verification"
    vdir.mkdir(exist_ok=True)
    port = free_port()
    srv = subprocess.Popen([sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1", "--directory",
                            str(args.out)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    W, H = (int(x) for x in args.viewport.split("x"))
    report = {"recording": args.recording, "url": player_url(args.recording, rec["initial_camera"]), "checks": {}}
    env = dict(os.environ, __EGL_VENDOR_LIBRARY_FILENAMES="/usr/share/glvnd/egl_vendor.d/50_mesa.json",
               EGL_PLATFORM="surfaceless")
    try:
        time.sleep(1.0)
        with sync_playwright() as p:
            browser = p.chromium.launch(executable_path=args.chromium, env=env, args=[
                "--headless=new", "--ozone-platform=headless", "--use-gl=angle", "--use-angle=gl-egl",
                "--ignore-gpu-blocklist", "--no-sandbox"])
            page = browser.new_page(viewport={"width": W, "height": H})
            errors = []
            page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
            page.on("pageerror", lambda e: errors.append(str(e)))
            t0 = time.time()
            page.goto(f"http://127.0.0.1:{port}/{report['url']}")
            deadline = time.time() + 300
            while time.time() < deadline and slider(page)[0] is None:
                time.sleep(0.5)
            now, duration = slider(page)
            report["checks"]["loaded"] = {"ok": now is not None, "seconds_to_playback": round(time.time() - t0, 1),
                                          "duration_s": duration}
            if now is None:
                raise RuntimeError("playback bar never appeared")
            webgl = page.evaluate("() => !!document.querySelector('canvas') && "
                                  "!!document.querySelector('canvas').getContext('webgl2')")
            time.sleep(2.0)
            a = shot(page, vdir / f"{args.recording}_t_a.png")
            # real-time playback + loop: sample the playback position for (duration + 4) s
            samples = []
            t_start = time.time()
            while time.time() - t_start < (duration or 10) + 4:
                samples.append((time.time() - t_start, slider(page)[0]))
                time.sleep(0.25)
            b = shot(page, vdir / f"{args.recording}_t_b.png")
            vals = np.array([s[1] for s in samples])
            wraps = int((np.diff(vals) < -0.5 * (duration or 1)).sum())
            seg = [i for i in range(1, len(samples)) if samples[i][1] >= samples[i - 1][1]]
            rate = None
            if seg:
                dt_wall = sum(samples[i][0] - samples[i - 1][0] for i in seg)
                dt_play = sum(samples[i][1] - samples[i - 1][1] for i in seg)
                rate = dt_play / dt_wall if dt_wall else None
            report["checks"]["playback"] = {"ok": wraps >= 1 and rate is not None and 0.8 < rate < 1.2,
                                            "loops_observed": wraps, "playback_s_per_wall_s": rate,
                                            "first_last": [samples[0], samples[-1]]}
            canvas_frac = float((np.abs(a - a[0, 0]).sum(-1) > 30).mean())
            report["checks"]["rendering"] = {"ok": bool(webgl) and canvas_frac > 0.05, "webgl2": bool(webgl),
                                             "non_background_fraction": canvas_frac,
                                             "animation_mean_abs_diff": float(np.abs(a - b).mean())}
            # camera interaction: orbit drag + zoom, then let playback run: the view must not snap back
            pre = shot(page, vdir / f"{args.recording}_pre_drag.png")
            cx, cy = W // 2, H // 2
            page.mouse.move(cx, cy)
            page.mouse.down()
            for i in range(1, 21):
                page.mouse.move(cx + 12 * i, cy + 3 * i)
                time.sleep(0.02)
            page.mouse.up()
            page.mouse.wheel(0, -400)
            time.sleep(1.0)
            post0 = shot(page, vdir / f"{args.recording}_post_drag_0s.png")
            time.sleep(5.0)
            post5 = shot(page, vdir / f"{args.recording}_post_drag_5s.png")
            d_move = float(np.abs(post0 - pre).mean())
            d_keep = float(np.abs(post5 - post0).mean())
            report["checks"]["camera_interaction"] = {
                "ok": d_move > 2.0 and d_keep < 0.7 * d_move, "pre_vs_after_drag": d_move,
                "after_drag_vs_5s_later": d_keep,
                "note": "after-drag images differ only by the animation (view kept); before/after differ by the orbit"}
            page.goto(f"http://127.0.0.1:{port}/index.html")
            time.sleep(3.0)
            frames = page.frames
            report["checks"]["index_page"] = {"ok": len(frames) >= 2 and page.title() != "",
                                              "title": page.title(), "frames": len(frames)}
            shot(page, vdir / "index_page.png")
            report["console_errors"] = errors[:20]
            browser.close()
    finally:
        srv.terminate()
    report["ok"] = all(c["ok"] for c in report["checks"].values())
    (vdir / f"{args.recording}.json").write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps(report, indent=1, default=float))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
