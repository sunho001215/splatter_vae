"""Looping MP4 previews of a recorded episode viewer, rendered by the same browser player (deterministic frames).

The player is paused and its playback time set frame by frame (time box of the playback bar), then screenshotted, so
the video shows exactly what the interactive viewer shows, at the episode's timing (``--fps`` samples of real time).
Viewpoints: ``--view NAME=px,py,pz:lx,ly,lz`` (camera position : look-at, robot base frame, metres); default: the
recording's initial camera.

    python scripts/realworld_viewer_video.py OUT_DIR --recording compare --fps 10 --out OUT_DIR/videos/compare.mp4
"""

from __future__ import annotations

import argparse
import io
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--recording", default="compare")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--view", help="px,py,pz:lx,ly,lz")
    ap.add_argument("--size", default="1280x720")
    ap.add_argument("--chromium")
    ap.add_argument("--video", type=Path, required=True)
    args = ap.parse_args(argv)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import imageio.v2 as imageio
    from PIL import Image
    from playwright.sync_api import sync_playwright
    from realworld_viewer import player_url

    meta = json.loads((args.out / "viewer.json").read_text())
    rec = meta["recordings"][args.recording]
    cam = dict(rec["initial_camera"])
    if args.view:
        p, lk = args.view.split(":")
        cam["position"] = [float(x) for x in p.split(",")]
        cam["look_at"] = [float(x) for x in lk.split(",")]
    url = player_url(args.recording, cam) + "&hideViserLogo"
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    srv = subprocess.Popen([sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1", "--directory",
                            str(args.out)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    W, H = (int(x) for x in args.size.split("x"))
    env = dict(os.environ, __EGL_VENDOR_LIBRARY_FILENAMES="/usr/share/glvnd/egl_vendor.d/50_mesa.json",
               EGL_PLATFORM="surfaceless")
    frames = []
    try:
        time.sleep(1.0)
        with sync_playwright() as pw:
            browser = pw.chromium.launch(executable_path=args.chromium, env=env, args=[
                "--headless=new", "--ozone-platform=headless", "--use-gl=angle", "--use-angle=gl-egl",
                "--ignore-gpu-blocklist", "--no-sandbox"])
            page = browser.new_page(viewport={"width": W, "height": H})
            page.goto(f"http://127.0.0.1:{port}/{url}")
            deadline = time.time() + 300
            while time.time() < deadline and page.query_selector("[role=slider]") is None:
                time.sleep(0.5)
            duration = float(page.query_selector("[role=slider]").get_attribute("aria-valuemax"))
            time.sleep(1.5)
            page.keyboard.press("Space")  # pause (the player's spacebar toggle)
            for note in page.query_selector_all("button[aria-label='Close'], .mantine-Notification-closeButton"):
                note.click()  # the "software WebGL" notice is a headless-only artefact
            box = page.query_selector("input.mantine-NumberInput-input") or page.query_selector("input")
            n = int(duration * args.fps)
            for i in range(n):
                t = i / args.fps
                box.fill(f"{t:.2f}")
                box.press("Enter")
                page.mouse.move(W - 5, 5)
                time.sleep(0.25)
                frames.append(np.asarray(Image.open(io.BytesIO(page.screenshot())).convert("RGB")))
            browser.close()
    finally:
        srv.terminate()
    args.video.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(args.video, frames, fps=args.fps, codec="libx264", quality=7, macro_block_size=8)
    print(json.dumps({"video": str(args.video), "frames": len(frames), "fps": args.fps, "duration_s": duration,
                      "camera": cam}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
