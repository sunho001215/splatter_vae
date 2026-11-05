"""PointWorld-style interactive viewer of one MEASURED/preprocessed real-world episode (not a model prediction).

Shows, in the robot-base frame: per-frame RGB-coloured point clouds of the three cameras (raw RealSense depth and/or
MapAnything depth), the Panda + Robotiq visual meshes animated from the recorded joints and gripper state (URDF forward
kinematics; no MuJoCo), camera frustums with the current frame, and the tracks the ``track_motion`` stage classified
as moving, as trails (lines = display interpolation between measured samples, dots = measured samples; trails break at
invalid samples, motion-segment breaks and the recording pause; cameras and query windows are never joined).
Display-only downsampling; the processed outputs are only read. Runs on CPU, no GPU and no model.

Modes (run with the viewer environment that has ``viser``; the repository is put on ``sys.path``):

    # portable recordings (+ the viser static client and an index page); serve the directory with any static server
    python scripts/realworld_viewer.py record --ma-root PROCESSED/v2-panda --raw-root PROCESSED/v2-panda-raw \\
        --episode mug-hanging-test-000000 --out OUT_DIR
    python -m http.server 8090 --directory OUT_DIR   # then open http://localhost:8090/
    # live viewer with every control (playback, timeline, speed, cameras, robot opacity, point size, trail length/density)
    python scripts/realworld_viewer.py serve --ma-root ... --raw-root ... --episode ... --port 8080

In a recording every node is created once and only its points/colours/pose change per frame, so the playback
scene-tree panel's visibility toggles and property edits (point size, robot opacity) persist while it loops.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from s4d.realworld import viewer_data as vd  # noqa: E402

CAM_COLOURS = np.array([[230, 75, 50], [40, 120, 220], [30, 160, 90]], np.float32)  # per camera (frustums)
ROBOT_FLOW = np.array([255, 0, 255], np.float32)  # PointWorld's robot-flow magenta
MIN_BRIGHTNESS = 0.35  # PointWorld-style brightness ramp: older trail segments darker
ARM_COLOUR, GRIPPER_COLOUR, COUPLING_COLOUR = (236, 238, 242), (70, 72, 78), (35, 35, 38)


def wxyz_pos(T: np.ndarray):
    from scipy.spatial.transform import Rotation

    x, y, z, w = Rotation.from_matrix(T[:3, :3]).as_quat()
    return np.array([w, x, y, z]), T[:3, 3].copy()


def load_link_meshes(manifest: dict, max_faces: int | None = 4000) -> dict:
    """{link: (vertices, faces, colour)} from the descriptions' VISUAL meshes (one merged mesh per link)."""
    import trimesh

    pk = manifest["packages"]
    arm = Path(manifest["inputs"]["arm"]["path"])
    grip = Path(manifest["inputs"]["gripper"]["path"])
    visuals = {**vd.urdf_visuals(arm, pk), **vd.urdf_visuals(grip, pk)}
    cg = manifest.get("coupling_geometry")
    if cg:  # mesh in the coupling LINK frame: the composed URDF's fixed joint (cg xyz/rpy) is applied by FK only
        path = cg["visual_mesh"]
        for name, directory in pk.items():
            path = path.replace(f"package://{name}/", f"{directory}/")
        visuals[cg["link"]] = [(path, np.eye(4), np.ones(3))]
    out = {}
    for link, items in visuals.items():
        vs, fs, off = [], [], 0
        for path, T, scale in items:
            m = trimesh.load(path, force="mesh")
            v = (np.asarray(m.vertices) * scale) @ T[:3, :3].T + T[:3, 3]
            vs.append(v)
            fs.append(np.asarray(m.faces) + off)
            off += len(v)
        colour = (COUPLING_COLOUR if link == (cg or {}).get("link") else
                  GRIPPER_COLOUR if link.startswith("robotiq") else ARM_COLOUR)
        V, F = np.concatenate(vs), np.concatenate(fs)
        if max_faces and len(F) > max_faces:  # display-only decimation (shape kept, triangle count bounded)
            m = trimesh.Trimesh(V, F, process=True).simplify_quadric_decimation(face_count=max_faces)
            V, F = np.asarray(m.vertices), np.asarray(m.faces)
        out[link] = (V.astype(np.float32), F.astype(np.uint32), colour)
    return out


class EpisodeScene:
    """Builds/updates the viser scene for one or more named sources (side by side, synchronized). A source is
    {"label", "root", "depth_stage"}: the clouds use that root's depth stage, the trails that root's moving tracks."""

    def __init__(self, server, ve, specs: dict, sources: list, *, offset_m: float = 1.6, stride: int = 6,
                 roi_stride: int = 3, roi_box=None, crop_box=None, trail_length: int = 8, max_tracks: int = 400,
                 dense_tracks: int = 1500, point_size: float = 0.006, robot_opacity: float = 0.45,
                 frustum_images: bool = True, cameras: list | None = None):
        from s4d.realworld.pipeline import RobotGeometry

        self.server, self.ve, self.sources = server, ve, sources
        self.specs = {s: specs[s] for s in sources}
        self.roots = {s: Path(specs[s]["root"]) for s in sources}
        self.cams = list(range(len(ve.streams))) if cameras is None else list(cameras)
        roots = self.roots
        self.kw = dict(stride=stride, roi_stride=roi_stride, roi_box=roi_box, crop_box=crop_box)
        self.trail_length, self.max_tracks, self.dense_tracks = trail_length, max_tracks, dense_tracks
        self.offsets = {s: np.array([0.0, -offset_m * i, 0.0]) for i, s in enumerate(sources)}
        self.geom = RobotGeometry(ve.meta)
        self.q, self.g, self.q_ok = vd.robot_state_at_sets(ve)
        manifest = json.loads(Path(ve.meta.robot.urdf_path).with_suffix(".manifest.json").read_text())
        self.meshes = load_link_meshes(manifest)
        self.tracks = {s: {v: vd.load_camera_tracks(roots[s], ve.episode_id, ve.streams[v]) for v in self.cams}
                       for s in sources}
        self.break_frames = [{int(ve.sets["index"][k, v]) for k, _ in ve.gaps} for v in range(len(ve.streams))]
        self.segment = vd.segment_of_set(ve)
        sc = server.scene
        sc.set_up_direction("+z")
        self.h = {}
        for s in sources:
            o = self.offsets[s]
            sc.add_frame(f"/{s}", show_axes=False, position=o)
            sc.add_label(f"/{s}/title", self.specs[s]["label"], position=(0.45, 0.0, 0.95), font_screen_scale=1.2)
            for v in self.cams:
                cam = ve.streams[v]
                self.h[(s, "cloud", v)] = sc.add_point_cloud(f"/{s}/clouds/cam{v}_{cam.split('/')[0]}",
                                                             np.zeros((1, 3)), np.zeros((1, 3), np.uint8),
                                                             point_size=point_size, point_shape="rounded")
                for kind, visible in (("trails", True), ("trails_dense", False)):
                    self.h[(s, kind, v)] = sc.add_line_segments(f"/{s}/{kind}/cam{v}", np.zeros((1, 2, 3)),
                                                                np.zeros((1, 2, 3), np.uint8), thickness=2.0,
                                                                thickness_units="screen", visible=visible)
                    self.h[(s, kind + "_samples", v)] = sc.add_point_cloud(
                        f"/{s}/{kind}_measured_samples/cam{v}", np.zeros((1, 3)), np.zeros((1, 3), np.uint8),
                        point_size=0.0035, point_shape="circle", visible=visible)
            self.h[(s, "robot_flow")] = sc.add_line_segments(f"/{s}/robot_flow_gripper", np.zeros((1, 2, 3)),
                                                             np.zeros((1, 2, 3), np.uint8), thickness=2.0,
                                                             thickness_units="screen")
            for variant, opacity, visible in (("robot_translucent", robot_opacity, True), ("robot_opaque", None, False)):
                for link, (V, F, colour) in self.meshes.items():
                    fr = sc.add_frame(f"/{s}/{variant}/{link}", show_axes=False, visible=visible)
                    self.h[(s, variant, link, "mesh")] = sc.add_mesh_simple(
                        f"/{s}/{variant}/{link}/mesh", V, F, color=colour, opacity=opacity, flat_shading=False,
                        cast_shadow=False, receive_shadow=False)
                    self.h[(s, variant, link)] = fr
            for v in self.cams:
                cam = ve.streams[v]
                K = ve.meta.stream(cam).intrinsics
                fov = 2 * np.arctan(K.height / 2 / K.fy)
                wxyz, pos = wxyz_pos(np.linalg.inv(ve.T_cam_base[v]))
                self.h[(s, "frustum", v)] = sc.add_camera_frustum(
                    f"/{s}/cameras/cam{v}_{cam.split('/')[0]}", fov=float(fov), aspect=K.width / K.height, scale=0.05,
                    thickness=1.5, thickness_units="screen", color=tuple(int(c) for c in CAM_COLOURS[v]), wxyz=wxyz,
                    position=pos, image=np.zeros((60, 80, 3), np.uint8) if frustum_images else None)
                if frustum_images:  # depth inspection layer (same frustum, turbo 0.3-2.0 m, invalid black), hidden
                    self.h[(s, "frustum_depth", v)] = sc.add_camera_frustum(
                        f"/{s}/cameras_depth/cam{v}_{cam.split('/')[0]}", fov=float(fov), aspect=K.width / K.height,
                        scale=0.05, thickness=1.5, thickness_units="screen", color=tuple(int(c) for c in CAM_COLOURS[v]),
                        wxyz=wxyz, position=pos, image=np.zeros((60, 80, 3), np.uint8), visible=False)
        self.info = sc.add_label("/info", "", position=(0.45, 0.55, 0.9), font_screen_scale=1.0)
        self.break_sets = {k for k, _ in ve.gaps}

    def _trail_arrays(self, s, v, frame, dense: bool):
        segs, age, pts, page, _ = vd.trail_segments(self.tracks[s][v], frame, length=self.trail_length,
                                                    max_tracks=self.dense_tracks if dense else self.max_tracks,
                                                    break_frames=self.break_frames[v])
        if not len(segs):
            return np.zeros((1, 2, 3), np.float32), np.zeros((1, 2, 3), np.uint8), np.zeros((1, 3), np.float32), \
                np.zeros((1, 3), np.uint8)
        cols = np.stack([self._time_colour(age + 1), self._time_colour(age)], 1)
        return segs, cols, pts, self._time_colour(page)

    def _time_colour(self, age, base=None):
        """turbo over the trail window (oldest blue .. newest red) times a brightness ramp, as PointWorld's flows;
        ``base`` instead scales one fixed colour (robot flow)."""
        u = 1.0 - np.clip(np.asarray(age, float) / max(1, self.trail_length), 0, 1)
        bright = (MIN_BRIGHTNESS + (1 - MIN_BRIGHTNESS) * u)[:, None]
        rgb = vd.turbo(u) * 255.0 if base is None else np.broadcast_to(base, (len(u), 3))
        return np.clip(rgb * bright, 0, 255).astype(np.uint8)

    def show(self, k: int) -> None:
        ve = self.ve
        with self.server.atomic():
            ok = bool(self.q_ok[k])
            fk = vd.link_poses(self.geom, self.q[k], self.g[k]) if ok else None
            for s in self.sources:
                src, root = self.specs[s]["depth_stage"], self.roots[s]
                for v in self.cams:
                    frame = int(ve.sets["index"][k, v])
                    pts, rgb = vd.cloud(ve, v, frame, src, root=root, **self.kw)
                    h = self.h[(s, "cloud", v)]
                    h.points, h.colors = (pts, rgb) if len(pts) else (np.zeros((1, 3)), np.zeros((1, 3), np.uint8))
                    for kind, dense in (("trails", False), ("trails_dense", True)):
                        segs, cols, sp, scol = self._trail_arrays(s, v, frame, dense)
                        self.h[(s, kind, v)].points, self.h[(s, kind, v)].colors = segs, cols
                        hs = self.h[(s, kind + "_samples", v)]
                        hs.points, hs.colors = sp, scol
                    fr = self.h[(s, "frustum", v)]
                    if fr.image is not None:
                        rgb_full, dmap, dval, _ = vd.view_rgbd(ve, v, frame, src, root)
                        fr.image = rgb_full[::8, ::8].copy()
                        dimg = (vd.turbo((dmap[::8, ::8] - 0.3) / 1.7) * 255).astype(np.uint8)
                        dimg[~dval[::8, ::8]] = 0
                        self.h[(s, "frustum_depth", v)].image = dimg
                for variant in ("robot_translucent", "robot_opaque"):
                    for link in self.meshes:
                        h = self.h[(s, variant, link)]
                        if fk is not None:
                            h.wxyz, h.position = wxyz_pos(fk[link])
            segs, age = vd.gripper_flow(self.geom, self.q, self.g, self.q_ok, k, length=self.trail_length,
                                        break_sets=self.break_sets)
            if len(segs):
                cols = np.stack([self._time_colour(age + 1, ROBOT_FLOW), self._time_colour(age, ROBOT_FLOW)], 1)
            else:
                segs, cols = np.zeros((1, 2, 3), np.float32), np.zeros((1, 2, 3), np.uint8)
            for s in self.sources:
                self.h[(s, "robot_flow")].points, self.h[(s, "robot_flow")].colors = segs, cols
            seg = int(self.segment[k])
            paused = any(kk == k + 1 for kk, _ in ve.gaps)
            self.info.text = (f"t = {ve.times_s[k]:6.2f} s   set {k + 1}/{ve.n_sets}   recording segment {seg}"
                              + ("   (recording paused next)" if paused else "")
                              + ("" if ok else "   robot state unavailable"))


def default_boxes(ve, q, g, ok, geom):
    """Base-frame crop (workspace) and region of interest (the mug/hook interaction) from the robot's own motion: the
    flange path box with a margin, the ROI around the gripper while it is closed (holding the mug)."""
    flange = np.stack([geom.model.fk(geom.q(q[k], g[k]))[ve.meta.robot.flange_frame][:3, 3] for k in np.flatnonzero(ok)])
    lo, hi = flange.min(0), flange.max(0)
    crop = np.array([[lo[0] - 0.45, lo[1] - 0.6, -0.25], [hi[0] + 0.45, hi[1] + 0.6, hi[2] + 0.35]])
    closed = g[ok] > 0.5 * np.nanmax(g[ok])
    tips = flange[closed] if closed.any() else flange
    centre = np.median(tips, 0) - np.array([0, 0, 0.12])
    roi = np.array([centre - 0.22, centre + 0.22])
    return crop, roi, centre


def make_server(port: int | None = None):
    import viser

    return viser.ViserServer(port=port or 8099, verbose=False)


def parse_sources(args) -> tuple[dict, list]:
    """(specs, recordings) from --source/--recording, or the episode-0 shorthand --raw-root/--ma-root."""
    specs = {}
    for item in args.source or []:
        name, rest = item.split("=", 1)
        root, stage, label = rest.split(":", 2)
        specs[name] = {"root": root, "depth_stage": stage, "label": label}
    if args.ma_root:
        specs.setdefault("raw", {"root": args.raw_root, "depth_stage": "sensor_depth", "label": "RealSense raw depth"})
        specs.setdefault("mapanything", {"root": args.ma_root, "depth_stage": "mapanything_depth",
                                         "label": "MapAnything depth"})
    recs = []
    for item in getattr(args, "recording", None) or []:
        name, rest = item.split("=", 1)
        srcs, _, cams = rest.partition("@")
        recs.append((name, srcs.split("+"), [int(c) for c in cams.split(",")] if cams else None))
    if not recs:
        recs = [(n, [n], None) for n in specs]
        if args.ma_root and not args.source:
            recs.append(("compare", ["raw", "mapanything"], None))
    if not specs:
        raise SystemExit("give --source NAME=ROOT:STAGE:LABEL (repeatable) or --ma-root/--raw-root")
    return specs, recs


def cmd_record(args) -> int:
    specs, plan = parse_sources(args)
    ve = vd.load_episode(Path(next(iter(specs.values()))["root"]), args.episode)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    from s4d.realworld.pipeline import RobotGeometry

    geom = RobotGeometry(ve.meta)
    q, g, ok = vd.robot_state_at_sets(ve)
    crop, roi, centre = default_boxes(ve, q, g, ok, geom)
    meta = {**vd.describe(ve), "crop_box_base_m": crop.tolist(), "roi_box_base_m": roi.tolist(),
            "look_at_base_m": centre.tolist(), "recordings": {}, "display": {
                "stride_px": args.stride, "roi_stride_px": args.roi_stride, "trail_length_frames": args.trail_length,
                "tracks_per_camera_sparse": args.max_tracks, "tracks_per_camera_dense": args.dense_tracks,
                "note": "display-only downsampling; lines between measured samples are display interpolation"}}
    if args.only:
        plan = [p for p in plan if p[0] in args.only.split(",")]
    for name, sources, cams in plan:
        server = make_server(args.port)
        t0 = time.time()
        scene = EpisodeScene(server, ve, specs, sources, stride=args.stride, roi_stride=args.roi_stride, roi_box=roi,
                             crop_box=crop, trail_length=args.trail_length, max_tracks=args.max_tracks,
                             dense_tracks=args.dense_tracks, cameras=cams)
        scene.show(0)
        serializer = server.get_scene_serializer()
        n_sets = min(ve.n_sets, args.max_sets or ve.n_sets)
        for k in range(n_sets):
            scene.show(k)
            dt = (ve.times_s[k + 1] - ve.times_s[k]) if k + 1 < n_sets else ve.period_s
            serializer.insert_sleep(float(dt) / args.speed)
        data = serializer.serialize()
        (out / f"{name}.viser").write_bytes(data)
        server.stop()
        cam_centres = np.stack([np.linalg.inv(T)[:3, 3] for T in ve.T_cam_base])
        view_dir = cam_centres.mean(0) - centre
        view_dir = view_dir / np.linalg.norm(view_dir)
        c, s_ = np.cos(np.radians(35)), np.sin(np.radians(35))  # oblique: 35 deg around z from the cameras' mean view
        oblique = np.array([c * view_dir[0] - s_ * view_dir[1], s_ * view_dir[0] + c * view_dir[1], view_dir[2]])
        wide = len(sources) > 1
        look = centre + np.array([0.0, -0.8 * (len(sources) - 1), 0.0])
        # side-by-side: just far enough to frame every copy (1.6 m apart) with the mug/hook region still large
        cam_pos = look + oblique * ((0.6 + 0.6 * len(sources)) if wide else 0.75) + np.array([0, 0, 0.35])
        meta["recordings"][name] = {"file": f"{name}.viser", "bytes": len(data), "sources": sources,
                                    "labels": [specs[s]["label"] for s in sources], "cameras": cams,
                                    "roots": {s: specs[s]["root"] for s in sources},
                                    "seconds_to_build": round(time.time() - t0, 1),
                                    "initial_camera": {"position": cam_pos.round(3).tolist(),
                                                       "look_at": look.round(3).tolist(), "up": [0, 0, 1]}}
        print(name, len(data) / 1e6, "MB", round(time.time() - t0, 1), "s", flush=True)
    client = out / "client"
    if not (client / "index.html").is_file():
        import viser

        shutil.copytree(Path(viser.__file__).parent / "client" / "build", client, dirs_exist_ok=True)
    (out / "viewer.json").write_text(json.dumps(meta, indent=1))
    (out / "index.html").write_text(index_page(meta))
    print(json.dumps({k: v["bytes"] for k, v in meta["recordings"].items()}))
    return 0


def player_url(name: str, cam: dict) -> str:
    from urllib.parse import quote

    f = ",".join
    return (f"client/index.html?playbackPath={quote('../' + name + '.viser')}"
            f"&initialCameraPosition={quote(f(str(x) for x in cam['position']))}"
            f"&initialCameraLookAt={quote(f(str(x) for x in cam['look_at']))}"
            f"&initialCameraUp={quote(f(str(x) for x in cam['up']))}")


def index_page(meta: dict) -> str:
    recs = meta["recordings"]
    tabs = "".join(f'<button data-src="{player_url(n, r["initial_camera"])}">{n}</button>' for n, r in recs.items())
    first = next(iter(recs))
    pauses = ", ".join(f"{p['gap_s']:.2f} s before set {p['before_set'] + 1}" for p in meta["pauses"]) or "none"
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Mug-hanging episode viewer</title>
<style>body{{margin:0;font:13px system-ui,sans-serif;background:#f6f7f8;color:#222}}
header{{padding:8px 12px;display:flex;gap:12px;align-items:center;flex-wrap:wrap}}
button{{padding:4px 10px;border:1px solid #bbb;background:#fff;border-radius:4px;cursor:pointer}}
button.on{{background:#222;color:#fff}} iframe{{width:100%;height:calc(100vh - 92px);border:0;display:block}}
p{{margin:0 12px 6px;color:#555}}</style></head><body>
<header><b>{meta['episode']}</b> {tabs}</header>
<p>Measured episode, not a prediction. {meta['sets']} synchronized sets, {meta['duration_s']:.2f} s,
recording pause: {pauses}.
Trails: tracks classified moving (lines between measured dots are display interpolation; red/blue/green = camera).
Loops automatically; drag to orbit, right-drag to pan, scroll to zoom. Scene tree (playback bar) toggles cameras,
clouds, trails (dense variant hidden), robot (opaque variant hidden) and edits point size / opacity.</p>
<iframe id="v" allow="fullscreen" src="{player_url(first, recs[first]['initial_camera'])}"></iframe>
<script>const b=[...document.querySelectorAll('button')];b[0].classList.add('on');
b.forEach(x=>x.onclick=()=>{{b.forEach(y=>y.classList.remove('on'));x.classList.add('on');
document.getElementById('v').src=x.dataset.src;}});</script></body></html>"""


def cmd_serve(args) -> int:
    specs, _ = parse_sources(args)
    ve = vd.load_episode(Path(next(iter(specs.values()))["root"]), args.episode)
    from s4d.realworld.pipeline import RobotGeometry

    q, g, ok = vd.robot_state_at_sets(ve)
    crop, roi, centre = default_boxes(ve, q, g, ok, RobotGeometry(ve.meta))
    server = make_server(args.port)
    scene = EpisodeScene(server, ve, specs, list(specs), stride=args.stride, roi_stride=args.roi_stride,
                         roi_box=roi, crop_box=crop, trail_length=args.trail_length, max_tracks=args.max_tracks,
                         dense_tracks=args.dense_tracks)
    gui = server.gui
    playing = gui.add_checkbox("Play (loops)", True)
    timeline = gui.add_slider("Set", 0, ve.n_sets - 1, 1, 0)
    speed = gui.add_dropdown("Speed", ("0.25x", "0.5x", "1x", "2x"), "1x")
    point = gui.add_slider("Point size (m)", 0.001, 0.015, 0.0005, 0.006)
    opacity = gui.add_slider("Robot opacity", 0.0, 1.0, 0.05, 0.45)
    trail = gui.add_slider("Trail length (frames)", 1, 15, 1, args.trail_length)
    dense = gui.add_checkbox("Dense trails", False)
    cams = {v: gui.add_checkbox(f"Camera {v} ({ve.streams[v].split('/')[0]})", True) for v in scene.cams}
    srcs = {s: gui.add_checkbox(scene.specs[s]["label"], True) for s in scene.sources}
    depth_layer = gui.add_checkbox("Frustum shows depth (turbo 0.3-2 m)", False)

    def restyle(_=None):
        for s in scene.sources:
            for v in scene.cams:
                on = cams[v].value and srcs[s].value
                scene.h[(s, "cloud", v)].visible = on
                scene.h[(s, "cloud", v)].point_size = point.value
                for kind in ("trails", "trails_samples"):
                    scene.h[(s, kind, v)].visible = on and not dense.value
                for kind in ("trails_dense", "trails_dense_samples"):
                    scene.h[(s, kind, v)].visible = on and dense.value
                scene.h[(s, "frustum", v)].visible = on and not depth_layer.value
                if (s, "frustum_depth", v) in scene.h:
                    scene.h[(s, "frustum_depth", v)].visible = on and depth_layer.value
            for link in scene.meshes:
                scene.h[(s, "robot_translucent", link)].visible = srcs[s].value and opacity.value < 0.99
                scene.h[(s, "robot_opaque", link)].visible = srcs[s].value and opacity.value >= 0.99
            for link in scene.meshes:
                scene.h[(s, "robot_translucent", link, "mesh")].opacity = max(opacity.value, 0.02)

    for w in (point, opacity, dense, depth_layer, *cams.values(), *srcs.values()):
        w.on_update(restyle)

    def on_trail(_):
        scene.trail_length = int(trail.value)
        scene.show(int(timeline.value))

    trail.on_update(on_trail)
    timeline.on_update(lambda _: scene.show(int(timeline.value)))
    restyle()
    scene.show(0)
    print(f"viewer at http://localhost:{args.port}  (Ctrl-C to stop)", flush=True)
    while True:
        if playing.value:
            k = int(timeline.value)
            nxt = (k + 1) % ve.n_sets
            dt = (ve.times_s[nxt] - ve.times_s[k]) if nxt else ve.period_s
            time.sleep(max(0.0, dt / float(speed.value.rstrip("x"))))
            timeline.value = nxt
        else:
            time.sleep(0.05)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("record", "serve"):
        p = sub.add_parser(name)
        p.add_argument("--source", action="append", help="NAME=ROOT:DEPTH_STAGE:LABEL (repeatable): clouds from that "
                                                            "root's depth stage, trails from its moving tracks")
        p.add_argument("--ma-root", help="episode-0 shorthand: root with mapanything_depth tracks")
        p.add_argument("--raw-root", help="episode-0 shorthand: root whose lift3d/track_motion used sensor_depth")
        p.add_argument("--episode", required=True)
        p.add_argument("--stride", type=int, default=6, help="display pixel stride (a multiple of --roi-stride)")
        p.add_argument("--roi-stride", type=int, default=3, help="denser display stride around the mug/hook")
        p.add_argument("--trail-length", type=int, default=8)
        p.add_argument("--max-tracks", type=int, default=400)
        p.add_argument("--dense-tracks", type=int, default=1500)
        p.add_argument("--port", type=int)
        if name == "record":
            p.add_argument("--out", required=True)
            p.add_argument("--speed", type=float, default=1.0, help="playback speed of the recording (1 = real time)")
            p.add_argument("--recording", action="append",
                           help="NAME=SRC[+SRC...][@v0,v1] (repeatable; default one recording per source)")
            p.add_argument("--only", help="comma-separated subset of the recording names")
            p.add_argument("--max-sets", type=int, help="first N sets only (development)")
    args = ap.parse_args(argv)
    return cmd_record(args) if args.cmd == "record" else cmd_serve(args)


if __name__ == "__main__":
    raise SystemExit(main())
