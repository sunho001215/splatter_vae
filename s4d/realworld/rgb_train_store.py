"""RGB-only compact export using the established writer algorithm; isolated from existing row scopes."""
from __future__ import annotations
from s4d.realworld.train_store import (
    CALIBRATION_OPTIONAL_LAYOUT, CONVENTIONS, CR, EPISODE_SCHEMA, FILE,
    FORMAT, GZIP, IMAGE_SIZE, MAX_SET_MOTION_M, NEAREST_RESAMPLE,
    Path, RGB_RESAMPLE, Rotation, SCHEMA_VERSION, SEMANTIC_CLASSES,
    STAGE, SYNC_CRITERION, WRITER_VERSION, _camera_tracks, _code_commit,
    box_affine, component_sizes, h5py, json, mutual_sync,
    nearest_resample, np, recording_segments, rio, set_surface_motion,
    square_box, sync_counts, time, training_depth,
)

def write_episode(root: Path, episode_id: str, out_dir: Path, *, task: str | None = None, depth_stage: str | None = None,
                  semantic: bool | None = None, max_sync_ns: int | None = None, gripper_points: int = 32,
                  image_size: int = IMAGE_SIZE, chunk: int = 16, allow_mock: bool = False, force: bool = False,
                  cross_camera_cap_ns: int | None = None, max_set_motion_m: float = MAX_SET_MOTION_M,
                  rgb_only: bool = True) -> dict:
    """Publish RGB/robot/M5 fields from normalized frames with camera geometry held.

    Isolated variant of train_store.write_episode; the established writer stays byte-identical.
    The unused geometry branches remain visible to make comparison with that algorithm direct.

    Inputs (complete, validated stage records of ``root``): ingest (normalized frames, timestamps, robot trajectory),
    ``depth_stage`` (default: the FoundationStereo IR depth on the colour grid), ``calibrate`` (accepted T_cam_base),
    ``track2d``/``lift3d`` (TrackEverything) and ``track_motion``; ``sam3_masks`` when ``semantic`` (default: when its
    record exists). ``max_sync_ns`` is the per-camera ROBOT <-> camera sync tolerance (default: a third of the reference
    period, as the window export); it no longer bounds the cross-camera sets: those pair mutual nearest frames without a
    time cap (``cross_camera_cap_ns`` None) and store each set's robot-surface motion (criterion M5; the loaders accept
    a set iff it is <= their threshold, default ``max_set_motion_m`` 5 mm, which also sets the recorded counts).
    ``task`` defaults to the dataset folder of the recording (the adapter's task identity). Cameras without an accepted
    calibration retain RGB and robot sync under ``CALIBRATION_OPTIONAL_LAYOUT``. Geometry fields are omitted for
    those cameras; multi-view and geometric consumers must refuse them. Fully accepted episodes keep the existing
    layout and fingerprint (including the existing metric-depth/track eligibility rules).
    ``rgb_only=True`` explicitly reads ingest alone and omits ALL camera geometry and semantics,
    even when later stages exist. RGB, native frame selection, robot sync and M5 computation
    use the same implementation; this mode cannot supply geometric consumers."""
    from s4d.data.droid.pointworld import depth_test
    from s4d.realworld.export import motion_target_source
    from s4d.realworld.pipeline import (
        RobotGeometry,
        accepted_calibration,
        calibrated_geometry_config,
        calibrated_version,
        calibration_depth_scale,
        calibration_robot_model,
        camera_views,
        default_depth_stage,
        load_meta,
        read_frames,
        robot_state_at,
        to_model_rgb,
    )
    from s4d.realworld.stages.depth import load_view_depth

    if not rgb_only:
        raise ValueError("use the established writer for geometric exports")
    if rgb_only and semantic:
        raise ValueError("rgb_only cannot publish semantic targets")
    root, out_dir = Path(root), Path(out_dir)
    meta = load_meta(root, episode_id)
    depth_stage = depth_stage or default_depth_stage(meta, "lift")
    task = task or Path(meta.provenance.source_root or "").name
    if not task:
        raise ValueError(f"{episode_id}: no task given and the recording has no dataset folder")
    norm = rio.read_record(rio.episode_dir(root, episode_id) / "normalized")
    names = [depth_stage, "calibrate", "track2d", "lift3d", "track_motion"]
    if rgb_only:
        names, semantic = [], False
    if semantic is None:
        semantic = rio.read_record(rio.stage_dir(root, episode_id, "sam3_masks")) is not None
    if semantic:
        names.append("sam3_masks")
    records = {k: rio.read_record(rio.stage_dir(root, episode_id, k)) for k in names}
    views = camera_views(meta)
    calibration = {v.stream_id: (None if rgb_only else accepted_calibration(root, episode_id, v.stream_id))
                   for v in views}
    partial = any(calibration[v.stream_id] is None for v in views)
    required = [] if rgb_only else (["calibrate", *(["sam3_masks"] if semantic else [])] if partial else names)
    missing = [k for k in required if records[k] is None] + ([] if norm else ["ingest"])
    if missing:
        raise RuntimeError(f"{STAGE} needs complete upstream stages: {missing}")
    mock = any('"is_mock": true' in json.dumps(r, default=str) for r in records.values())
    if mock and not allow_mock:
        raise RuntimeError("upstream stages used MOCK model outputs; refusing to write training data (allow_mock=False)")
    excluded, chosen, geometry = {}, [], set()
    for view in views:
        cam = view.stream_id
        reason = None
        if calibration[cam] is None:
            reason = "calibration missing or rejected"
        elif meta.rig_rgbd(view.rig_id).status != "metric":
            reason = "no metric depth"
        elif "chunks" not in ((records["lift3d"] or {}).get("summary") or {}).get(cam, {}):
            reason = "no lifted tracks"
        elif meta.stream(cam).modality != "rgb":
            reason = f"modality {meta.stream(cam).modality} (training images are RGB)"
        if reason is None:
            missing = [k for k in names if records[k] is None]
            if missing:
                raise RuntimeError(f"{STAGE} needs complete upstream stages: {missing}")
            geometry.add(cam)
        if reason is None or partial and meta.stream(cam).modality == "rgb":
            chosen.append(view)
        else:
            excluded[view.rig_id] = reason
    if not chosen:
        raise RuntimeError(f"no camera for the training store: {excluded}")
    chosen.sort(key=lambda v: v.rig_id)
    ref_stamps = np.asarray(meta.stream_timestamps_ns[chosen[0].stream_id], np.int64)
    if max_sync_ns is None:
        max_sync_ns = int(np.median(np.diff(ref_stamps)) // 3) if len(ref_stamps) > 1 else 0
    config = {"task": task, "depth_stage": depth_stage, "semantic": bool(semantic), "max_sync_ns": int(max_sync_ns),
              "gripper_points": int(gripper_points), "image_size": int(image_size), "schema": SCHEMA_VERSION,
              "writer_version": WRITER_VERSION, "allow_mock": bool(allow_mock), "sync_criterion": SYNC_CRITERION,
              "cross_camera_cap_ns": None if cross_camera_cap_ns is None else int(cross_camera_cap_ns),
              "max_set_motion_m": float(max_set_motion_m)}
    if rgb_only:
        config["rgb_only"] = True
    if partial:
        config["layout"] = CALIBRATION_OPTIONAL_LAYOUT
    # calibration-derived settings of the GEOMETRY fields (stored depth and gripper-visibility test x the accepted
    # per-camera depth scale; gripper target points posed with the accepted robot model). Robot proprioception
    # (probe_state closure) and the M5 set motion keep the legacy robot geometry, so consumers that use no geometry
    # see identical data. Recorded - and the writer version bumped - only when they differ from the legacy calibration.
    cal_extra = calibrated_geometry_config(root, episode_id, meta, depth_stage) if geometry else {}
    config.update(cal_extra)
    writer_version = calibrated_version(WRITER_VERSION, cal_extra)
    scales = {v.stream_id: (calibration_depth_scale(root, episode_id, v.stream_id, depth_stage)
                            if v.stream_id in geometry else 1.0) for v in chosen}
    inputs = {"ingest": rio.build_id(norm), **{k: rio.build_id(v) for k, v in records.items() if v is not None}}

    def build(tmp: Path) -> dict:
        t_start = time.perf_counter()
        geom = RobotGeometry(meta)  # legacy: probe closure + M5 set motion (unchanged for non-geometric consumers)
        model = cal_extra.get("robot_model")
        ggeom = geom if model is None else RobotGeometry(meta, gripper_map=model["gripper_map"],
                                                         surface=model["surface"])
        if model is not None and calibration_robot_model(root, episode_id, meta) != model:
            raise RuntimeError("robot model changed while writing")
        rng = np.random.default_rng(0)  # the window export's gripper sample (same points)
        links = ggeom.gripper_links()
        pool = np.concatenate([ggeom.surface.link_points[n] for n in links])
        pool_link = np.concatenate([[n] * len(ggeom.surface.link_points[n]) for n in links])
        pick = rng.choice(len(pool), min(gripper_points, len(pool)), replace=False)
        source = meta.provenance.source_files[0] if meta.provenance.source_files else None
        timing, calibration_status = {}, {}
        with h5py.File(tmp / FILE, "x") as f:
            f.attrs.update(
                format=FORMAT, schema_version=SCHEMA_VERSION, writer_version=writer_version,
                code_commit=_code_commit() or "unknown", episode_id=episode_id, task=task,
                episode_schema=EPISODE_SCHEMA, created_unix=time.time(),
                raw_recording=str(source.path) if source else "", raw_sha256=str(source.sha256) if source else "",
                cameras=json.dumps([v.rig_id for v in chosen]), reference_camera=chosen[0].rig_id,
                excluded_cameras=json.dumps(excluded), image_size=int(image_size),
                depth_encoding="float32 metres (metric camera-z of the depth stage, bitwise), 0 = invalid",
                conventions=json.dumps(CONVENTIONS), rgb_resample=RGB_RESAMPLE, nearest_resample=NEAREST_RESAMPLE,
                semantic_classes=json.dumps(SEMANTIC_CLASSES if semantic else []),
                upstream=json.dumps(inputs), depth_stage=depth_stage,
                models=json.dumps({depth_stage: (records.get(depth_stage) or {}).get("summary", {}).get("model"),
                                   "track2d": {"tracker": (records.get("track2d") or {}).get("config", {}).get("tracker"),
                                               "worker": (records.get("track2d") or {}).get("config", {}).get("worker")},
                                   **({"sam3_masks": {c: s.get("model") for c, s in records["sam3_masks"]["summary"]
                                                      .items() if isinstance(s, dict)}} if semantic else {})},
                                  default=str),
                track_source=((motion_target_source(records["track2d"], depth_stage) if geometry else
                               "unavailable") + "; ONLY tracks classified "
                              "moving by track_motion, full valid trajectories with their motion segments"),
                gripper_target=f"{len(pick)} surface points of the URDF gripper links {links} via FK at the "
                               "camera frame's interpolated robot state (seed-0 sample, as the window export)",
                probe_state="panda flange xyz (m), flange euler xyz (rad, scipy 'xyz'), gripper closure in [0,1] "
                            "(0 open; recorded state through the URDF joint range)",
                max_sync_ns=int(max_sync_ns), MOCK_MODEL_OUTPUTS=bool(mock))
            if partial:
                f.attrs["layout"] = CALIBRATION_OPTIONAL_LAYOUT
            if cal_extra:
                f.attrs["calibration_derived"] = json.dumps(cal_extra)
            tr = meta.trajectory
            g = f.create_group("robot")
            g.create_dataset("timestamps_ns", data=tr.timestamps_ns.astype(np.int64))
            g.create_dataset("joint_positions", data=np.asarray(tr.joint_positions, np.float64), **GZIP)
            g.create_dataset("gripper_state", data=np.asarray(tr.gripper_state, np.float64), **GZIP)
            if tr.ee_pose_base is not None:
                g.create_dataset("ee_pose_base", data=np.asarray(tr.ee_pose_base, np.float64), **GZIP)
            r = meta.robot
            g.attrs.update(joint_names=json.dumps(r.joint_names), flange_frame=r.flange_frame,
                           ee_frame=str(tr.ee_frame), gripper_model=r.gripper_model,
                           gripper_state_semantics=r.gripper_state_semantics, urdf_sha256=json.dumps(r.asset_sha256),
                           robot_sync_max_ns=int(max_sync_ns))
            stamps_all, states_all, ok_all = [], [], []
            for view in chosen:
                cam, rig = view.stream_id, view.rig_id
                t0 = time.perf_counter()
                s = meta.stream(cam)
                stamps = np.asarray(meta.stream_timestamps_ns[cam], np.int64)
                stamps_all.append(stamps)
                n, H, W = len(stamps), s.intrinsics.height, s.intrinsics.width
                box = square_box(H, W)
                aff = box_affine(box, image_size)
                has_geometry = cam in geometry
                if has_geometry:
                    _, _, K_nat = load_view_depth(root, episode_id, cam, [0], depth_stage)
                else:
                    K_nat = s.intrinsics.K()
                w2c = None if rgb_only else accepted_calibration(root, episode_id, cam)
                cg = f.create_group(f"cameras/{rig}")
                cal_path = rio.stage_dir(root, episode_id, "calibrate") / f"{rio.safe_name(cam)}.json"
                cal = (json.loads(cal_path.read_text()) if not rgb_only and cal_path.is_file()
                       else {"status": "missing"})
                calibration_status[rig] = cal["status"]
                cg.attrs.update(stream_id=cam, rig_id=rig, modality=s.modality, channel_order="RGB",
                                native_hw=[H, W], depth_stage=depth_stage,
                                calibration=json.dumps({"status": cal["status"], "reasons": cal.get("reasons"),
                                                        "init": cal.get("init_method")}, default=str))
                if partial:
                    cg.attrs.update(calibration_status=cal["status"], geometry_available=has_geometry)
                cg.create_dataset("timestamps_ns", data=stamps)
                cg.create_dataset("segment", data=recording_segments(meta, stamps))
                cg.create_dataset("K_native", data=np.asarray(K_nat, np.float64))
                cg.create_dataset("K", data=aff.apply_K(K_nat))
                cg.create_dataset("affine", data=aff.matrix())
                cg.create_dataset("crop_box", data=np.array([box.x0, box.y0, box.side], np.int64))
                if has_geometry:
                    cg.create_dataset("w2c", data=np.asarray(w2c, np.float64))
                S = int(image_size)
                rgb_ds = cg.create_dataset("rgb", (n, S, S, 3), np.uint8, chunks=(1, S, S, 3), **GZIP)
                if has_geometry:
                    dep_ds = cg.create_dataset("depth", (n, S, S), np.float32, chunks=(1, S, S), shuffle=True, **GZIP)
                sem_path = rio.stage_dir(root, episode_id, "sam3_masks") / f"{rio.safe_name(cam)}.h5"
                camera_semantic = bool(semantic and sem_path.is_file())
                if semantic and not camera_semantic and has_geometry:
                    raise RuntimeError(f"{cam}: missing semantic masks")
                if camera_semantic:
                    sem_ds = cg.create_dataset("semantic", (n, S, S), np.uint8, chunks=(1, S, S), **GZIP)
                    sem_file = h5py.File(sem_path, "r")
                    if json.loads(sem_file.attrs.get("class_names", json.dumps(SEMANTIC_CLASSES))) != list(SEMANTIC_CLASSES):
                        raise RuntimeError(f"{cam}: unexpected semantic classes")
                    cg.create_dataset("class_present", data=sem_file["class_present"][:].astype(bool))
                try:
                    for start in range(0, n, chunk):
                        idx = list(range(start, min(n, start + chunk)))
                        frames = to_model_rgb(read_frames(root, episode_id, cam, idx), s)
                        if frames.shape[1:] != (H, W, 3):
                            raise RuntimeError(f"{cam}: frames {frames.shape} are not RGB {H}x{W}")
                        rgb_ds[start:start + len(idx)] = np.stack([CR.crop_rgb(im, box, S) for im in frames])
                        if has_geometry:
                            depth, valid, _ = load_view_depth(root, episode_id, cam, idx, depth_stage)
                            if scales[cam] != 1.0:  # the accepted calibration's depth scale (consistent with w2c)
                                depth = depth * np.float32(scales[cam])
                            depth = training_depth(depth, valid)
                            dep_ds[start:start + len(idx)] = np.stack([nearest_resample(d, box, S) for d in depth])
                        if camera_semantic:
                            sem_ds[start:start + len(idx)] = np.stack([nearest_resample(sem_file["semantic"][i], box, S)
                                                                       for i in idx])
                finally:
                    if camera_semantic:
                        sem_file.close()
                # robot state at this camera's frame times (the window export's rule at the reference camera)
                q, gs, ok, err = robot_state_at(meta, stamps, max_sync_ns)
                states_all.append(list(zip(q, gs)))
                ok_all.append(np.asarray(ok, bool))
                grip = np.zeros((n, len(pick), 3), np.float32)
                probe = np.zeros((n, 7), np.float32)
                for k in range(n):
                    fk = geom.model.fk(geom.q(q[k], gs[k]))
                    gfk = fk if ggeom is geom else ggeom.model.fk(ggeom.q(q[k], gs[k]))
                    grip[k] = np.stack([gfk[pool_link[i]][:3, :3] @ pool[i] + gfk[pool_link[i]][:3, 3] for i in pick])
                    T = fk[meta.robot.flange_frame]
                    probe[k] = np.concatenate([T[:3, 3], Rotation.from_matrix(T[:3, :3]).as_euler("xyz"),
                                               [geom.closure(gs[k])]])
                rg = cg.create_group("robot")
                rg.create_dataset("ok", data=ok.astype(bool))
                rg.create_dataset("sync_err_ns", data=np.asarray(err, np.int64))
                rg.create_dataset("joint_positions", data=np.asarray(q, np.float64), **GZIP)
                rg.create_dataset("gripper_state", data=np.asarray(gs, np.float64), **GZIP)
                rg.create_dataset("probe_state", data=probe, **GZIP)
                rg.create_dataset("gripper_points", data=grip, shuffle=True, **GZIP)
                if has_geometry:
                    tracks = _camera_tracks(root, episode_id, cam, stamps)
                    tg = cg.create_group("tracks")
                    tg.attrs["chunk_names"] = json.dumps(tracks.pop("_names"))
                    for key, arr in tracks.items():
                        big = arr.ndim >= 2 and arr.size > 64
                        tg.create_dataset(key, data=arr, **({**GZIP, "shuffle": True} if big else {}))
                timing[rig] = round(time.perf_counter() - t0, 2)
            frames_sync, err_sync = mutual_sync(stamps_all, cross_camera_cap_ns)
            disp = set_surface_motion(lambda st: geom.surface.points(geom.q(st[0], st[1]))[0], frames_sync,
                                      states_all, ok_all)
            counts = sync_counts(frames_sync, disp, max_set_motion_m)
            sg = f.create_group("sync")
            sg.create_dataset("frames", data=frames_sync)
            sg.create_dataset("error_ns", data=err_sync)
            sg.create_dataset("surface_disp_m", data=disp.astype(np.float64))  # float64: counts == loaders exactly
            # gripper FK points at the reference frame times, depth-tested (2 cm) against each camera's NATIVE-resolution
            # depth of its synchronised frame: the window export's gripper visibility, kept exactly (G bits per set)
            ref_grip = f[f"cameras/{chosen[0].rig_id}/robot/gripper_points"][:]
            gvis = np.zeros((len(frames_sync), len(chosen), len(pick)), bool)
            for v, view in enumerate(chosen):
                if view.stream_id not in geometry:
                    continue
                K_nat = f[f"cameras/{view.rig_id}/K_native"][:].astype(np.float32)
                w2c = f[f"cameras/{view.rig_id}/w2c"][:].astype(np.float32)
                rows = np.flatnonzero(frames_sync[:, v] >= 0)
                for start in range(0, len(rows), chunk):
                    sel = rows[start:start + chunk]
                    depth, valid, _ = load_view_depth(root, episode_id, view.stream_id, frames_sync[sel, v], depth_stage)
                    depth = np.where(valid, depth * np.float32(scales[view.stream_id]), 0.0).astype(np.float32)
                    for k, t in enumerate(sel):
                        gvis[t, v] = depth_test(ref_grip[t], K_nat, w2c, depth[k], continuous=True)[0]
            if len(geometry) == len(chosen):
                sg.create_dataset("gripper_visible", data=gvis, **GZIP)
            n_points = int(sum(len(x) for x in geom.surface.link_points.values()))
            sg.attrs.update(reference_camera=chosen[0].rig_id, cameras=json.dumps([v.rig_id for v in chosen]),
                            robot_sync_max_ns=int(max_sync_ns), criterion=SYNC_CRITERION,
                            cross_camera_cap_ns=-1 if cross_camera_cap_ns is None else int(cross_camera_cap_ns),
                            default_max_set_motion_m=float(max_set_motion_m), counts=json.dumps(counts),
                            policy=("M5: mutual nearest frame of every camera to the reference frame, no time cap; a set "
                                    "is accepted iff surface_disp_m (largest URDF-surface displacement between any two "
                                    "members, robot state interpolated to each member's timestamp) <= the loader's "
                                    f"threshold (default {max_set_motion_m} m)"),
                            surface=f"RobotGeometry URDF surface, {n_points} points over "
                                    f"{len(geom.surface.link_points)} links (seed 0)")
        sizes = component_sizes(tmp / FILE)
        return {"format": FORMAT, "schema_version": SCHEMA_VERSION, "writer_version": writer_version,
                "code_commit": _code_commit(), "task": task, "cameras": [v.rig_id for v in chosen],
                "excluded_cameras": excluded, "frames": {v.rig_id: len(t) for v, t in zip(chosen, stamps_all)},
                "sync_sets": int((frames_sync >= 0).all(1).sum()), "sync": {"criterion": SYNC_CRITERION, **counts},
                "semantic": bool(semantic),
                "moving_tracks": sizes.pop("_tracks"), "bytes": sizes, "file_bytes": (tmp / FILE).stat().st_size,
                "file_sha256": rio.sha256_file(tmp / FILE), "write_seconds": round(time.perf_counter() - t_start, 2),
                "camera_seconds": timing, "mock": bool(mock),
                **({"layout": CALIBRATION_OPTIONAL_LAYOUT, "calibration_status": calibration_status}
                   if partial else {})}

    return rio.run_stage(out_dir, stage=STAGE, version=writer_version, config=config, inputs=inputs, build=build,
                         force=force)
