"""Pure pick-and-place-v2 stage graph and queue-row construction. No scheduling, network or file writes.

Upstream stages (arrival -> check-data -> ingest -> Phase A -> rig -> Phase B -> publications -> scene) run on the
established prep dispatchers/producers and are gated as evidence by ``ops/jobs/pnpv2_registration.py``. Queue rows
start at each view's readiness barrier and split into two independent waves:

* ``rgb`` (calibration-independent: DynaMo, ReViWo, E2E CNN, E2E CNN noaug): needs ingest + Phase A + the RGB
  publication only. No row of this wave depends on calibration, Phase B or the geometry barrier.
* ``geometry`` (Ours orig/ch01, SinCro, SNeRL, 3D-MVP single/multiview): needs the accepted rig, all 100 Phase-B
  per-episode calibration records and the geometry publication (+ the train-only native scene measurement).

Row argv mirror the drawer-open-v2 dsreset/rgb/rwv3 rows with concrete published paths; the pnpv2 entrypoints only
install the new task process-locally before the established scripts run.
"""

from __future__ import annotations

import copy
import json

from s4d.rwlearn.pnpv2_spec import METHODS, PAUSE, RGB_METHODS, TASK, row_id, values

TOOL = "scripts/pnpv2.py"
REMOTE_HOST = "192.168.10.22"
PRETRAIN = ("ours_orig", "ours_ch01", "dynamo", "reviwo", "sincro", "snerl", "mvp3d_multiview")
# Measured drawer-open-v2 footprints (dsreset_plan / rwv3 matrix rows): (mem_gb, ram_gb).
PRE_FOOTPRINT = {"ours": (8, 16), "dynamo": (12, 32), "reviwo": (4, 16), "sincro": (12, 24), "snerl": (16, 20),
                 "mvp3d": (12, 24)}
# Stage graph: name -> (kind, depends on). "external" stages are evidence gates evaluated at registration time.
STAGES = {
    "arrival": ("external", ()),
    "check-data": ("external", ("arrival",)),
    "ingest": ("external", ("check-data",)),
    "phaseA": ("external", ("ingest",)),
    "rgb-publication": ("external", ("phaseA",)),
    "rig": ("external", ("phaseA",)),
    "phaseB": ("external", ("rig",)),
    "geometry-publication": ("external", ("phaseB",)),
    "scene": ("external", ("geometry-publication",)),
    "wave:rgb": ("queue", ("check-data", "ingest", "phaseA", "rgb-publication")),
    "wave:geometry": ("queue", ("check-data", "ingest", "phaseA", "rig", "phaseB", "geometry-publication", "scene")),
}


def wave_of(method):
    return "rgb" if method in RGB_METHODS else "geometry"


def base(method):
    return "ours" if method.startswith("ours_") else "mvp3d" if method.startswith("mvp3d") else method


def condition(method):
    return "ours" if method.startswith("ours_") else "e2e_cnn" if method.startswith("e2e_cnn") else method


def build_rows(spec, release, *, scene=None, waves=("rgb", "geometry")):
    """All rows of the selected waves. ``scene`` = validated {lookat, pose_scale, bounds} (geometry wave only)."""
    if not set(waves) <= {"rgb", "geometry"} or not waves:
        raise ValueError("waves must be rgb and/or geometry")
    if "geometry" in waves and (scene is None or any(scene.get(k) is None for k in ("lookat", "pose_scale",
                                                                                     "bounds"))):
        raise ValueError("geometry wave needs the measured lookat/pose_scale/bounds")
    p = values(spec)
    local = release.get("source") == "local"
    common = dict(host="remote", pin_host="remote", remote_host=REMOTE_HOST, gpu="any", max_restarts=1,
                  oom_score_adj=0, cpu_threads=4, disk_gb=2, realworld_target=TASK, realworld_variant="S",
                  planned_commit=release["commit"], planned_image=release["image"], registration_tool=TOOL,
                  release_source=release.get("source", "role"), local_placement_pause=PAUSE,
                  evidence_manifest="docs/tests.json (local evidence)" if local else
                  f"runs/remote/acceptance/{release['commit'][:7]}-cc/manifest.json")
    rows = []

    def add(stage, method, wave, script, args, deps, results, *, seed=None, mem=2, ram=8, role=None, **extra):
        rid = row_id(stage, method, seed)
        if role is None:
            role = ("realworld_preprocess" if stage == "gate" else
                    "realworld_pretrain" if stage in ("stats", "pre", "export") else "realworld_fql")
        priority = (-4 if stage in ("gate", "stats", "native") else
                    (-3 if stage in ("pre", "export") else -1) if wave == "rgb" else
                    (-3 if method.startswith("ours") else 0) if stage == "pre" else
                    (-1 if method.startswith("ours") else 1))
        rows.append(dict(copy.deepcopy(common), id=rid, script=script, args=args, deps=deps, required_results=results,
                         role=role, priority=priority, mem_gb=mem, ram_gb=ram, pnpv2_wave=wave,
                         realworld_method=method, calibration_dependent=wave == "geometry",
                         realworld_inputs=p[f"{wave}_dir"], **({"fql_seed": seed} if seed is not None else {}),
                         **extra))
        return rid

    def view(wave, key):
        return p[f"{wave}_{key}"]

    gates, stats, native, exports = {}, None, None, {}
    for wave in waves:
        gates[wave] = add("gate", wave, wave, "scripts/pnpv2_realworld_dsreset_check.py",
                          ["--stage", "prep" if wave == "rgb" else "calib",
                           "--manifest", view(wave, "manifest"), "--split", view(wave, "split"),
                           "--outcomes", view(wave, "outcomes"), "--gate-report", view(wave, "gate"),
                           "--store", view(wave, "store"), "--out", f"runs/{row_id('gate', wave)}/readiness.json"],
                          [], ["readiness.json"], mem=1, ram=4)
    if "geometry" in waves:
        stats = add("stats", "ours", "geometry", "scripts/pnpv2_compute_workspace_stats.py",
                    ["--regime", "realworld", "--path", view("geometry", "store"), "--split", view("geometry", "split"),
                     "--out", f"runs/{row_id('stats', 'ours')}/workspace_stats.json"],
                    [gates["geometry"]], ["workspace_stats.json"])
        native = row_id("native", "mvp3d")
        add("native", "mvp3d", "geometry", "scripts/pnpv2_realworld_native_depth.py",
            ["build", "--store", view("geometry", "store"), "--rig", p["rig"], "--out-dir", f"runs/{native}/overlays",
             "--out-store", f"runs/{native}/store-native.yaml"], [gates["geometry"]], ["store-native.yaml"], mem=1)
    native_store = f"runs/{native}/store-native.yaml"
    for method in PRETRAIN:
        wave = wave_of(method)
        if wave not in waves:
            continue
        b = base(method)
        pre, exp = row_id("pre", method), row_id("export", method)
        if b == "ours":
            arm = method.split("_", 1)[1]
            sets = [f"data.store={view(wave, 'store')}", f"data.split_manifest={view(wave, 'split')}",
                    f"data.gate_report={view(wave, 'gate')}", f"model.anchor_stats={p['workspace_stats']}"]
            add("pre", method, wave, "scripts/pnpv2_train.py",
                ["--config", f"configs/realworld/ours_matrix/{TASK}-S-{arm}-scratch400k.yaml", "--name", pre,
                 "--output-root", "runs", "--resume", "auto", "--set", *sets], [gates[wave], stats],
                ["completion.json", "metrics.jsonl", "checkpoints/step_0400000.pt"], mem=8, ram=16,
                max_restarts=2, realworld_arm={"orig": "Original", "ch01": "Original+chamfer0.1"}[arm],
                realworld_protocol="scratch", realworld_checkpoint=400000)
            add("export", method, wave, "scripts/pnpv2_export_encoder.py",
                ["--run", f"runs/{pre}", "--checkpoint", f"runs/{pre}/checkpoints/step_0400000.pt",
                 "--out", f"runs/{exp}/encoder.pt"], [pre], ["encoder.pt"])
        else:
            source = native_store if b == "mvp3d" else view(wave, "store")
            sets = [f"dataset.hdf5_path={source}", f"data_interface.split_manifest={view(wave, 'split')}",
                    f"data_interface.gate_report={view(wave, 'gate')}"]
            if b == "mvp3d":
                sets.append("render.bounds=" + json.dumps(scene["bounds"], separators=(",", ":")))
            if b == "snerl":
                sets += ["data_interface.lookat=" + json.dumps(scene["lookat"], separators=(",", ":")),
                         f"data_interface.pose_scale={float(scene['pose_scale'])!r}"]
            resume = ["--resume", "auto"] if b in ("mvp3d", "snerl", "dynamo") else []
            mem, ram = PRE_FOOTPRINT[b]
            add("pre", method, wave, f"scripts/pnpv2_train_{b}.py",
                ["--config", f"configs/realworld/baselines/{b}/{TASK}.yaml", "--name", pre, "--output-root", "runs",
                 *resume, "--set", *sets], [gates[wave]] + ([native] if b == "mvp3d" else []),
                ["encoder.pt", "metrics.jsonl"], mem=mem, ram=ram)
            add("export", method, wave, "scripts/pnpv2_realworld_dsreset_artifact.py",
                ["--source", f"runs/{pre}/encoder.pt", "--kind", "encoder", "--out", f"runs/{exp}/encoder.pt"],
                [pre], ["encoder.pt"])
        exports[method] = exp
    exports["mvp3d_single"] = exports.get("mvp3d_multiview")
    for method in METHODS:
        wave = wave_of(method)
        if wave not in waves:
            continue
        pixel = method.startswith("e2e_cnn")
        cond = condition(method)
        data = row_id("fqldata", method)
        source = native_store if method.startswith("mvp3d") else view(wave, "store")
        args = ["build", "--store", source, "--condition", cond, "--out", f"runs/{data}/dataset.h5",
                "--gate-report", view(wave, "gate"), "--outcomes", view(wave, "outcomes"),
                "--device", "cpu" if pixel else "cuda"]
        deps = [gates[wave]]
        if not pixel:
            args += ["--export", f"runs/{exports[method]}/encoder.pt"]
            deps.append(exports[method])
        if method.startswith("mvp3d"):
            deps.append(native)
        add("fqldata", method, wave, "scripts/pnpv2_realworld_fql_dataset.py", args, deps, ["dataset.h5"],
            mem=1 if pixel else 8, ram=8 if pixel else 16, fql_condition=cond)
        for seed in spec["seeds"]:
            fql = row_id("fql", method, seed)
            sets = [f"condition={cond}", f"data.dataset=runs/{data}/dataset.h5",
                    f"data.split_manifest={view(wave, 'split')}", f"data.gate_report={view(wave, 'gate')}",
                    f"train.seed={seed}"]
            if pixel:
                sets.append(f"data.store={view(wave, 'store')}")
            add("fql", method, wave, "scripts/pnpv2_train_realworld_fql.py",
                ["--config", f"configs/realworld/fql/{TASK}-{method}.yaml", "--name", fql, "--output-root", "runs",
                 "--set", *sets], [data], ["policy.pt", "metrics.jsonl"], seed=seed,
                mem=12 if pixel else 6, ram=24 if pixel else 12, fql_condition=cond,
                fql_setting="a10" if pixel else "a10nq")
            ev = row_id("eval", method, seed)
            add("eval", method, wave, "scripts/pnpv2_realworld_dsreset_artifact.py",
                ["--source", f"runs/{fql}/policy.pt", "--kind", "policy", "--out", f"runs/{ev}/evaluation.json"],
                [fql], ["evaluation.json"], seed=seed)
    return rows


def counts(rows):
    out = {}
    for row in rows:
        key = (row["pnpv2_wave"], row["id"].split("-")[2])
        out[key] = out.get(key, 0) + 1
    return {f"{w}/{s}": n for (w, s), n in sorted(out.items())}
