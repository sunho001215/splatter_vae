"""Pure, stdout-only candidate graph for a dataset that has not been collected.

These are review rows, not admission evidence or queue publication. Do not reuse
old drawer/v1 IDs, data, gate amendments, ownership records or calibration rigs.
"""

from pathlib import Path

from s4d.rwlearn.pnpv2_spec import METHODS, RGB_METHODS, SPEC, TASK, config, row_id, values

SCRIPT = "scripts/pick_and_place_v2.py"


def build_rows(spec, root, spec_path=SPEC):
    p = values(spec)
    rows = []

    def add(stage, method, entry, args, deps, results, cfg=None, seed=None, mem=2, ram=8):
        rid = row_id(stage, method, seed)
        argv = ["--entry", entry, "--task-spec", str(spec_path)]
        if cfg:
            config(Path(root) / cfg, spec)  # reject unknown tokens during preparation
            argv += ["--config", cfg]
        argv += ["--", *args]
        rows.append(
            dict(
                id=rid,
                script=SCRIPT,
                args=argv,
                deps=deps,
                required_results=results,
                realworld_target=TASK,
                realworld_variant="S",
                realworld_method=method,
                preparation_only=True,
                local_placement_pause="pnpv2-await-explicit-authorization",
                planned_commit=spec["release"]["commit"],
                planned_image=spec["release"]["image"],
                cpu_threads=4,
                mem_gb=mem,
                ram_gb=ram,
                disk_gb=2,
                max_restarts=1,
                priority=-4
                if stage in ("gate", "stats", "native")
                else (-3 if method.startswith("ours") else 0)
                if stage == "pre"
                else 1,
            )
        )
        return rid

    gates = {}
    for view in ("rgb", "geometry"):
        rid = row_id("gate", view)
        gates[view] = add(
            "gate", view, "gate", ["--view", view, "--out", f"runs/{rid}/readiness.json"], [], ["readiness.json"]
        )
    stats = row_id("stats", "ours")
    add(
        "stats",
        "ours",
        "stats",
        ["--regime", "realworld", "--path", p["geometry_store"], "--split", p["split"], "--out", p["workspace_stats"]],
        [gates["geometry"]],
        ["workspace_stats.json"],
    )
    native = row_id("native", "mvp3d")
    native_store = f"runs/{native}/store-native.yaml"
    add(
        "native",
        "mvp3d",
        "native",
        [
            "build",
            "--store",
            p["geometry_store"],
            "--rig",
            p["rig"],
            "--out-dir",
            f"runs/{native}/overlays",
            "--out-store",
            native_store,
        ],
        [gates["geometry"]],
        ["store-native.yaml"],
    )
    exports = {}
    for method in ("ours_orig", "ours_ch01", "dynamo", "reviwo", "sincro", "snerl", "mvp3d_multiview"):
        ours = method.startswith("ours_")
        base = "mvp3d" if method.startswith("mvp3d") else method
        cfg = (
            f"configs/realworld/ours_matrix/{TASK}-S-{method[5:]}-scratch400k.yaml"
            if ours
            else f"configs/realworld/baselines/{base}/{TASK}.yaml"
        )
        view = "rgb" if method in RGB_METHODS else "geometry"
        pre = row_id("pre", method)
        args = ["--name", pre, "--output-root", "runs"]
        if ours or base in ("dynamo", "snerl", "mvp3d"):
            args += ["--resume", "auto"]
        if base == "mvp3d":
            args += ["--set", f"dataset.hdf5_path={native_store}"]
        deps = [gates[view]] + ([stats] if ours else [native] if base == "mvp3d" else [])
        results = (
            ["completion.json", "metrics.jsonl", "checkpoints/step_0400000.pt"] if ours else ["encoder.pt", "metrics.jsonl"]
        )
        mem, ram = (
            (8, 16)
            if ours
            else {"dynamo": (12, 32), "reviwo": (4, 16), "sincro": (12, 24), "snerl": (16, 20), "mvp3d": (12, 24)}[base]
        )
        add("pre", method, "train" if ours else base, args, deps, results, cfg, mem=mem, ram=ram)
        exp = row_id("export", method)
        if ours:
            args = [
                "--run",
                f"runs/{pre}",
                "--checkpoint",
                f"runs/{pre}/checkpoints/step_0400000.pt",
                "--out",
                f"runs/{exp}/encoder.pt",
            ]
            entry = "export"
        else:
            args = ["--source", f"runs/{pre}/encoder.pt", "--kind", "encoder", "--out", f"runs/{exp}/encoder.pt"]
            entry = "artifact"
        exports[method] = add("export", method, entry, args, [pre], ["encoder.pt"])
    exports["mvp3d_single"] = exports["mvp3d_multiview"]
    for method in METHODS:
        pixel = method.startswith("e2e_cnn")
        condition = "ours" if method.startswith("ours_") else "e2e_cnn" if pixel else method
        view = "rgb" if method in RGB_METHODS else "geometry"
        data = row_id("fqldata", method)
        source = native_store if method.startswith("mvp3d") else p[view + "_store"]
        args = [
            "build",
            "--store",
            source,
            "--condition",
            condition,
            "--out",
            f"runs/{data}/dataset.h5",
            "--gate-report",
            p[view + "_gate"],
            "--outcomes",
            p["outcomes"],
            "--device",
            "cpu" if pixel else "cuda",
        ]
        deps = [gates[view]]
        if not pixel:
            args += ["--export", f"runs/{exports[method]}/encoder.pt"]
            deps += [exports[method]]
        if method.startswith("mvp3d"):
            deps += [native]
        add("fqldata", method, "fqldata", args, deps, ["dataset.h5"], mem=1 if pixel else 8, ram=8 if pixel else 16)
        for seed in spec["seeds"]:
            fql = row_id("fql", method, seed)
            add(
                "fql",
                method,
                "fql",
                ["--name", fql, "--output-root", "runs", "--set", f"train.seed={seed}"],
                [data],
                ["policy.pt", "metrics.jsonl"],
                f"configs/realworld/fql/{TASK}-{method}.yaml",
                seed,
                mem=12 if pixel else 6,
                ram=24 if pixel else 12,
            )
            ev = row_id("eval", method, seed)
            add(
                "eval",
                method,
                "eval",
                ["--source", f"runs/{fql}/policy.pt", "--out", f"runs/{ev}/evaluation.json"],
                [fql],
                ["evaluation.json"],
                f"configs/realworld/eval/{TASK}-{method}.yaml",
                seed,
            )
    return rows
