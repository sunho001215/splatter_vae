"""Dedicated future PNP v2 workflow entrypoint. No scheduler, robot or remote API.

Config paths are templates resolved from one task spec before existing trainers
are imported. This entrypoint is not called by the preparation-only registrar.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from s4d.rwlearn.pnpv2_spec import SPEC, config, load_spec, require_collected, values

ENTRIES = dict(
    train="scripts.train",
    dynamo="scripts.train_dynamo",
    reviwo="scripts.train_reviwo",
    sincro="scripts.train_sincro",
    snerl="scripts.train_snerl",
    mvp3d="scripts.train_mvp3d",
    fql="scripts.train_realworld_fql",
    fqldata="scripts.realworld_fql_dataset",
    stats="scripts.compute_workspace_stats",
    native="scripts.realworld_native_depth",
    export="scripts.export_encoder",
    artifact="scripts.realworld_dsreset_artifact",
    snapshot="scripts.realworld_dsreset_snapshot",
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entry", choices=(*ENTRIES, "gate", "eval", "manifest", "publish"), required=True)
    parser.add_argument("--task-spec", type=Path, default=Path(SPEC))
    parser.add_argument("--config", type=Path)
    parser.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    spec = load_spec(args.task_spec)
    require_collected(spec)
    forwarded = args.args[1:] if args.args[:1] == ["--"] else args.args
    # Future operations keep the existing device guard ahead of torch imports.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _bootstrap import guard_gpus

    guard_gpus()
    from s4d.rwlearn.pnpv2_runtime import install

    install()
    if args.entry == "gate":
        from s4d.rwlearn.pnpv2_publication import check_publication

        sub = argparse.ArgumentParser()
        sub.add_argument("--view", choices=("rgb", "geometry"), required=True)
        sub.add_argument("--out", type=Path, required=True)
        a = sub.parse_args(forwarded)
        result = check_publication(spec, a.view)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(result, indent=2) + "\n")
        return
    if args.entry == "manifest":
        from s4d.rwlearn.dsreset import paired_manifests, prep_bundle, write_bundle
        from s4d.rwlearn.pnpv2_publication import manifest_check

        sub = argparse.ArgumentParser()
        sub.add_argument("--out", type=Path, required=True)
        a = sub.parse_args(forwarded)
        p = values(spec)
        if not args.config or not spec["sam3_prompts"]:
            raise ValueError("resolved task prep template and reviewed SAM3 prompts required")
        manifests = paired_manifests(
            spec["task"],
            {"success": p["raw_success"]},
            variants=("S",),
            expected={"success": 100},
            seed=0,
            validation_fraction=0.1,
        )
        manifest_check(manifests["S"], spec)
        prep = prep_bundle(manifests, config(args.config, spec), spec["roots"]["processed"])
        write_bundle(a.out, manifests, prep)
        return
    if args.entry == "publish":
        publish(spec, forwarded)
        return
    cfg = config(args.config, spec) if args.config else None
    if args.entry in ("snerl", "mvp3d"):
        keys = ("lookat", "pose_scale") if args.entry == "snerl" else ("bounds",)
        if any(spec["scene"][key] is None for key in keys):
            raise ValueError("measure task scene parameters on its accepted store first")
    if args.entry == "eval":
        if cfg != dict(
            schema="pick-and-place-v2-offline-eval/1",
            task=spec["task"],
            method=cfg["method"],
            kind="policy",
            required_steps=1000000,
            require_final_validation=True,
            robot_io=False,
            actuation_eligible=False,
        ):
            raise ValueError("only the prepared offline policy evaluation protocol is supported")
        forwarded = ["--kind", "policy", *forwarded]
        cfg = None
    module = importlib.import_module(ENTRIES.get(args.entry, "scripts.realworld_dsreset_artifact"))
    if args.entry == "snapshot":
        # The historical --success-only reader requires drawer/v1 amendments.
        # Substitute only its membership reader in this new-task process.
        from s4d.rwlearn.pnpv2_publication import manifest_check

        def read_manifest(path, *, success_only=False):
            if not success_only:
                raise ValueError("pick-and-place-v2 snapshot requires --success-only")
            return manifest_check(json.loads(path.read_text()), spec)

        module.read_manifest = read_manifest
    with tempfile.TemporaryDirectory(prefix="pnpv2-config-") as temp:
        if cfg is not None:
            resolved = Path(temp) / "config.yaml"
            resolved.write_text(yaml.safe_dump(cfg, sort_keys=False))
            forwarded = ["--config", str(resolved), *forwarded]
        sys.argv = [module.__file__, *forwarded]
        module.main()


def publish(spec, argv):
    """Materialize a fresh consumer-specific view, preserving one common split."""
    from s4d.rwlearn.dsreset import learning_inputs
    from s4d.rwlearn.pnpv2_publication import manifest_check, no_override
    from s4d.rwlearn.store import open_store

    sub = argparse.ArgumentParser()
    sub.add_argument("--view", choices=("rgb", "geometry"), required=True)
    sub.add_argument("--manifest", type=Path, required=True)
    sub.add_argument("--ingest-manifest", type=Path, required=True)
    sub.add_argument("--store", type=Path, required=True)
    a = sub.parse_args(argv)
    m = manifest_check(json.loads(a.manifest.read_text()), spec)
    ingest = json.loads(a.ingest_manifest.read_text())
    no_override(ingest)
    descriptor = yaml.safe_load(a.store.read_text())
    docs = learning_inputs(m, ingest, open_store(descriptor))
    p = values(spec)
    view = Path(p[a.view + "_store"]).parent
    common = {
        Path(p[key]): json.dumps(docs[name], indent=2) + "\n"
        for key, name in (("manifest", "manifest.json"), ("split", "split_seed0.json"), ("outcomes", "outcomes.json"))
    }
    if view.exists():
        raise ValueError("publication view already exists; preserve immutable outputs")
    for path, data in common.items():
        if path.exists() and path.read_text() != data:
            raise ValueError("RGB/geometry membership, split or outcomes differ")
    view.mkdir(parents=True, exist_ok=False)
    for path, data in common.items():
        if not path.exists():
            with path.open("x") as stream:
                stream.write(data)
    with (view / "store.yaml").open("x") as stream:
        stream.write(yaml.safe_dump(descriptor, sort_keys=False))
    with (view / "gate_report.json").open("x") as stream:
        stream.write(json.dumps(docs["gate_report.json"], indent=2) + "\n")


if __name__ == "__main__":
    main()
