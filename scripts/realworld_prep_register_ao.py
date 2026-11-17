"""Phase B compile for a rig accepted for OFFLINE use by its task's acceptance override (calib.acceptance_override).

Runs the unchanged, execution-scoped ``scripts/realworld_prep_register.py`` with the same arguments (``--phase B
--rig-map M`` required). Every rig of the map is verified first (``phase_b_binding``: all cameras "validated" or
"validated_by_override"; anything else refuses like the established compiler). During that one compile the compiler's
automatic-acceptance check is answered by the override-aware check (``compiler_admits_offline_acceptance``), then every
job compiled for an override rig carries ``acceptance_override`` and its episode step runs through
``python -m s4d.realworld.calib.acceptance_override rwprep -- <same argv>`` (process-local rig-mode installation).
Jobs of automatically accepted rigs are left exactly as the established compiler wrote them.
"""
import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ops.jobs.rwprep_mig import atomic, validate  # noqa: E402
from s4d.realworld.calib.acceptance_override import (  # noqa: E402
    compiler_admits_offline_acceptance,
    phase_b_binding,
    phase_b_episode_step,
)
from s4d.realworld.calib.rig import load_rig  # noqa: E402

COMPILER = Path(__file__).resolve().parent / "realworld_prep_register.py"


def compile_phase_b(argv: list) -> int:
    if "--phase" not in argv or argv[argv.index("--phase") + 1] != "B" or "--rig-map" not in argv:
        raise SystemExit("realworld_prep_register_ao compiles phase B with a --rig-map only")
    out = Path(argv[argv.index("--out") + 1]).resolve()
    rigmap = json.loads(Path(argv[argv.index("--rig-map") + 1]).read_text())
    bindings = {}
    for path in sorted(set(rigmap.values())):
        document, _, _ = load_rig(path)
        bindings[str(Path(path).resolve())] = phase_b_binding(document, path)
    spec = importlib.util.spec_from_file_location("realworld_prep_register_compiler", COMPILER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    saved = sys.argv
    try:
        sys.argv = [str(COMPILER), *argv]
        with compiler_admits_offline_acceptance():
            module.main()
    finally:
        sys.argv = saved
    jobs = json.loads((out / "jobs.json").read_text())
    for job in jobs:
        rig_input = next(p for p in job["inputs"] if p.endswith(".json") and Path(p).with_suffix(
            ".reference.npz").as_posix() in job["inputs"])
        ref = bindings.get(rig_input)
        if ref is None:
            continue
        job["steps"][0] = phase_b_episode_step(job["steps"][0])
        job["acceptance_override"] = ref
    config = json.loads((out / "dispatcher.json").read_text())
    validate(config, jobs)
    atomic(out / "jobs.json", jobs)
    print(json.dumps({"acceptance_override_jobs": sum("acceptance_override" in j for j in jobs), "jobs": len(jobs)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(compile_phase_b(sys.argv[1:]))
