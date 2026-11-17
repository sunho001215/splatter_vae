"""Shared body of the pick-and-place-v2 queue-row entrypoints (scripts/pnpv2_*.py).

Each entrypoint already ran the GPU guard. Here the new task is installed process-locally, a ``--config`` template
must be fully overridden by the row's ``--set`` values (a leftover ``${pnpv2.*}`` token refuses), and then the
established script's ``main`` runs with the unchanged argv.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _sets(argv):
    if "--set" not in argv:
        return []
    out = []
    for value in argv[argv.index("--set") + 1:]:
        if value.startswith("--"):
            break
        out.append(value)
    return out


def refuse_templates(argv):
    if "--config" not in argv:
        return
    from s4d.config import load_config
    from s4d.rwlearn.pnpv2_spec import leftover_tokens

    path = argv[argv.index("--config") + 1]
    left = leftover_tokens(load_config([path], _sets(argv)))
    if left:
        raise SystemExit(f"unresolved pick-and-place-v2 template fields (supply --set): {left}")


def run(module_name, argv=None):
    from s4d.rwlearn.pnpv2_runtime import install

    install()
    argv = list(sys.argv[1:] if argv is None else argv)
    refuse_templates(argv)
    module = importlib.import_module(module_name)
    sys.argv = [str(module.__file__), *argv]
    module.main()


def readiness_main(argv=None):
    """``--stage prep`` = calibration-independent RGB view, ``--stage calib`` = geometry view (resolver spellings)."""
    import argparse

    from s4d.rwlearn.pnpv2_runtime import install

    install()
    from s4d.rwlearn.pnpv2_publication import readiness
    from s4d.rwlearn.pnpv2_spec import SPEC, load_spec

    p = argparse.ArgumentParser(description=readiness_main.__doc__, allow_abbrev=False)
    p.add_argument("--stage", choices=("prep", "calib"), required=True)
    for flag in ("manifest", "split", "outcomes", "gate-report", "store", "out"):
        p.add_argument("--" + flag, required=True)
    a = p.parse_args(argv)
    result = readiness(a.stage, a.manifest, a.split, a.outcomes, a.gate_report, a.store, load_spec(ROOT / SPEC))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name("." + out.name + ".partial")
    tmp.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(tmp, out)
    print(json.dumps(result))
