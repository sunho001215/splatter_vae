"""Ball-hiding identity/host adaptation of the unchanged pnp-v2 scientific row graph.

Import only after ball_hiding.bootstrap(). Geometry is materialized only after its
measured scene exists. RGB can be registered HELD before publication.
"""

import copy

from ops.jobs.pnpv2_plan import build_rows as pnp_rows
from s4d.rwlearn.pnpv2_spec import WRAPPERS

WRAPPER_ALIASES = {k.replace("scripts/pnpv2_", "runs/setup/tools/bh_"): v for k, v in WRAPPERS.items()}
TOOL = "runs/setup/tools/bh_register.py"
PAUSE = "bh-await-placement-20261120"
# Keep artifact-bearing chains on one host. Only audited readiness crosses hosts.
LOCAL_METHODS = {"ours", "ours_orig", "ours_ch01", "reviwo", "snerl", "e2e_cnn"}


def rows(spec, release, *, waves=("rgb",), scene=None, current=(), route="hybrid"):
    if route not in {"hybrid", "all-local"}:
        raise ValueError("BH route must be hybrid or all-local")
    result = copy.deepcopy(pnp_rows(spec, release, waves=waves, scene=scene))
    priority = max([5, *(float(r.get("priority", 5)) for r in current)]) + 1
    for row in result:
        row["script"] = row["script"].replace("scripts/pnpv2_", "runs/setup/tools/bh_")
        row["registration_tool"] = TOOL
        row["release_source"] = "local"  # remote also requires an exact-pin proof; never inherit ROLE_COMMIT
        row["evidence_manifest"] = "exact-ID bh-isolated grant (local) / native acceptance (remote)"
        row["priority"] = priority
        method = row["realworld_method"]
        host = "local" if route == "all-local" or "-gate-" in row["id"] or method in LOCAL_METHODS else "remote"
        row["host"] = row["pin_host"] = host
    return result
