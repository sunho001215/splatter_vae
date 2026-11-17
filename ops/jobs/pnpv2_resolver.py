"""Coordinator-only resolver aliases for the pick-and-place-v2 entrypoints (``scripts/pnpv2_*.py``).

Each pnpv2 entrypoint installs the new task process-locally and then runs the established script with the
established argv, so input resolution, placement and closeout treat it as that script. Installed by host_roles
before realworld_local / remote_offload bind ``legacy_row``; the worker entrypoint table stays unchanged.
"""
from s4d.rwlearn.pnpv2_spec import WRAPPERS


def install_resolver_alias():
    from s4d.rwlearn import dsreset_entrypoints

    previous = dsreset_entrypoints.legacy_row
    if getattr(previous, "pnpv2_alias", False):
        return

    def legacy_row(row):
        script = WRAPPERS.get(row.get("script"))
        return previous({**row, "script": script}) if script else previous(row)

    legacy_row.pnpv2_alias = True
    dsreset_entrypoints.legacy_row = legacy_row
