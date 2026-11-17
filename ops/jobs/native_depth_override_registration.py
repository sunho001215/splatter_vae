"""Coordinator-only resolver alias for the additive offline native-depth entrypoint.

The worker entrypoint table stays byte-identical for persisted execution scopes.
Installing this one alias does not rewrite a queue row or choose a rig reader;
only the explicit override worker does that, after validating its evidence.
"""
OVERRIDE_SCRIPT = 'scripts/dsreset_realworld_native_depth_override.py'
LEGACY_SCRIPT = 'scripts/realworld_native_depth.py'
BASE_COMMIT = '65f896f41c33d65a367b08dbfd2c590565edb708'
TARGET = 'rwv2-20261113-prep-mvp3d_multiview-drawer-open-v2-S'


from s4d.rwlearn.dsreset_entrypoints import legacy_row as _paired_legacy_row


def legacy_row(row):
    if row.get('script') == OVERRIDE_SCRIPT:
        return {**row, 'script': LEGACY_SCRIPT}
    return _paired_legacy_row(row)


def install_resolver_alias():
    # host_roles loads before realworld_local / remote_offload import this
    # function. Keep ENTRYPOINTS unchanged: this worker does not call the
    # ordinary entry script, unlike the paired wrappers in that table.
    from s4d.rwlearn import dsreset_entrypoints
    dsreset_entrypoints.legacy_row = legacy_row
