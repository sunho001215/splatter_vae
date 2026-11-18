"""Opt-in PNP offline acceptance. Execution-scoped RGB spec/publication/readers stay byte-identical.

The task spec still requires the original criteria and permits no user-supplied override. Only the committed AO
record, verified against the rig's complete automatic evidence, can authorize offline geometry. New geometry gate
and native-depth entrypoints install these readers in their own process; deployment never imports this module.
"""
from __future__ import annotations

import contextlib
import re
from pathlib import Path

from s4d.realworld.calib import pnpv2_acceptance_override as AO
from s4d.rwlearn.pnpv2_spec import TASK
from s4d.rwlearn.pnpv2_publication import readiness as _readiness

OFFLINE_WRAPPERS = {
    'scripts/pnpv2_offline_check.py': 'scripts/realworld_dsreset_check.py',
    'scripts/pnpv2_offline_native_depth.py': 'scripts/realworld_native_depth.py',
}
ROW_WRAPPERS = dict(zip(('scripts/pnpv2_realworld_dsreset_check.py',
                         'scripts/pnpv2_realworld_native_depth.py'), OFFLINE_WRAPPERS))


def check_reference(ref):
    """Validate the full downstream AO summary, or release's four-field evidence reference, against this record."""
    found = AO.record_for(TASK)
    if found is None:
        raise ValueError('pick-and-place-v2 has no committed calibration acceptance override')
    rec, sha, _ = found
    expected = dict(id=rec['id'], record_sha256=sha,
                    automatic_decision_sha256=rec['automatic_acceptance']['decision_sha256'])
    if not isinstance(ref, dict) or any(ref.get(k) != v for k, v in expected.items()):
        raise ValueError('pick-and-place-v2 acceptance override differs from the committed record')
    if not re.fullmatch('[0-9a-f]{64}', str(ref.get('override_decision_sha256', ''))):
        raise ValueError('acceptance override lacks its decision hash')
    if set(ref) == {*expected, 'override_decision_sha256'}:
        return
    expected.update(task=TASK, status=AO.STATUS, robot_execution_authorized=False,
                    offline_uses=list(AO.OFFLINE_USES), cameras=sorted(rec['cameras']),
                    exceptions=[e['id'] for e in rec['exceptions']],
                    rig=f"rig-loma-{TASK}-{rec['automatic_acceptance']['criteria_commit'][:7]}-c2"
                        + rec['document_name_suffix'])
    if set(ref) != {*expected, 'override_decision_sha256'} or any(ref.get(k) != v for k, v in expected.items()):
        raise ValueError('pick-and-place-v2 acceptance override scope differs from the committed record')


def no_override(obj):
    """Automatic inputs or exact committed PNP references only; arbitrary override blocks still refuse."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in ('acceptance_override', 'acceptance_overrides') and value:
                for ref in value if isinstance(value, list) else [value]:
                    check_reference(ref)
            else:
                no_override(value)
    elif isinstance(obj, list):
        for value in obj:
            no_override(value)


def rig_gate(path):
    from s4d.realworld.calib.rig import load_rig, rig_camera_status

    doc, _, _ = load_rig(path)
    if doc.get('acceptance_override') is None:
        # Preserve the established automatic route (including synthetic rigs without a decision file).
        if doc.get('task') != TASK or not all(rig_camera_status(doc, c)[0] == 'validated'
                                             for c in doc['cameras']):
            raise ValueError('rig must be automatically accepted for the complete task')
        return dict(mode='automatic', acceptance_override=None)
    try:
        gate = AO.offline_gate(path, TASK)
    except ValueError as exc:
        raise ValueError(f"acceptance override refused: {exc}") from exc
    if gate['mode'] != 'acceptance_override':
        raise ValueError('invalid PNP offline rig gate')
    check_reference(gate['acceptance_override'])
    return gate


def phase_b_compiler(rig, override):
    if override:
        if AO.offline_gate(rig, TASK)['mode'] != 'acceptance_override':
            raise ValueError('Phase B requires the committed offline override')
        return 'scripts/realworld_prep_register_ao.py'
    return 'scripts/realworld_prep_register.py'


@contextlib.contextmanager
def publication_readers():
    """One call into the unchanged publication module; propagate calibration references like drawer inputs."""
    from s4d.rwlearn import dsreset, pnpv2_publication as publication

    previous_check, previous_inputs = publication.no_override, dsreset.learning_inputs

    def learning_inputs(manifest, ingest, store):
        docs = previous_inputs(manifest, ingest, store)
        from scripts.realworld_dsreset_inputs import acceptance_overrides
        refs = acceptance_overrides(manifest, ingest)
        if refs:
            for ref in refs:
                check_reference(ref)
            docs['gate_report.json']['acceptance_override'] = refs
        return docs

    publication.no_override, dsreset.learning_inputs = no_override, learning_inputs
    try:
        yield publication
    finally:
        publication.no_override, dsreset.learning_inputs = previous_check, previous_inputs


def build_geometry(*args, **kwargs):
    with publication_readers() as publication:
        return publication.build_geometry(*args, **kwargs)


def readiness(*args, **kwargs):
    with publication_readers() as publication:
        result = _readiness(*args, **kwargs)
    import json
    report = json.loads(Path(args[4] if len(args) > 4 else kwargs['gate_path']).read_text())
    refs = report.get('acceptance_override')
    if refs:
        result['acceptance_override'] = refs
        result['effective_rig_status'] = AO.STATUS
    return result


def readiness_main():
    # The old entrypoint imports readiness at call time; install only for this new geometry entrypoint.
    from s4d.rwlearn import pnpv2_entry, pnpv2_publication
    previous = pnpv2_publication.readiness
    pnpv2_publication.readiness = readiness
    try:
        pnpv2_entry.readiness_main()
    finally:
        pnpv2_publication.readiness = previous


def native_main():
    from s4d.rwlearn.pnpv2_runtime import install
    install()
    import scripts.realworld_native_depth as entry
    from s4d.rwlearn.pnpv2_native_depth_override import build_native_depth_override
    from s4d.rwlearn import native_depth_mapped

    previous, mapped = entry.build_native_depth, native_depth_mapped.build_native_depth_mapped
    entry.build_native_depth = build_native_depth_override
    native_depth_mapped.build_native_depth_mapped = build_native_depth_override
    try:
        entry.main()
    finally:
        entry.build_native_depth, native_depth_mapped.build_native_depth_mapped = previous, mapped
