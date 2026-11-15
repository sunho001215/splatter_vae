"""Hybrid stage authorization, no duplicate work and portable remote resume."""
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from ops.jobs import local_phase_a as a, local_phase_a_hybrid as h

EP = 'covering-failure-episode_000001'


def test_stage_contract_rejects_sam3_and_unknown_modes():
    assert not h.is_hybrid({})
    job = dict(mode=h.MODE, stages=list(h.STAGES), remote_remaining_stages=['sam3_masks'])
    assert h.is_hybrid(job) and h.suffix(job) == 'hybridA'
    for field, value in [('mode', 'typo'), ('stages', list(a.STAGES)), ('remote_remaining_stages', [])]:
        with pytest.raises(ValueError, match='stage contract'):
            h.is_hybrid({**job, field: value})


def test_full_gate_is_still_required_for_legacy_jobs(tmp_path):
    with patch.object(h, 'require_gate', return_value='subset') as subset, patch.object(a, 'require_gate', side_effect=ValueError('full')):
        with pytest.raises(ValueError, match='full'):
            a.require_job_gate(tmp_path, 'sha', {})
        subset.assert_not_called()
        assert a.require_job_gate(tmp_path, 'sha', dict(mode=h.MODE, stages=list(h.STAGES), remote_remaining_stages=['sam3_masks'])) == 'subset'


def plan(tmp_path, records):
    p = tmp_path/'plan.json'; a.atomic(p, records)
    ledger = tmp_path/'ledger.json'; a.atomic(ledger, {'jobs': {}})
    return p, ledger


def test_audit_reuses_successes_and_preserves_pending_failure(tmp_path):
    cfg = tmp_path/'cfg.yaml'; cfg.write_text('adapter: {}')
    p, ledger = plan(tmp_path, [{'episode': 'covering-success-episode_000000', 'reuse': '/accepted/compact', 'raw_sha256': 'b'*64},
                               {'episode': EP, 'config': str(cfg), 'raw_sha256': 'a'*64}])
    rows, audit = h.audit_plans([p], [ledger])
    assert len(rows) == 1 and audit['eligible'] == 1 and len(audit['reused']) == 1
    assert audit['evidence_files'][str(cfg)] == a.sha(cfg)
    with pytest.raises(ValueError, match='duplicate'):
        h.audit_plans([p, p], [ledger])


@pytest.mark.parametrize('status', ['done', 'running', 'unknown', 'launching', 'pending'])
def test_audit_never_duplicates_remote_raw_identity(tmp_path, status):
    p, ledger = plan(tmp_path, [{'episode': EP, 'config': 'unused', 'raw_sha256': 'a'*64}])
    a.atomic(ledger, {'episodes': {'legacy-name': {'input_sha256': 'a'*64, 'status': status}}})
    with pytest.raises(ValueError, match='remote ledger already owns'):
        h.audit_plans([p], [ledger])


def output(tmp_path):
    root = tmp_path/'local'; base = root/'episodes'/EP
    for stage in h.STAGES:
        tail = base/('normalized' if stage == 'ingest' else 'derived/'+stage)
        a.atomic(tail/'_COMPLETE.json', {'build_id': stage+'-immutable', 'stage': stage})
        (tail/'data.h5').write_bytes(b'payload')
    a.atomic(base/'normalized/episode.json', {'episode_id': EP, 'provenance': {'source_files': [
        {'path': '/home/ws/data/real_world/covering/failure/episode_000001.h5', 'sha256': 'a'*64}]},
        'robot': {'urdf_path': str(a.ASSETS/'panda_assets/robot.urdf')}})
    cfg = {'output_root': str(a.PROCESSED.joinpath(*a.identity(EP))), 'adapter': {'config': {'episodes': {
        EP: '/home/ws/data/real_world/covering/failure/episode_000001.h5'}}}, 'stages': {}}
    (root/'pipeline.yaml').write_text(yaml.safe_dump(cfg))
    h.finish(root, dict(episode=EP, raw_sha256='a'*64, envelope_sha256='e'*64))
    return root


def test_manifest_refuses_missing_completion_local_sam_and_drift(tmp_path):
    root = output(tmp_path)
    assert h.verify_output(root, EP)['remote_remaining_stages'] == ['sam3_masks']
    (root/'episodes'/EP/'derived/sensor_depth/data.h5').write_bytes(b'drift')
    with pytest.raises(ValueError, match='payload drift'):
        h.verify_output(root, EP)
    (root/'episodes'/EP/'derived/sam3_masks').mkdir()
    with pytest.raises(ValueError, match='contains local SAM3'):
        h.finish(root, dict(episode=EP, raw_sha256='a', envelope_sha256='b'))


def test_relocation_preserves_completion_build_ids_and_source_provenance(tmp_path):
    root = output(tmp_path)
    payloads, hashes = h.portable_payloads(root, EP)
    contract = f'episodes/{EP}/normalized/episode.json'
    assert b'/home/compu/kaist/sunho/data/real_world/' in payloads[contract]
    assert b'/home/ws/data/real_world/' not in payloads[contract]
    manifest = json.loads(payloads['hybrid-stage-manifest.json'])
    assert manifest['relocation']['original_files'][contract] == a.sha(root/contract)
    for stage in h.STAGES:
        tail = 'normalized' if stage == 'ingest' else 'derived/'+stage
        name = f'episodes/{EP}/{tail}/_COMPLETE.json'
        assert payloads[name] == root/name and hashes[name] == a.sha(root/name)
    assert (root/contract).read_bytes() != payloads[contract]  # local original untouched


def test_remote_plan_requires_all_payloads_and_requests_sam_only(tmp_path):
    root = output(tmp_path)
    processed = tmp_path/'remote'; target = processed.joinpath(*a.identity(EP))
    import shutil
    shutil.copytree(root, target)
    cfg = tmp_path/'cfg.yaml'; cfg.write_text((target/'pipeline.yaml').read_text())
    p, _ = plan(tmp_path, [{'episode': EP, 'config': str(cfg), 'raw_sha256': 'a'*64}])
    jobs = h.remote_jobs([p], tmp_path/'compiled', processed, tmp_path/'real_world')
    assert len(jobs) == 1 and jobs[0]['id'].endswith('-remoteSAM3')
    assert jobs[0]['steps'][0]['argv'][-1] == 'sam3_masks'
    assert str(target/'hybrid-stage-manifest.json') in jobs[0]['inputs']
    assert str(target/'episodes'/EP/'derived/foundation_stereo_depth/data.h5') in jobs[0]['inputs']
    (target/'episodes'/EP/'normalized/episode.json').unlink()
    with pytest.raises(ValueError, match='payload drift'):
        h.remote_jobs([p], tmp_path/'bad', processed, tmp_path/'real_world')
