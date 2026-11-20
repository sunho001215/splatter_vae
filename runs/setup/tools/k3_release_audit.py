"""Read-only audit of the immutable k0 release boundary; never deploys or unholds.

Run with the main venv, CUDA hidden and nice/ionice, from any checkout.
Only --out writes, and it must name a new directory. No production imports.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile

MAIN = Path('/home/ws/ws/splatter4d')
PRIOR = Path('/home/ws/ws/realworld-ops/k0-20261120')
COMMIT = 'ffedec5464ba10b013f5071603b7bd6bc517b937'
IMAGE = 'sha256:59082834c2c2c0d688be50a0b3b362bc46bdd2b1923d8ab712a473a0bd51870a'
TASKS = ('drawer-open-v2', 'covering', 'pick-and-place-v2')
IDS = tuple(f'rwk0-20261120-{stage}-ours-{task}-S-orig-k0-400k'
            for task in TASKS for stage in ('pre', 'export', 'eval'))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def git(*args):
    return subprocess.run(['git', '-C', str(MAIN), *args], capture_output=True,
                          check=True, timeout=30).stdout


def function(source, name, namespace):
    """Execute only one actual function AST; no entrypoint imports or side effects."""
    node = next(n for n in ast.parse(source).body
                if isinstance(n, ast.FunctionDef) and n.name == name)
    module = ast.Module(body=[node], type_ignores=[])
    exec(compile(module, '<actual-release-function>', 'exec'), namespace)
    return namespace[name]


def probe_gate(source, payload, fingerprints):
    with tempfile.TemporaryDirectory(prefix='k3-gate-') as temp:
        directory = Path(temp)
        (directory / 'tests.json').write_text(json.dumps(payload))
        gate = function(source, 'require_passed_tests', {
            'json': json, 'test_evidence_directory': lambda: directory,
            'source_fingerprints': lambda: fingerprints,
        })
        try:
            gate()
        except RuntimeError as exc:
            return str(exc)
        return 'accepted'


def native_summary():
    root = PRIOR / 'evidence/k2-native-01/results'
    summary = json.loads((root / 'summary.json').read_bytes())
    environment = json.loads((root / 'environment.json').read_bytes())
    assert summary['complete'] is True and summary['cpu_dry'] is False
    assert environment['commit'] == COMMIT and environment['cpu_dry'] is False
    counts, artifacts = {}, {}
    for task in TASKS:
        counts[task] = {}
        for mode, wanted in (('mixed', 20), ('k0', 10), ('k1', 10)):
            path = root / f'{task}-{mode}.jsonl'
            data = path.read_bytes()
            rows = [json.loads(line) for line in data.splitlines() if line]
            assert len(rows) == wanted
            assert [r['step'] for r in rows] == list(range(wanted))
            for row in rows:
                values = [*row['losses'].values(), row['grad_norm_preclip'],
                          *row['grad_norms_postclip'].values()]
                assert all(math.isfinite(v) for v in values)
                assert len(row['targets']['strides']) == 16
                if row['targets']['k0_samples']:
                    assert row['targets']['candidate_max_abs'] == 0
                    assert row['targets']['motion_score_max_abs'] == 0
                    assert row['targets']['candidate_valid'] > 0
            counts[task][mode] = wanted
            artifacts[str(path)] = sha(data)
    for path in (root / 'summary.json', root / 'environment.json'):
        artifacts[str(path)] = sha(path.read_bytes())
    return {'complete': True, 'optimizer_updates': sum(sum(v.values()) for v in counts.values()),
            'counts': counts, 'finite_losses_and_gradients': True, 'batch_size': 16,
            'environment': environment, 'artifacts_sha256': artifacts}


def processes():
    """Keep only scheduler/driver commands; never emit a shell's exported environment."""
    found = []
    labels = ('remote_campaign_driver', 'stage2_local_driver', 'baseline_local_driver',
              'realworld-closeout', 'realworld_local.py closeout')
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args = path.read_bytes().decode().split('\0')[:-1]
            if not args:
                continue
            executable = Path(args[0]).name
            text = ' '.join(args)
            if executable in ('bash', 'sh') and '-c' in args:
                command = args[args.index('-c') + 1]
                # The parent launcher has environment exports before eval. Discard all of them.
                command = command[command.rfind('while true'):]
                end = command.find('done')
                if end < 0 or not any(label in command[:end] for label in labels):
                    continue
                safe = command[:end + 4]
                kind = 'long-lived shell; child executable rerun each iteration'
            elif executable.startswith('python') and ('ops/jobs/scheduler.py' in text or
                    any(label in text for label in labels) or
                    ('scripts/jobs.py' in text and 'daemon' in args)):
                safe = ' '.join(a for a in args if a in ('-I', '-m', 'daemon', 'closeout', '--loop')
                                or a.endswith('.py') or a.startswith('ops.jobs'))
                kind = 'scheduler daemon' if 'daemon' in args else 'per-cycle Python child'
            else:
                continue
            stat = (path.parent / 'stat').read_text().rsplit(')', 1)[1].split()
            found.append({'pid': int(path.parent.name), 'ppid': int(stat[1]),
                          'start_ticks': int(stat[19]), 'kind': kind, 'command': safe})
        except (OSError, ValueError, IndexError):
            continue
    return sorted(found, key=lambda item: item['pid'])


def audit():
    import yaml
    # Read complete prior analysis without relying on its conclusion.
    prior_analysis = json.loads((PRIOR / 'evidence/k2-release-analysis.json').read_bytes())
    raw = (MAIN / 'experiments/queue.yaml').read_bytes()
    queue = yaml.load(raw, Loader=yaml.CSafeLoader)
    rows = {r['id']: r for r in queue['jobs'] if r['id'] in IDS}
    assert set(rows) == set(IDS)
    assert {r['priority'] for r in rows.values()} == {5}
    assert {r['planned_commit'] for r in rows.values()} == {COMMIT}
    assert {r['planned_image'] for r in rows.values()} == {IMAGE}
    pauses = json.loads((MAIN / 'runs/setup/admission-pauses.json').read_bytes())
    hold = [p for p in pauses['pauses'] if p.get('name') == 'k0-20261120-held']
    assert len(hold) == 1 and hold[0]['active'] is True
    assert set(hold[0]['ids']) == set(IDS) and not hold[0].get('prefixes')
    source = git('show', f'{COMMIT}:scripts/_bootstrap.py').decode()
    current = {'sentinel': 'exact-source'}
    targeted = {'acceptance_kind': 'exact-id-targeted', 'ids': list(IDS),
                'commit': COMMIT, 'exit_code': 0, 'counts': {
                    'passed': 13, 'failed': 0, 'error': 0, 'skipped': 0},
                'source_sha256': current}
    existing = {**targeted, 'full_suite_passed': True}
    gate = {'targeted_evidence': probe_gate(source, targeted, current),
            'synthetic_full_suite_positive_control': probe_gate(source, existing, current),
            'synthetic_stale_suite_control': probe_gate(source, existing, {'changed': 'source'})}
    assert gate['synthetic_full_suite_positive_control'] == 'accepted'
    assert gate['targeted_evidence'] != 'accepted'
    assert gate['synthetic_stale_suite_control'] != 'accepted'
    native = native_summary()
    original_logs = {}
    for name in ('tests-k0-final.log', 'tests-existing.log', 'tests-workers.log', 'tests-loss.log'):
        path = PRIOR / 'evidence' / name
        data = path.read_bytes()
        original_logs[str(path)] = {'sha256': sha(data), 'tail': data.decode().splitlines()[-1]}
    acceptance = {
        'schema_version': 1, 'kind': 'k3-exact-id-targeted-evidence-inventory',
        'commit': COMMIT, 'image_digest': IMAGE, 'ids': list(IDS),
        'policy_owner_decision': 'Approve releasing all three k=0 pretraining experiments (drawer-open-v2, covering, '
            'pick-and-place-v2) using isolated, version-pinned code and targeted tests. Retain the original '
            'architecture, losses, schedules, and FQL. Admit training only through normal scheduling, strictly '
            'behind existing work, without global re-pinning, expensive full-suite tests, or admission delays.',
        'targeted_scientific_evidence_verified': True,
        'scheduler_acceptance': False, 'deployment_authorization': False,
        'reason': 'Unmodified ffedec5 runtime has no exact-ID targeted-evidence reader. No full-suite claim.',
        'rows': {i: {'row_sha256': sha(json.dumps(rows[i], sort_keys=True, separators=(',', ':')).encode()),
                     'row': rows[i]} for i in IDS},
        'native': native, 'inherited_targeted_logs': original_logs,
        'prior_analysis_sha256': sha((PRIOR / 'evidence/k2-release-analysis.json').read_bytes()),
    }
    return {
        'main_head': git('rev-parse', 'HEAD').decode().strip(),
        'queue_sha256_at_read': sha(raw), 'nine_rows_held_priority_5': True,
        'unrelated_rows': len(queue['jobs']) - len(IDS),
        'immutable_runtime_gate_probes': gate, 'processes': processes(),
        'prior_analysis_keys_read': sorted(prior_analysis),
        'source_fingerprint_roots': ['mujoco_mig_setup.py', 's4d', 'ops', 'scripts', 'tests', 'configs'],
        'per_cycle_wrappers_excluded_from_fingerprint': True,
        'runtime_or_main_mutations': [], 'acceptance_record': acceptance,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = audit()
    args.out.mkdir(parents=True, exist_ok=False)
    acceptance = result.pop('acceptance_record')
    for name, payload in (('release-audit.json', result), ('exact-nine-targeted-evidence.json', acceptance)):
        (args.out / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'out': str(args.out), 'ids': len(IDS), 'native_updates': acceptance['native']['optimizer_updates'],
                      'scheduler_acceptance': False, 'processes': len(result['processes'])}))


if __name__ == '__main__':
    main()
