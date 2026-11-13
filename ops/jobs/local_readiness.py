"""Read-only owning-driver admission gates shared by cross-driver reservations.

Campaign entry modules execute operations on import. This module reads only their
registration/hold evidence and exposes the same readiness predicates without
importing either entry module. Construct once per pass, after reading the registry.
"""
import ast
import json
from datetime import datetime
from pathlib import Path

STAGE2_REGISTRATIONS = (
    'stage2-registration-20261013.json', 'stage2-crop-registration-20261015.json',
    'stage2-nosynth-rl-registration-20261016.json', 'stage2-extra-rl-registration-20261016.json',
    'followup-registration-20261017.json', 'crop-reeval-registration-20261015.json',
)
MIDDLE_WORK = ('scripts/train.py', 'scripts/train_rl.py', 'scripts/eval_rl.py')


class LocalReadiness:
    def __init__(self, queue, state, root, *, now=None):
        self.root, self.state = Path(root), state
        self.rows = {j['id']: j for j in queue['jobs']}
        self.now = now or datetime.now().astimezone()
        self.stage2_ids, self.original = set(), set()
        for name in STAGE2_REGISTRATIONS:
            rows = self.read(name).get('jobs', [])
            if name == STAGE2_REGISTRATIONS[0]:
                self.original = {j['id'] for j in rows}
            self.stage2_ids.update(j['id'] for j in rows
                                  if name != 'followup-registration-20261017.json' or j.get('stage') == 2)
        self.stage2_ids.update(j['id'] for j in queue['jobs'] if j.get('stage') == 2)
        # Read literal registration declarations; never import/execute the entry module.
        declarations = ast.parse((Path(__file__).resolve().parents[1] / 'campaign/baseline_local_driver.py').read_text())
        constants = {n.targets[0].id: ast.literal_eval(n.value) for n in declarations.body
                     if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                     and n.targets[0].id in ('REGISTRATIONS', 'ONLY')}
        self.baseline_known, self.baseline_allowed = set(), set()
        for name in constants['REGISTRATIONS']:
            ids = {j['id'] for j in self.read(name).get('jobs', [])}
            self.baseline_known |= ids
            self.baseline_allowed |= ids & constants['ONLY'][name] if name in constants['ONLY'] else ids
        self.frozen = set(self.read('cnn-freeze-20261024.json').get('frozen', []))
        self.hold = self.read('stage2-admission-hold.json')
        self.original_rl_pending = any(self.pending(j) for i, j in self.rows.items()
                                       if i in self.original and j['script'] in MIDDLE_WORK[1:])
        self.middle_pretrain_pending = any(self.pending(j) for i, j in self.rows.items()
                                          if i in self.stage2_ids and j.get('stage2_admission_rank') == 'middle'
                                          and j['script'] == 'scripts/train.py')

    def read(self, name):
        path = self.root / 'runs/setup' / name
        return json.loads(path.read_text()) if path.is_file() else {}

    def pending(self, job):
        current = self.state.get(job['id'], {})
        return (current.get('status', 'pending') == 'pending'
                or (current.get('status') == 'crashed'
                    and current.get('failures', 0) <= int(job.get('max_restarts', 1))))

    def eligible_state(self):
        """Use each row's own restart limit, including rows owned by another driver."""
        return {i: {**s, 'status': 'pending' if i in self.rows and self.pending(self.rows[i]) else s['status']}
                for i, s in self.state.items()}

    def stage2_admissible(self, job):
        script = job['script']
        middle = job.get('stage2_admission_rank') == 'middle'
        if middle and script in MIDDLE_WORK and self.original_rl_pending:
            return False
        if (script == 'scripts/train.py' and not middle and job.get('priority', 5) != 0
                and self.middle_pretrain_pending):
            return False
        if script == 'scripts/evaluate.py':
            return (self.root / 'runs' / job['id'] / 'config.yaml').is_file()
        if script in MIDDLE_WORK[1:] and next(iter(job.get('deps', [])), None) in self.stage2_ids:
            return (self.root / 'runs/setup' / f"s2-{job['deps'][0]}-completion.json").is_file()
        return True

    def owns(self, job):
        return job['id'] in self.stage2_ids | self.baseline_known

    def __call__(self, job):
        if job['id'] in self.baseline_known - self.baseline_allowed:
            return False
        if job['id'].removesuffix('-eval') in self.frozen or job.get('pin_host') == 'remote':
            return False
        if job['id'] in self.stage2_ids:
            if not self.stage2_admissible(job):
                return False
            if self.hold and (job['id'] not in self.hold['allow']
                              or self.now >= datetime.fromisoformat(self.hold['admit_until'])):
                return False
        # Both local drivers withhold companions until their trainer is executing
        # or done. A waiting evaluator cannot reserve capacity for itself.
        if job['script'] == 'scripts/eval_rl.py':
            trainer = job.get('reevaluates') or job['id'].removesuffix('-eval')
            if self.state.get(trainer, {}).get('status') not in ('running', 'launching', 'sync_pending', 'done'):
                return False
        return True
