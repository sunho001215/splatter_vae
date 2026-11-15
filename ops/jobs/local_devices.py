"""Local device eligibility for the row's pinned release and actual entry-point checkout.

Read literal allowlists without importing old code or CUDA. A refusal stays pending;
no attempt, registry event or process is created. User 2026-11-15 MIG expansion.
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
from functools import lru_cache
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]


def _allowlist(guard: str, config: str) -> frozenset[str]:
    tree = ast.parse(guard)
    assignment = next(n for n in tree.body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'APPROVED_HOST_GPUS' for t in n.targets))
    mapping = assignment.value
    if isinstance(mapping, ast.Call):
        mapping = mapping.args[0]
    values = next(v for k, v in zip(mapping.keys, mapping.values) if ast.literal_eval(k) == 'local')
    approved = frozenset(ast.literal_eval(values))
    host = yaml.safe_load(config)
    if (host['name'] != 'local' or not isinstance(host['gpu_uuids'], list)
            or len(host['gpu_uuids']) != len(approved) or set(host['gpu_uuids']) != approved
            or host.get('runtime_root') is not None):
        return frozenset()
    return approved


@lru_cache(maxsize=256)
def _release_devices(root: Path, commit: str) -> frozenset[str]:
    if not re.fullmatch(r'[0-9a-f]{7,40}', commit):
        return frozenset()
    def read(name):
        return subprocess.run(['git', '-C', str(root), 'show', f'{commit}:{name}'],
                              check=True, capture_output=True, text=True, timeout=10).stdout
    try:
        return _allowlist(read('ops/resources/gpu_guard.py'), read('configs/hosts/local.yaml'))
    except (OSError, ValueError, SyntaxError, KeyError, StopIteration, AttributeError, TypeError, IndexError, yaml.YAMLError,
            subprocess.SubprocessError):
        return frozenset()


def execution_root(job: dict, root: Path = REPO) -> Path:
    """The nearest checkout containing the resolved script (including .worktrees/.../scripts)."""
    root = Path(root).resolve()
    path = (root / job.get('script', '')).resolve()
    if path == root:
        return root
    for parent in path.parents:
        if parent == root or (parent / 'ops/resources/gpu_guard.py').is_file():
            return parent
    return root


def identical_placed_source(job: dict, root: Path) -> bool:
    """Reuse the existing local-placement scope proof without changing the row's pin.

    Only an explicitly placed local row qualifies. Its recorded scope was proved
    against planned_commit by realworld_local. Hash the actual checkout again;
    a different script/config/upstream or an incomplete record fails closed.
    Normal admission still checks ownership, pauses, state and all inputs.
    """
    from ops.provenance.upstreams import working_tree_identity

    try:
        identity = job['id']
        if (job.get('host', 'local') != 'local' or Path(identity).name != identity
                or identity in ('.', '..') or execution_root(job, root) != root.resolve()):
            return False
        record = json.loads((root / 'runs/setup/ownership' / (identity + '.json')).read_bytes())
        scope = record['execution_scope']
        files = scope['files']
        if (record.get('job_id') != identity or record.get('kind') != 'role_placement'
                or record.get('state') != 'placed' or record.get('owner_host') != 'local'
                or not job.get('planned_commit') or scope.get('source_commit') != job['planned_commit']
                or not isinstance(files, dict) or not files or job.get('script') not in files):
            return False
        # Include every repository argument checked at original placement, as well
        # as the script. Changing argv cannot borrow another script/config proof.
        for arg in job.get('args', []):
            value = str(arg).partition('=')[2] if '=' in str(arg) else str(arg)
            path = Path(value)
            if path.is_absolute():
                try:
                    value = str(path.relative_to(root))
                except ValueError:
                    continue
            if value.startswith(('configs/', 'scripts/', 's4d/', 'ops/')) and value not in files:
                return False
        upstreams = working_tree_identity(root)
        for name, digest in files.items():
            path = (root / name).resolve()
            if '@' in name:
                actual = upstreams.get(name)
            elif root.resolve() not in path.parents or not path.is_file():
                return False
            else:
                actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != digest:
                return False
        # An empty upstream subset cannot waive newly added or changed gitlinks.
        if any(files.get(name) != digest for name, digest in upstreams.items()):
            return False
        return True
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError):
        return False


def allows(job: dict, gpu: str, root: Path = REPO) -> bool:
    if job.get('host', 'local') != 'local':
        return True
    from ops.resources.gpu_guard import APPROVED_HOST_GPUS

    # Pure planners also accept abstract slot names. Real queue validation rejects them.
    if not gpu.startswith(('MIG-', 'GPU-')):
        return True
    if gpu not in APPROVED_HOST_GPUS['local']:
        return False
    pin = job.get('planned_commit')
    if pin:
        devices = _release_devices(Path(root).resolve(), str(pin))
        if not devices or (gpu not in devices and not identical_placed_source(job, Path(root).resolve())):
            return False
    checkout = execution_root(job, root)
    if checkout != Path(root).resolve():
        try:
            return gpu in _allowlist((checkout / 'ops/resources/gpu_guard.py').read_text(),
                                     (checkout / 'configs/hosts/local.yaml').read_text())
        except (OSError, ValueError, SyntaxError, KeyError, StopIteration, AttributeError, TypeError, IndexError,
                yaml.YAMLError):
            return False
    return True
