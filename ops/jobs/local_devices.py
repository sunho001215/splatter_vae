"""Local device eligibility for the row's pinned release and actual entry-point checkout.

Read literal allowlists without importing old code or CUDA. A refusal stays pending;
no attempt, registry event or process is created. User 2026-11-15 MIG expansion.
"""
from __future__ import annotations

import ast
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
    if pin and gpu not in _release_devices(Path(root).resolve(), str(pin)):
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
