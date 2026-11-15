#!/usr/bin/env bash
set -euo pipefail
/usr/bin/python3 -I -B - "$@" <<'PY'
import json, os, signal, sys
from pathlib import Path
pid = 574928
expected_start = '114490525'
expected_cwd = '/home/ws/ws/splatter4d'
expected_argv = ['.venv/bin/python', '-I', 'runs/setup/tools/register_local_phase_a_20261115.py', '--local-subset', '--bundle', 'runs/hybrid-phase-a-20261115-release-r2', '--plan', '/home/ws/ws/realworld-ops/dsreset-20261113/pipeline/drawer-canonical-manifests/prep-plan.json', '--plan', '/home/ws/ws/realworld-ops/dsreset-20261113/pipeline/covering-canonical-manifests/prep-plan.json', '--plan', '/home/ws/ws/realworld-ops/dsreset-20261113/pipeline/pick-and-place-canonical-manifests/prep-plan.json', '--execute']
p = Path('/proc')/str(pid)
if not p.exists():
    print('registrar already gone; refresh audit and reconcile publication before proceeding'); raise SystemExit(0)
pidfd = os.pidfd_open(pid)
actual_argv = [s.decode() for s in (p/'cmdline').read_bytes().split(b'\0') if s]
actual_start = (p/'stat').read_text().rsplit(')',1)[1].split()[19]
if actual_start != expected_start or actual_argv != expected_argv or str((p/'cwd').resolve()) != expected_cwd:
    raise SystemExit('PID identity changed; refuse to signal anything')
if sys.argv[1:] not in ([], ['--execute']):
    raise SystemExit('usage: stop-cancelled-registrar.sh [--execute]')
print(json.dumps({'pid':pid,'argv':actual_argv,'execute':bool(sys.argv[1:]),'signal':'SIGTERM'}))
if sys.argv[1:]:
    signal.pidfd_send_signal(pidfd, signal.SIGTERM)
    print('registrar signalled; refresh audit, inspect partial publication, preserve all existing results')
os.close(pidfd)
PY
