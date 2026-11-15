#!/usr/bin/env bash
# One-step runtime rollback: no code release, job signalling or topology change.
set -euo pipefail
candidate_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
main_root="${1:-/home/ws/ws/splatter4d}"
python_bin=/home/ws/ws/splatter4d/.venv/bin/python
exec "$python_bin" "$candidate_root/runs/setup/tools/apply_local_mig16_20261115.py" --root "$main_root" --rollback
