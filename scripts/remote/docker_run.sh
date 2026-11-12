#!/usr/bin/env bash
# The Python launcher preserves 20 (verified pre-create refusal) and 75 (unknown outcome).
set -euo pipefail
repo="$(realpath -- "$(dirname -- "${BASH_SOURCE[0]}")/../..")"
python="${S4D_REMOTE_PYTHON:-$repo/.venv/bin/python}"
exec "$python" -I -c 'import runpy, sys; sys.path.insert(0, sys.argv.pop(1)); runpy.run_module("ops.containers.launch", run_name="__main__")' "$repo" "$@"
