#!/usr/bin/env bash
set -euo pipefail
repo="$(realpath -- "$(dirname -- "${BASH_SOURCE[0]}")/../..")"
python="${S4D_REMOTE_PYTHON:-$repo/.venv/bin/python}"
exec "$python" -I -c 'import runpy, sys; sys.path.insert(0, sys.argv.pop(1)); runpy.run_module("s4d.remote_launch", run_name="__main__")' "$repo" "$@"
