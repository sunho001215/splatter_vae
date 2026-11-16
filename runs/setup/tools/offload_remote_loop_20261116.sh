#!/usr/bin/env bash
# Dry-run by default. Pass --execute explicitly after adopting on the coordinator.
# STOP is checked between rows/passes; an in-progress upload finishes safely.
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
stop_file="${OFFLOAD_STOP_FILE:-$repo/runs/setup/STOP-OFFLOAD-REMOTE}"
interval="${OFFLOAD_INTERVAL_SECONDS:-60}"
python_bin="${OFFLOAD_PYTHON:-$repo/.venv/bin/python}"
cd -- "$repo"
while [[ ! -e "$stop_file" ]]; do
    if ! "$python_bin" -B runs/setup/tools/offload_remote_20261116.py --stop-file "$stop_file" "$@"; then
        printf '%s\n' 'Offload pass failed; retrying after the interval (STOP still applies).' >&2
    fi
    for ((elapsed=0; elapsed<interval; elapsed++)); do
        [[ ! -e "$stop_file" ]] || exit 0
        sleep 1
    done
done
