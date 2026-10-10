#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
exec "$S4D_REMOTE_PYTHON" -I "$S4D_REMOTE_REPO/s4d/remote_access.py" check-tunnel "$@"
