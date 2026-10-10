#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
"$S4D_REMOTE_REPO/scripts/remote/check_tunnel.sh"
exec "$S4D_REMOTE_PYTHON" -I "$S4D_REMOTE_REPO/s4d/remote_access.py" ssh "$@"
