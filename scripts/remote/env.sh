#!/usr/bin/env bash
set -euo pipefail

S4D_REMOTE_REPO="$(realpath -- "$(dirname -- "${BASH_SOURCE[0]}")/../..")"
S4D_REMOTE_PYTHON="${S4D_REMOTE_PYTHON:-python3}"
S4D_REMOTE_CREDENTIALS="${S4D_REMOTE_CREDENTIALS:-${HOME}/.config/s4d_remote/credentials}"
export S4D_REMOTE_REPO S4D_REMOTE_PYTHON S4D_REMOTE_CREDENTIALS
