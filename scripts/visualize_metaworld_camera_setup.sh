#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-button-press-wall}"
CONFIG="${CONFIG:-agents/drqv2/config/sincro/${ENV_NAME}.yaml}"
OUT="${OUT:-outputs/camera_setup/${ENV_NAME}_camera_setup.png}"

uv run python visualize/visualize_metaworld_camera_setup.py \
  --config "${CONFIG}" \
  --out "${OUT}" \
  "$@"
