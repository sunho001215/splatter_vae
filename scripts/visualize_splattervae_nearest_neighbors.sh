#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-button-press-wall}"
CONFIG="${CONFIG:-config/splattervae/metaworld/temporal/${ENV_NAME}.yaml}"
DATASET="${DATASET:-../../data/metaworld/with-depth/${ENV_NAME}.hdf5}"
CKPT="${CKPT:-}"
OUT="${OUT:-outputs/splattervae_nearest_neighbors/${ENV_NAME}_nearest_neighbors.png}"
CAMERAS="${CAMERAS:-cam0,cam1,cam2,cam3,cam4,cam5}"

if [[ -z "${CKPT}" ]]; then
  echo "Set CKPT to a SplatterVAE checkpoint path." >&2
  exit 1
fi

uv run python visualize/visualize_splattervae_nearest_neighbors.py \
  --config "${CONFIG}" \
  --dataset "${DATASET}" \
  --ckpt "${CKPT}" \
  --out "${OUT}" \
  --cameras "${CAMERAS}" \
  "$@"
