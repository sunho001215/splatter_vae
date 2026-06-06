#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-button-press-wall}"
CONFIG="${CONFIG:-config/metaworld/${ENV_NAME}.yaml}"
DATASET="${DATASET:-../../data/metaworld/with-depth/${ENV_NAME}.hdf5}"
CKPT="${CKPT:-}"
OUT_DIR="${OUT_DIR:-outputs/splattervae_view_swap/${ENV_NAME}}"

if [[ -z "${CKPT}" ]]; then
  echo "Set CKPT to a SplatterVAE checkpoint path." >&2
  exit 1
fi

uv run python visualize/visualize_splattervae_view_swap_cross_source.py \
  --config "${CONFIG}" \
  --dataset "${DATASET}" \
  --ckpt "${CKPT}" \
  --out_dir "${OUT_DIR}" \
  "$@"
