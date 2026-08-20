#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-button-press-wall}"
CONFIG="${CONFIG:-config/splattervae/metaworld/temporal-no-segmentation-mask/${ENV_NAME}.yaml}"
DATASET="${DATASET:-../../data/metaworld/with-depth/${ENV_NAME}.hdf5}"
CKPT="${CKPT:-}"
OUT="${OUT:-outputs/splattervae_tsne/${ENV_NAME}_tsne.png}"

if [[ -z "${CKPT}" ]]; then
  echo "Set CKPT to a SplatterVAE checkpoint path." >&2
  exit 1
fi

uv run python visualize/visualize_splattervae_tsne.py \
  --config "${CONFIG}" \
  --dataset "${DATASET}" \
  --ckpt "${CKPT}" \
  --out "${OUT}" \
  "$@"
