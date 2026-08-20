#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-button-press-wall}"
CONFIG="${CONFIG:-config/splattervae/metaworld/temporal-no-segmentation-mask/${ENV_NAME}.yaml}"
DATASET="${DATASET:-../../data/metaworld/with-depth/${ENV_NAME}.hdf5}"
CKPT="${CKPT:-}"
OUT_DIR="${OUT_DIR:-outputs/gaussian_exports/${ENV_NAME}}"

if [[ -z "${CKPT}" ]]; then
  echo "Set CKPT to a SplatterVAE checkpoint path." >&2
  exit 1
fi

uv run python visualize/export_splattervae_gaussians.py \
  --config "${CONFIG}" \
  --dataset "${DATASET}" \
  --ckpt "${CKPT}" \
  --out_dir "${OUT_DIR}" \
  "$@"
