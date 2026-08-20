#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-button-press-wall}"
DRQ_CONFIG="${DRQ_CONFIG:-agents/drqv2/config/sincro/${ENV_NAME}.yaml}"
DATASET="${DATASET:-../../data/metaworld/with-depth/${ENV_NAME}.hdf5}"
SPLATTER_CONFIG="${SPLATTER_CONFIG:-config/splattervae/metaworld/temporal-no-segmentation-mask/${ENV_NAME}.yaml}"
SINCRO_CONFIG="${SINCRO_CONFIG:-${DRQ_CONFIG}}"
OUT_DIR="${OUT_DIR:-outputs/render_quality_videos/${ENV_NAME}}"
SOURCE_CAM="${SOURCE_CAM:-cam0}"
BASE_CAMERAS="${BASE_CAMERAS:-${BASE_CAMERA:-cam0,cam1,cam2,cam3,cam4,cam5}}"
TIMESTEP="${TIMESTEP:-0}"
TRAJECTORY="${TRAJECTORY:-both}"
METHODS="${METHODS:-splattervae,sincro}"

IFS="," read -r -a BASE_CAMERA_ARRAY <<< "${BASE_CAMERAS}"

run_one() {
  local base_camera="$1"
  shift
  local source_cam="${SOURCE_CAM}"
  if [[ "${source_cam}" == "${base_camera}" ]]; then
    source_cam="${SELF_BASE_SOURCE_CAM:-cam1}"
    if [[ "${source_cam}" == "${base_camera}" ]]; then
      source_cam="cam0"
    fi
  fi
  local args=(
    --drq_config "${DRQ_CONFIG}"
    --dataset "${DATASET}"
    --source_cam "${source_cam}"
    --base_camera "${base_camera}"
    --timestep "${TIMESTEP}"
    --trajectory "${TRAJECTORY}"
    --methods "${METHODS}"
    --splatter_config "${SPLATTER_CONFIG}"
    --sincro_config "${SINCRO_CONFIG}"
    --out_dir "${OUT_DIR}/${base_camera}"
  )

  if [[ -n "${SPLATTER_CKPT:-}" ]]; then
    args+=(--splatter_ckpt "${SPLATTER_CKPT}")
  fi
  if [[ -n "${SINCRO_CKPT:-}" ]]; then
    args+=(--sincro_ckpt "${SINCRO_CKPT}")
  fi

  uv run python visualize/visualize_render_quality_trajectory_video.py "${args[@]}" "$@"
}

for base_camera in "${BASE_CAMERA_ARRAY[@]}"; do
  base_camera="${base_camera//[[:space:]]/}"
  [[ -z "${base_camera}" ]] && continue
  run_one "${base_camera}" "$@"
done
