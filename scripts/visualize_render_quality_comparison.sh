#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-button-press-wall}"
DATASET="${DATASET:-../../data/metaworld/with-depth/${ENV_NAME}.hdf5}"
SPLATTER_CONFIG="${SPLATTER_CONFIG:-config/splattervae/metaworld/temporal/${ENV_NAME}.yaml}"
SINCRO_CONFIG="${SINCRO_CONFIG:-agents/drqv2/config/sincro/${ENV_NAME}.yaml}"
REVIWO_CONFIG="${REVIWO_CONFIG:-agents/drqv2/config/reviwo/${ENV_NAME}.yaml}"
OUT_DIR="${OUT_DIR:-outputs/render_quality/${ENV_NAME}}"
OUT="${OUT:-}"
SOURCE_CAM="${SOURCE_CAM:-cam0}"
CAMERAS="${CAMERAS:-cam0,cam1,cam2,cam3,cam4,cam5}"
TARGET_CAMS="${TARGET_CAMS:-${TARGET_CAM:-${CAMERAS}}}"
TIMESTEP="${TIMESTEP:-0}"
METHODS="${METHODS:-splattervae,sincro,reviwo}"
REQUIRE_LPIPS="${REQUIRE_LPIPS:-1}"

IFS="," read -r -a TARGET_CAM_ARRAY <<< "${TARGET_CAMS}"

run_one() {
  local target_cam="$1"
  shift
  local source_cam="${SOURCE_CAM}"
  if [[ "${source_cam}" == "${target_cam}" ]]; then
    source_cam="${SELF_TARGET_SOURCE_CAM:-cam1}"
    if [[ "${source_cam}" == "${target_cam}" ]]; then
      source_cam="cam0"
    fi
  fi
  local out_path="${OUT}"
  if [[ -z "${out_path}" || "${#TARGET_CAM_ARRAY[@]}" -gt 1 ]]; then
    out_path="${OUT_DIR}/${source_cam}_to_${target_cam}_comparison.png"
  fi

  local args=(
    --dataset "${DATASET}"
    --source_cam "${source_cam}"
    --target_cam "${target_cam}"
    --timestep "${TIMESTEP}"
    --num_scenes "${NUM_SCENES:-3}"
    --methods "${METHODS}"
    --splatter_config "${SPLATTER_CONFIG}"
    --sincro_config "${SINCRO_CONFIG}"
    --reviwo_config "${REVIWO_CONFIG}"
    --out "${out_path}"
  )

  if [[ -n "${SPLATTER_CKPT:-}" ]]; then
    args+=(--splatter_ckpt "${SPLATTER_CKPT}")
  fi
  if [[ -n "${SINCRO_CKPT:-}" ]]; then
    args+=(--sincro_ckpt "${SINCRO_CKPT}")
  fi
  if [[ -n "${REVIWO_CKPT:-}" ]]; then
    args+=(--reviwo_ckpt "${REVIWO_CKPT}")
  fi
  if [[ "${REQUIRE_LPIPS}" == "1" || "${REQUIRE_LPIPS}" == "true" || "${REQUIRE_LPIPS}" == "yes" ]]; then
    args+=(--require_lpips)
  fi

  uv run python visualize/visualize_render_quality_comparison.py "${args[@]}" "$@"
}

for target_cam in "${TARGET_CAM_ARRAY[@]}"; do
  target_cam="${target_cam//[[:space:]]/}"
  [[ -z "${target_cam}" ]] && continue
  run_one "${target_cam}" "$@"
done
