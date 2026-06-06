#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-button-press-wall}"
TRAJECTORY_CONFIG="${TRAJECTORY_CONFIG:-agents/drqv2/config/sincro/${ENV_NAME}.yaml}"
OUT_DIR="${OUT_DIR:-outputs/policy_camera_trajectory_eval/${ENV_NAME}}"
TRAJECTORY="${TRAJECTORY:-both}"
BASE_CAMERAS="${BASE_CAMERAS:-${BASE_CAMERA:-cam0,cam1,cam2,cam3,cam4,cam5}}"
NUM_EPISODES="${NUM_EPISODES:-20}"
VIDEO_EPISODES="${VIDEO_EPISODES:-${NUM_EPISODES}}"

yaml_value() {
  local file="$1"
  local key="$2"
  uv run python - "${file}" "${key}" <<"PY"
import sys, yaml
path, key = sys.argv[1:3]
with open(path, "r", encoding="utf-8") as f:
    value = yaml.safe_load(f)
for part in key.split("."):
    if not isinstance(value, dict) or part not in value:
        sys.exit(0)
    value = value[part]
if value is not None:
    print(value)
PY
}

latest_checkpoint() {
  local dir="$1"
  if [[ ! -d "${dir}" ]]; then
    return 0
  fi
  find "${dir}" -maxdepth 1 -type f \( -name "*.pt" -o -name "*.pth" -o -name "*.tar" \) | sort -V | tail -n 1
}

policy_ckpt() {
  local config="$1"
  local explicit="$2"
  if [[ -n "${explicit}" ]]; then
    echo "${explicit}"
    return 0
  fi
  local dir
  dir="$(yaml_value "${config}" "train.checkpoint_dir")"
  latest_checkpoint "${dir}"
}

POLICY_ARGS=()
add_policy() {
  local name="$1"
  local config="$2"
  local ckpt="$3"
  if [[ -n "${ckpt}" ]]; then
    echo "[policy] ${name}: ${ckpt}"
    POLICY_ARGS+=(--policy "${name}" "${config}" "${ckpt}")
  else
    echo "[policy] ${name}: no checkpoint found for ${config}" >&2
  fi
}

CNN_CONFIG="${CNN_CONFIG:-agents/drqv2/config/cnn/${ENV_NAME}.yaml}"
REVIWO_CONFIG="${REVIWO_CONFIG:-agents/drqv2/config/reviwo/${ENV_NAME}.yaml}"
SINCRO_CONFIG="${SINCRO_CONFIG:-agents/drqv2/config/sincro/${ENV_NAME}.yaml}"
SPLATTERVAE_CONFIG="${SPLATTERVAE_CONFIG:-agents/drqv2/config/splattervae/${ENV_NAME}.yaml}"

add_policy cnn "${CNN_CONFIG}" "$(policy_ckpt "${CNN_CONFIG}" "${CNN_CKPT:-}")"
add_policy reviwo "${REVIWO_CONFIG}" "$(policy_ckpt "${REVIWO_CONFIG}" "${REVIWO_POLICY_CKPT:-}")"
add_policy sincro "${SINCRO_CONFIG}" "$(policy_ckpt "${SINCRO_CONFIG}" "${SINCRO_POLICY_CKPT:-}")"
add_policy splattervae "${SPLATTERVAE_CONFIG}" "$(policy_ckpt "${SPLATTERVAE_CONFIG}" "${SPLATTERVAE_POLICY_CKPT:-}")"

if [[ "${#POLICY_ARGS[@]}" -eq 0 ]]; then
  echo "No policy checkpoints were found. Set CNN_CKPT, REVIWO_POLICY_CKPT, SINCRO_POLICY_CKPT, or SPLATTERVAE_POLICY_CKPT." >&2
  exit 1
fi

IFS="," read -r -a BASE_CAMERA_ARRAY <<< "${BASE_CAMERAS}"
for base_camera in "${BASE_CAMERA_ARRAY[@]}"; do
  base_camera="${base_camera//[[:space:]]/}"
  [[ -z "${base_camera}" ]] && continue
  uv run python agents/drqv2/evaluate_camera_trajectory.py \
    --trajectory_config "${TRAJECTORY_CONFIG}" \
    --trajectory "${TRAJECTORY}" \
    --base_camera "${base_camera}" \
    --num_episodes "${NUM_EPISODES}" \
    --video_episodes "${VIDEO_EPISODES}" \
    --out_dir "${OUT_DIR}/${base_camera}" \
    "${POLICY_ARGS[@]}" \
    "$@"
done
