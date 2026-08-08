#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ROOT}/.venv/bin/python"
CONFIG_ROOT="${ROOT}/dataset/metaworld/configs"
DATA_ROOT="/home/ws/data/metaworld/pre-training"
LOG_ROOT="${ROOT}/logs/dataset_generation"
PHASE="${1:-all}"
WAFT_BATCH_SIZE=160

ENVIRONMENTS=(
  button-press-wall
  drawer-open
  door-open
  hammer
  peg-unplug-side
  handle-press
  plate-slide
  stick-push
)

MIGS=(
  MIG-c2db2b36-91aa-5230-9337-9911fc28e9a9
  MIG-a400a724-7df2-5cc4-8a05-7f5fb8c98f8c
  MIG-53163a37-eee6-51b3-9b99-b38fc416cc41
  MIG-a10bf323-eb1c-567a-a6d8-91d1dc7306d3
  MIG-c8c9f08a-c87b-5fc9-874f-2ec68a020e0e
  MIG-b25bc715-171e-5c7c-934a-ae6828b0df44
  MIG-6dc2ec27-ea8a-5fc6-a3b1-50dfc4d52550
  MIG-fabf06f6-097d-5592-8124-9f6c4e16b3c5
)

if [[ "${PHASE}" != "all" && "${PHASE}" != "collect" && "${PHASE}" != "flow" ]]; then
  echo "Usage: $0 [all|collect|flow]" >&2
  exit 2
fi

mkdir -p "${DATA_ROOT}" "${LOG_ROOT}"

run_parallel_phase() {
  local phase="$1"
  local -a pids=()
  local -a names=()
  local index env mig log

  for index in "${!ENVIRONMENTS[@]}"; do
    env="${ENVIRONMENTS[$index]}"
    mig="${MIGS[$index]}"
    log="${LOG_ROOT}/${env}_${phase}.log"
    if [[ "${phase}" == "collect" ]]; then
      echo "[launch] collect ${env} on ${mig}"
      (
        cd "${ROOT}"
        exec env \
          PYTHONUNBUFFERED=1 \
          MUJOCO_GL=egl \
          CUDA_VISIBLE_DEVICES="${mig}" \
          "${PYTHON}" dataset/metaworld/collect.py \
          --config "${CONFIG_ROOT}/${env}.yaml"
      ) >"${log}" 2>&1 &
    else
      echo "[launch] WAFT ${env} on ${mig} (batch=${WAFT_BATCH_SIZE})"
      (
        cd "${ROOT}"
        exec env \
          PYTHONUNBUFFERED=1 \
          CUDA_VISIBLE_DEVICES="${mig}" \
          "${PYTHON}" dataset/flow/precompute.py \
          "${DATA_ROOT}/${env}.hdf5" \
          --batch-size "${WAFT_BATCH_SIZE}" \
          --compression lzf
      ) >"${log}" 2>&1 &
    fi
    pids+=("$!")
    names+=("${env}")
  done

  local failed=0
  for index in "${!pids[@]}"; do
    if wait "${pids[$index]}"; then
      echo "[done] ${phase} ${names[$index]}"
    else
      echo "[failed] ${phase} ${names[$index]} (see ${LOG_ROOT}/${names[$index]}_${phase}.log)" >&2
      failed=1
    fi
  done
  return "${failed}"
}

if [[ "${PHASE}" == "all" || "${PHASE}" == "collect" ]]; then
  run_parallel_phase collect || exit 1
fi

if [[ "${PHASE}" == "all" || "${PHASE}" == "flow" ]]; then
  run_parallel_phase flow || exit 1
fi
