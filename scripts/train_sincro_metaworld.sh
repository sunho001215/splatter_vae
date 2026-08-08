#!/usr/bin/env bash
set -euo pipefail

# Run this script from inside the multiview_policy container.
# Edit these lists to select the GPU instances and Meta-World environments.
CUDA_VISIBLE_DEVICES_LIST=(
  "MIG-c2db2b36-91aa-5230-9337-9911fc28e9a9"
  "MIG-a400a724-7df2-5cc4-8a05-7f5fb8c98f8c"
  "MIG-53163a37-eee6-51b3-9b99-b38fc416cc41"
  "MIG-a10bf323-eb1c-567a-a6d8-91d1dc7306d3"
  "MIG-c8c9f08a-c87b-5fc9-874f-2ec68a020e0e"
  "MIG-b25bc715-171e-5c7c-934a-ae6828b0df44"
  "MIG-6dc2ec27-ea8a-5fc6-a3b1-50dfc4d52550"
  "MIG-fabf06f6-097d-5592-8124-9f6c4e16b3c5"
)

ENVS=(
  "button-press-wall"
  "drawer-open"
  "door-open"
  "hammer"
  "peg-unplug-side"
  "handle-press"
  "plate-slide"
  "stick-push"
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_DIR="${REPO_ROOT}/baselines/SinCro/config/metaworld"
LOG_DIR="${REPO_ROOT}/logs/metaworld_env_trainings/sincro/$(date +%Y%m%d_%H%M%S)"
TRAIN_ENTRY="baselines/SinCro/train.py"

if [[ "${#CUDA_VISIBLE_DEVICES_LIST[@]}" -eq 0 ]]; then
  echo "CUDA_VISIBLE_DEVICES_LIST is empty. Add at least one GPU index or UUID." >&2
  exit 1
fi
if [[ "${#ENVS[@]}" -eq 0 ]]; then
  echo "ENVS is empty. Add at least one Meta-World environment." >&2
  exit 1
fi

for env_name in "${ENVS[@]}"; do
  config="${CONFIG_DIR}/${env_name}.yaml"
  if [[ ! -f "${config}" ]]; then
    echo "Missing config for ${env_name}: ${config}" >&2
    exit 1
  fi
done

mkdir -p "${LOG_DIR}"
cd "${REPO_ROOT}"

PIDS=()

cleanup() {
  if [[ "${#PIDS[@]}" -gt 0 ]]; then
    echo "Stopping ${#PIDS[@]} SinCro worker(s)..."
    kill "${PIDS[@]}" 2>/dev/null || true
  fi
}
trap cleanup INT TERM

run_worker() {
  local worker_idx="$1"
  local gpu="$2"
  local num_workers="${#CUDA_VISIBLE_DEVICES_LIST[@]}"
  local idx
  local env_name
  local config
  local log_file
  local status
  local worker_failed=0

  for ((idx = worker_idx; idx < ${#ENVS[@]}; idx += num_workers)); do
    env_name="${ENVS[$idx]}"
    config="${CONFIG_DIR}/${env_name}.yaml"
    log_file="${LOG_DIR}/${env_name}.log"

    echo "[launch] worker=${worker_idx} ${env_name} -> CUDA_VISIBLE_DEVICES=${gpu}"
    if (
      export CUDA_VISIBLE_DEVICES="${gpu}"
      uv run python "${TRAIN_ENTRY}" --config "${config}"
    ) >"${log_file}" 2>&1; then
      echo "[done] worker=${worker_idx} ${env_name}"
    else
      status="$?"
      echo "[failed] worker=${worker_idx} ${env_name} exited with status ${status}. See ${log_file}" >&2
      worker_failed=1
    fi
  done

  return "${worker_failed}"
}

echo "Repository: ${REPO_ROOT}"
echo "Logs: ${LOG_DIR}"
echo "Launching ${#ENVS[@]} SinCro training jobs over ${#CUDA_VISIBLE_DEVICES_LIST[@]} GPU worker(s)."

worker_count="${#CUDA_VISIBLE_DEVICES_LIST[@]}"
if [[ "${#ENVS[@]}" -lt "${worker_count}" ]]; then
  worker_count="${#ENVS[@]}"
fi

for ((worker_idx = 0; worker_idx < worker_count; worker_idx++)); do
  run_worker "${worker_idx}" "${CUDA_VISIBLE_DEVICES_LIST[$worker_idx]}" &
  PIDS+=("$!")
done

failed=0
for idx in "${!PIDS[@]}"; do
  pid="${PIDS[$idx]}"
  if wait "${pid}"; then
    echo "[worker done] ${idx}"
  else
    status="$?"
    echo "[worker failed] ${idx} exited with status ${status}. See ${LOG_DIR}" >&2
    failed=1
  fi
done

trap - INT TERM

echo "All SinCro training jobs finished. Logs are in ${LOG_DIR}"
exit "${failed}"
