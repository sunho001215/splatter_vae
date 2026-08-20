#!/usr/bin/env bash
set -euo pipefail

# Run this script from inside the multiview_policy container.
# Edit these lists to select the GPU instances and Meta-World environments.
CUDA_VISIBLE_DEVICES_LIST=(
  "MIG-f30be823-f40d-5292-9474-427746dd3703"
  "MIG-908fda99-f524-5319-ae5a-9f534a881ee1"
  "MIG-3b813dc6-21e9-5e2e-bc3c-4aa8659b51a8"
  "MIG-42dd67ec-ecc1-5c8f-b9a2-22e9a48a8c29"
)

ENVS=(
  "peg-unplug-side"
  "handle-press"
  "plate-slide"
  "stick-push"
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_DIR="${REPO_ROOT}/baselines/ReViWo/config/metaworld"
LOG_DIR="${REPO_ROOT}/logs/metaworld_env_trainings/reviwo/$(date +%Y%m%d_%H%M%S)"
TRAIN_ENTRY="baselines/ReViWo/train.py"

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
    echo "Stopping ${#PIDS[@]} ReViWo worker(s)..."
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
echo "Launching ${#ENVS[@]} ReViWo training jobs over ${#CUDA_VISIBLE_DEVICES_LIST[@]} GPU worker(s)."

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

echo "All ReViWo training jobs finished. Logs are in ${LOG_DIR}"
exit "${failed}"
