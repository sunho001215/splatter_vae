#!/usr/bin/env bash
set -euo pipefail

# Run this script from inside the multiview_policy container.
# Edit this list to choose the GPUs used by the training jobs.
CUDA_VISIBLE_DEVICES_LIST=(
  "GPU-fe480a9b-c516-3522-72b2-b633fc42834e"
  "GPU-b8ec1539-9673-d4f5-626c-fd5aa106c2d1"
  "GPU-32e2e165-4a57-b98e-d0a3-d3844b155190"
  "GPU-fd4be9bb-3d79-dfce-aaa6-9d9a62ffaa46"
  "GPU-dca08ecc-2a77-cb21-36d9-16cffe61ebb2"
  "GPU-c96e4904-1711-600f-5a1e-063fc3a67b1a"
  "GPU-739a6276-3558-fd02-66f0-7acfad38098f"
  "GPU-c952875c-7f47-e1ce-4d66-a380d7c014f1"
)

CONFIGS=(
  "config/metaworld/component_ablation/hammer-novoxel.yaml"
  "config/metaworld/component_ablation/drawer-open-novoxel.yaml"
  "config/metaworld/component_ablation/hammer-nodepth.yaml"
  "config/metaworld/component_ablation/drawer-open-nodepth.yaml"
  "config/metaworld/component_ablation/hammer-nocon.yaml"
  "config/metaworld/component_ablation/drawer-open-nocon.yaml"
  "config/metaworld/component_ablation/hammer-nocons.yaml"
  "config/metaworld/component_ablation/drawer-open-nocons.yaml"
  "config/metaworld/component_ablation/hammer-noshuf.yaml"
  "config/metaworld/component_ablation/drawer-open-noshuf.yaml"
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${REPO_ROOT}/logs/metaworld_component_ablations/$(date +%Y%m%d_%H%M%S)"

if [[ "${#CUDA_VISIBLE_DEVICES_LIST[@]}" -eq 0 ]]; then
  echo "CUDA_VISIBLE_DEVICES_LIST is empty. Add at least one GPU UUID." >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"
cd "${REPO_ROOT}"

PIDS=()

cleanup() {
  if [[ "${#PIDS[@]}" -gt 0 ]]; then
    echo "Stopping ${#PIDS[@]} worker(s)..."
    kill "${PIDS[@]}" 2>/dev/null || true
  fi
}
trap cleanup INT TERM

run_worker() {
  local worker_idx="$1"
  local gpu="$2"
  local num_workers="${#CUDA_VISIBLE_DEVICES_LIST[@]}"
  local idx
  local config
  local name
  local log_file

  for ((idx = worker_idx; idx < ${#CONFIGS[@]}; idx += num_workers)); do
    config="${CONFIGS[$idx]}"
    name="$(basename "${config}" .yaml)"
    log_file="${LOG_DIR}/${name}.log"

    echo "[launch] worker=${worker_idx} ${name} -> CUDA_VISIBLE_DEVICES=${gpu}"
    (
      export CUDA_VISIBLE_DEVICES="${gpu}"
      uv run train_model.py --config "${config}"
    ) >"${log_file}" 2>&1
    echo "[done] worker=${worker_idx} ${name}"
  done
}

echo "Repository: ${REPO_ROOT}"
echo "Logs: ${LOG_DIR}"
echo "Launching ${#CONFIGS[@]} training runs over ${#CUDA_VISIBLE_DEVICES_LIST[@]} GPU worker(s)."

worker_count="${#CUDA_VISIBLE_DEVICES_LIST[@]}"
if [[ "${#CONFIGS[@]}" -lt "${worker_count}" ]]; then
  worker_count="${#CONFIGS[@]}"
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

echo "All component-ablation runs finished. Logs are in ${LOG_DIR}"
exit "${failed}"
