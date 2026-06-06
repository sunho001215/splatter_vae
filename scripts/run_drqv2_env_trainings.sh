#!/usr/bin/env bash
set -euo pipefail

# Run this script from inside the multiview_policy container.
# Edit this list to choose the GPUs used by the training jobs.
CUDA_VISIBLE_DEVICES_LIST=(
  "GPU-fe480a9b-c516-3522-72b2-b633fc42834e"
  "GPU-b8ec1539-9673-d4f5-626c-fd5aa106c2d1"
  "GPU-32e2e165-4a57-b98e-d0a3-d3844b155190"
  "GPU-fd4be9bb-3d79-dfce-aaa6-9d9a62ffaa46"
  # "GPU-dca08ecc-2a77-cb21-36d9-16cffe61ebb2"
  # "GPU-c96e4904-1711-600f-5a1e-063fc3a67b1a"
  # "GPU-739a6276-3558-fd02-66f0-7acfad38098f"
  # "GPU-c952875c-7f47-e1ce-4d66-a380d7c014f1"
)

# Available config families: cnn, splattervae, reviwo, sincro.
CONFIG_FAMILY="splattervae"

ENVS=(
  "button-press-wall"
  "coffee-push"
  "door-open"
  "drawer-open"
  "faucet-close"
  "hammer"
  # "handle-pull"
  "lever-pull"
  # "peg-unplug-side"
  # "push-wall"
  # "sweep-into"
  "window-open"
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${REPO_ROOT}/logs/drqv2_env_trainings/${CONFIG_FAMILY}/$(date +%Y%m%d_%H%M%S)"

if [[ "${#CUDA_VISIBLE_DEVICES_LIST[@]}" -eq 0 ]]; then
  echo "CUDA_VISIBLE_DEVICES_LIST is empty. Add at least one GPU UUID." >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"
cd "${REPO_ROOT}"

PIDS=()

cleanup() {
  if [[ "${#PIDS[@]}" -gt 0 ]]; then
    echo "Stopping ${#PIDS[@]} DrQ-v2 training job(s)..."
    kill "${PIDS[@]}" 2>/dev/null || true
  fi
}
trap cleanup INT TERM

echo "Repository: ${REPO_ROOT}"
echo "Config family: ${CONFIG_FAMILY}"
echo "Logs: ${LOG_DIR}"
echo "Launching ${#ENVS[@]} DrQ-v2 training jobs over ${#CUDA_VISIBLE_DEVICES_LIST[@]} GPU assignment(s)."

for idx in "${!ENVS[@]}"; do
  env_name="${ENVS[$idx]}"
  config="agents/drqv2/config/${CONFIG_FAMILY}/${env_name}.yaml"
  gpu="${CUDA_VISIBLE_DEVICES_LIST[$((idx % ${#CUDA_VISIBLE_DEVICES_LIST[@]}))]}"
  log_file="${LOG_DIR}/${env_name}.log"

  if [[ ! -f "${config}" ]]; then
    echo "Missing config: ${config}" >&2
    exit 1
  fi

  echo "[launch] ${env_name} -> CUDA_VISIBLE_DEVICES=${gpu}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    uv run agents/drqv2/train_drqv2_metaworld.py --config "${config}"
  ) >"${log_file}" 2>&1 &

  PIDS+=("$!")
done

failed=0
for idx in "${!PIDS[@]}"; do
  pid="${PIDS[$idx]}"
  env_name="${ENVS[$idx]}"

  if wait "${pid}"; then
    echo "[done] ${env_name}"
  else
    status="$?"
    echo "[failed] ${env_name} exited with status ${status}. See ${LOG_DIR}/${env_name}.log" >&2
    failed=1
  fi
done

trap - INT TERM

echo "All DrQ-v2 training jobs finished. Logs are in ${LOG_DIR}"
exit "${failed}"
