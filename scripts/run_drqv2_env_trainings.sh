#!/usr/bin/env bash
set -euo pipefail

# Run this script from inside the multiview_policy container.
# Edit this list to choose the GPUs used by the training jobs.
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

# Available config families: cnn, splattervae, reviwo, sincro.
CONFIG_FAMILY="splattervae_single_timestep_ablation"

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
