#!/usr/bin/env bash
set -euo pipefail

# Run this script from inside the multiview_policy container.
# Edit this list to choose the GPUs used by the training jobs.
CUDA_VISIBLE_DEVICES_LIST=(
  # GPU 2
  # "MIG-c8c9f08a-c87b-5fc9-874f-2ec68a020e0e"
  # "MIG-b25bc715-171e-5c7c-934a-ae6828b0df44"
  # "MIG-6dc2ec27-ea8a-5fc6-a3b1-50dfc4d52550"
  # "MIG-fabf06f6-097d-5592-8124-9f6c4e16b3c5"
  # GPU 4
  # "MIG-8da55219-bd6b-5260-8be1-3b2a76ac733c"
  # "MIG-c4ec2a95-30cc-5398-82bf-d9b11787fc70"
  # "MIG-423203d9-5801-5892-a489-30f16a5a2d4a"
  # "MIG-02b55a66-ed41-53c2-9004-c3f5619d6369"
  # GPU 5
  "MIG-05a151e7-c6bd-5344-91cb-8bfb3a342631"
  "MIG-b50e85c3-8498-5ef0-906e-5eda9db34b90"
  "MIG-2ebd9e11-f2e0-5e22-870a-7779d44ec10a"
  "MIG-c3511b41-5f78-52f1-9b93-9c6794cdbbdd"
  # GPU 6
  "MIG-f30be823-f40d-5292-9474-427746dd3703"
  "MIG-908fda99-f524-5319-ae5a-9f534a881ee1"
  "MIG-3b813dc6-21e9-5e2e-bc3c-4aa8659b51a8"
  "MIG-42dd67ec-ecc1-5c8f-b9a2-22e9a48a8c29"
)

# Select either "temporal" or "ablations/single_timestep".
CONFIG_SET="${CONFIG_SET:-temporal}"
CONFIG_DIR="config/splattervae/metaworld/${CONFIG_SET}"

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
LOG_DIR="${REPO_ROOT}/logs/metaworld_env_trainings/${CONFIG_SET}/$(date +%Y%m%d_%H%M%S)"

if [[ "${#CUDA_VISIBLE_DEVICES_LIST[@]}" -eq 0 ]]; then
  echo "CUDA_VISIBLE_DEVICES_LIST is empty. Add at least one GPU UUID." >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"
cd "${REPO_ROOT}"

PIDS=()

cleanup() {
  if [[ "${#PIDS[@]}" -gt 0 ]]; then
    echo "Stopping ${#PIDS[@]} training job(s)..."
    kill "${PIDS[@]}" 2>/dev/null || true
  fi
}
trap cleanup INT TERM

echo "Repository: ${REPO_ROOT}"
echo "Config set: ${CONFIG_SET}"
echo "Logs: ${LOG_DIR}"
echo "Launching ${#ENVS[@]} training jobs over ${#CUDA_VISIBLE_DEVICES_LIST[@]} GPU assignment(s)."

for idx in "${!ENVS[@]}"; do
  env_name="${ENVS[$idx]}"
  config="${CONFIG_DIR}/${env_name}.yaml"
  gpu="${CUDA_VISIBLE_DEVICES_LIST[$((idx % ${#CUDA_VISIBLE_DEVICES_LIST[@]}))]}"
  log_file="${LOG_DIR}/${env_name}.log"

  if [[ ! -f "${config}" ]]; then
    echo "Missing config: ${config}" >&2
    exit 1
  fi

  echo "[launch] ${env_name} -> CUDA_VISIBLE_DEVICES=${gpu}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    uv run train_model.py --config "${config}"
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

echo "All training jobs finished. Logs are in ${LOG_DIR}"
exit "${failed}"
