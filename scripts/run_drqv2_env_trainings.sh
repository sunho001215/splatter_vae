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
  "MIG-8da55219-bd6b-5260-8be1-3b2a76ac733c"
  "MIG-c4ec2a95-30cc-5398-82bf-d9b11787fc70"
  "MIG-423203d9-5801-5892-a489-30f16a5a2d4a"
  "MIG-02b55a66-ed41-53c2-9004-c3f5619d6369"
  # GPU 5
  # "MIG-05a151e7-c6bd-5344-91cb-8bfb3a342631"
  # "MIG-b50e85c3-8498-5ef0-906e-5eda9db34b90"
  # "MIG-2ebd9e11-f2e0-5e22-870a-7779d44ec10a"
  # "MIG-c3511b41-5f78-52f1-9b93-9c6794cdbbdd"
  # GPU 6
  # "MIG-f30be823-f40d-5292-9474-427746dd3703"
  # "MIG-908fda99-f524-5319-ae5a-9f534a881ee1"
  # "MIG-3b813dc6-21e9-5e2e-bc3c-4aa8659b51a8"
  # "MIG-42dd67ec-ecc1-5c8f-b9a2-22e9a48a8c29"
)

# Available config families: cnn, splattervae, reviwo, sincro.
CONFIG_FAMILY="cnn"

# Edit this list to choose one or more random seeds. Each environment is
# launched once per seed (for example: SEEDS=(42 98 123)).
SEEDS=(
  11
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
LOG_DIR="${REPO_ROOT}/logs/drqv2_env_trainings/${CONFIG_FAMILY}/$(date +%Y%m%d_%H%M%S)"

if [[ "${#CUDA_VISIBLE_DEVICES_LIST[@]}" -eq 0 ]]; then
  echo "CUDA_VISIBLE_DEVICES_LIST is empty. Add at least one GPU UUID." >&2
  exit 1
fi

if [[ "${#SEEDS[@]}" -eq 0 ]]; then
  echo "SEEDS is empty. Add at least one non-negative integer seed." >&2
  exit 1
fi

declare -A SEEN_SEEDS=()
for seed in "${SEEDS[@]}"; do
  if [[ ! "${seed}" =~ ^[0-9]+$ ]]; then
    echo "Invalid seed: ${seed}. Seeds must be non-negative integers." >&2
    exit 1
  fi
  if [[ -n "${SEEN_SEEDS[${seed}]:-}" ]]; then
    echo "Duplicate seed: ${seed}. Each seed must be listed only once." >&2
    exit 1
  fi
  SEEN_SEEDS["${seed}"]=1
done

mkdir -p "${LOG_DIR}"
cd "${REPO_ROOT}"

PIDS=()
JOB_NAMES=()
JOB_LOG_FILES=()

cleanup() {
  if [[ "${#PIDS[@]}" -gt 0 ]]; then
    echo "Stopping ${#PIDS[@]} DrQ-v2 training job(s)..."
    kill "${PIDS[@]}" 2>/dev/null || true
  fi
}
trap cleanup INT TERM

echo "Repository: ${REPO_ROOT}"
echo "Config family: ${CONFIG_FAMILY}"
echo "Seeds: ${SEEDS[*]}"
echo "Logs: ${LOG_DIR}"
total_jobs=$(( ${#ENVS[@]} * ${#SEEDS[@]} ))
echo "Launching ${total_jobs} DrQ-v2 training jobs over ${#CUDA_VISIBLE_DEVICES_LIST[@]} GPU assignment(s)."

job_idx=0
for seed in "${SEEDS[@]}"; do
  for env_name in "${ENVS[@]}"; do
    config="agents/drqv2/config/${CONFIG_FAMILY}/${env_name}.yaml"
    gpu="${CUDA_VISIBLE_DEVICES_LIST[$((job_idx % ${#CUDA_VISIBLE_DEVICES_LIST[@]}))]}"
    log_file="${LOG_DIR}/${env_name}_seed${seed}.log"

    if [[ ! -f "${config}" ]]; then
      echo "Missing config: ${config}" >&2
      exit 1
    fi

    echo "[launch] ${env_name} seed=${seed} -> CUDA_VISIBLE_DEVICES=${gpu}"
    (
      export CUDA_VISIBLE_DEVICES="${gpu}"
      uv run agents/drqv2/train_drqv2_metaworld.py --config "${config}" --seed "${seed}"
    ) >"${log_file}" 2>&1 &

    PIDS+=("$!")
    JOB_NAMES+=("${env_name} seed=${seed}")
    JOB_LOG_FILES+=("${log_file}")
    ((job_idx += 1))
  done
done

failed=0
for idx in "${!PIDS[@]}"; do
  pid="${PIDS[$idx]}"
  job_name="${JOB_NAMES[$idx]}"
  log_file="${JOB_LOG_FILES[$idx]}"

  if wait "${pid}"; then
    echo "[done] ${job_name}"
  else
    status="$?"
    echo "[failed] ${job_name} exited with status ${status}. See ${log_file}" >&2
    failed=1
  fi
done

trap - INT TERM

echo "All DrQ-v2 training jobs finished. Logs are in ${LOG_DIR}"
exit "${failed}"
