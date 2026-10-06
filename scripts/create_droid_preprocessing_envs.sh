#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
env_root="${repo_root}/.preprocessing-envs"
uv_cache="${repo_root}/.uv-cache"

create_base() {
  local name="$1"
  local env_path="${env_root}/${name}"
  if [[ ! -x "${env_path}/bin/python" ]]; then
    uv venv --python 3.12 "${env_path}"
  fi
  UV_CACHE_DIR="${uv_cache}" uv pip install \
    --python "${env_path}/bin/python" \
    --index-strategy unsafe-best-match \
    -r "${repo_root}/preprocessing/environments/torch-cu128.txt" \
    -r "${repo_root}/preprocessing/environments/common-rlds.txt"
}

create_base da3
UV_CACHE_DIR="${uv_cache}" uv pip install \
  --python "${env_root}/da3/bin/python" \
  --index-strategy unsafe-best-match \
  -e "${repo_root}/third_party/Depth-Anything-3"

create_base megaflow
UV_CACHE_DIR="${uv_cache}" uv pip install \
  --python "${env_root}/megaflow/bin/python" \
  --index-strategy unsafe-best-match \
  -e "${repo_root}/third_party/MegaFlow"

create_base lagernvs
UV_CACHE_DIR="${uv_cache}" uv pip install \
  --python "${env_root}/lagernvs/bin/python" \
  --index-strategy unsafe-best-match \
  -r "${repo_root}/third_party/LagerNVS/requirements.txt"

for name in da3 megaflow lagernvs; do
  "${env_root}/${name}/bin/python" -c \
    'import sys, torch, tensorflow as tf; print(sys.version); print(torch.__version__, torch.version.cuda, tf.__version__)'
done
