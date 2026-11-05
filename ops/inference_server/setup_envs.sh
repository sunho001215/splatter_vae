#!/bin/sh
# Builds the isolated Python environments of the real-world inference container (run INSIDE the container).
#
# Everything lands under /opt/envs (= host ~/shkim/envs, bind mount). Package caches and temporary files go to the
# RAM tmpfs (/ram): nothing is written to the container's read-only root filesystem, and no model weights are
# cached on disk (weights are placed in /ram/weights per session; see ops/inference_server/README.md).
#
#   docker exec shkim_rw sh /opt/src/splatter4d/ops/inference_server/setup_envs.sh <env> [<env> ...]
#   envs: mapanything alltracker sam3 s4d tools foundation_stereo trackeverything
#
# Versions:
# - alltracker / sam3: the exact package sets of the environments verified on real DROID data
#   (/home/ws/ws/validation-20261031/envs/{alltracker,sam3}, frozen into ops/inference_server/locks/*.txt).
# - s4d: this repository's uv.lock (identical to the training .venv); gsplat and fused-ssim are copied from the
#   local .venv build (TORCH_CUDA_ARCH_LIST 12.0, same Ubuntu 22.04 / glibc 2.35) by the operator, not built here.
# - mapanything: torch 2.10.0+cu129 (as the other envs) + the pinned upstream checkout, non-editable.
# - foundation_stereo (optional depth backend, restored 2026-11-05 for the controlled depth comparison): the package
#   set of the environment verified locally (/home/ws/ws/validation-20261031/envs/foundation_stereo, frozen into
#   locks/foundation_stereo.txt); the worker imports the pinned upstream checkout from repo_dir (not installed).
set -eu
UV=/opt/bin/uv
SRC=/opt/src/splatter4d
LOCKS=$SRC/ops/inference_server/locks
TORCH_INDEX=https://download.pytorch.org/whl/cu129
mkdir -p /ram/uv-cache /ram/tmp
export UV_HTTP_TIMEOUT=900 UV_HTTP_RETRIES=5 UV_CACHE_DIR=/ram/uv-cache UV_PYTHON_INSTALL_DIR=/opt/envs/python UV_LINK_MODE=copy TMPDIR=/ram/tmp

venv() {  # name python-version (the managed CPython lives under /opt/envs/python)
    $UV python install -q $2
    [ -x /opt/envs/$1/bin/python ] || $UV venv -q --python $2 /opt/envs/$1
}

for env in "$@"; do
    echo "=== $env $(date -u +%FT%TZ)"
    case $env in
        mapanything)
            venv mapanything 3.12.12
            $UV pip install -q --python /opt/envs/mapanything/bin/python --index-url $TORCH_INDEX \
                torch==2.10.0+cu129 torchvision==0.25.0+cu129
            # editable: the imported code is the pinned checkout whose commit the worker reports (and verifies)
            $UV pip install -q --python /opt/envs/mapanything/bin/python -e /opt/upstream/map-anything
            ;;
        foundation_stereo)
            venv foundation_stereo 3.11.14  # newest 3.11 the container uv offers (local verified env: 3.11.17)
            $UV pip install -q --python /opt/envs/foundation_stereo/bin/python --index-strategy unsafe-best-match \
                --extra-index-url $TORCH_INDEX -r $LOCKS/foundation_stereo.txt
            ;;
        trackeverything)
            # Recipe stub: deliberately NOT executed/validated as part of the adapter implementation.
            # Upstream checkout must be /opt/upstream/trackeverything (the package directory itself).
            venv trackeverything 3.10.19
            $UV pip install -q --python /opt/envs/trackeverything/bin/python --index-strategy unsafe-best-match \
                --extra-index-url $TORCH_INDEX --find-links https://data.pyg.org/whl/torch-2.8.0+cu129.html \
                --only-binary torch-scatter -r $LOCKS/trackeverything.txt
            # sm_120 support of the pt28cu129 torch_scatter wheel is UNVERIFIED. Before inference, validate
            # scatter_mean on the allocated GPU. If it reports "no kernel image", replace ONLY torch_scatter
            # with a locally built torch_scatter 2.1.2 wheel against this exact torch/CUDA 12.9 ABI, using
            # TORCH_CUDA_ARCH_LIST=12.0 and --no-build-isolation on a host with the matching CUDA toolkit.
            # Record the replacement wheel sha256 and freeze the env; alternatively use a supported GPU.
            # Do not build libs/ops (training only); do not install either geometry estimator.
            ;;
        alltracker|sam3)
            venv $env 3.12.12
            $UV pip install -q --python /opt/envs/$env/bin/python --index-strategy unsafe-best-match \
                --extra-index-url $TORCH_INDEX -r $LOCKS/$env.txt
            if [ $env = sam3 ]; then
                $UV pip install -q --python /opt/envs/sam3/bin/python --no-deps -e /opt/upstream/sam3
                # Triton JIT-builds a small C helper for sam3's GPU NMS; the image has no compiler, so the env carries
                # zig cc (pinned wheel) behind a CC wrapper (the worker config sets CC).
                $UV pip install -q --python /opt/envs/sam3/bin/python ziglang==0.13.0.post1
                printf '#!/bin/sh\nexec /opt/envs/sam3/bin/python -I -B -m ziglang cc "$@"\n' > /opt/envs/sam3/bin/cc-zig
                chmod +x /opt/envs/sam3/bin/cc-zig
            fi
            ;;
        tools)  # git for s4d.third_party (pinned submodule identity); conda-forge build via the static micromamba
            MAMBA_ROOT_PREFIX=/ram/mamba /opt/bin/micromamba create -y -q -p /opt/envs/tools -c conda-forge git=2.47
            /opt/envs/tools/bin/git --version
            continue
            ;;
        s4d)
            cd $SRC
            UV_PROJECT_ENVIRONMENT=/opt/envs/s4d $UV sync -q --frozen --no-install-project \
                --no-install-package gsplat --no-install-package fused-ssim --python 3.10.19
            ;;
        *) echo "unknown env $env" >&2; exit 2 ;;
    esac
    /opt/envs/$env/bin/python -I -c "import sys, torch; print('$env', sys.version.split()[0], torch.__version__, \
torch.cuda.is_available() and torch.cuda.get_device_name(0), torch.cuda.device_count())"
done
du -sh /opt/envs/* 2>/dev/null
