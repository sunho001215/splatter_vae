ARG BASE_IMAGE
FROM ${BASE_IMAGE}
ARG BASE_IMAGE
ARG USER_ID=1000
ARG GROUP_ID=1000

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN grep -Eq '^(docker\.io/library/)?ubuntu:22\.04@sha256:[a-f0-9]{64}$' <<< "$BASE_IMAGE" \
    && test "$(uname -m)" = x86_64 \
    && source /etc/os-release && test "$ID:$VERSION_ID" = ubuntu:22.04
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ca-certificates curl git libstdc++6 libgomp1 libegl1 libgl1 libglvnd0 libopengl0 libglib2.0-0 \
    libx11-6 libxrandr2 libxcursor1 libxi6 \
    && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /opt/s4d/downloads /opt/s4d/uv \
    && curl --fail --location --retry 3 \
        https://github.com/astral-sh/uv/releases/download/0.10.4/uv-x86_64-unknown-linux-gnu.tar.gz \
        --output /opt/s4d/downloads/uv.tar.gz \
    && echo '6b52a47358deea1c5e173278bf46b2b489747a59ae31f2a4362ed5c6c1c269f7  /opt/s4d/downloads/uv.tar.gz' \
        | sha256sum --check --strict \
    && tar -xzf /opt/s4d/downloads/uv.tar.gz --directory /opt/s4d/uv --strip-components=1 \
    && test "$(/opt/s4d/uv/uv --version)" = 'uv 0.10.4'

ENV UV_PYTHON_INSTALL_DIR=/opt/s4d/python \
    UV_PROJECT_ENVIRONMENT=/opt/s4d/.venv \
    UV_CACHE_DIR=/opt/s4d/build-cache/uv \
    UV_LINK_MODE=copy \
    UV_NO_BUILD=1 \
    UV_PYTHON_DOWNLOADS=manual
COPY pyproject.toml uv.lock runtime_versions.json requirements.lock.txt manifest.json \
    native_payload.tar package_runtime.py Dockerfile /opt/s4d/bundle/
RUN /opt/s4d/uv/uv python install 3.10.19 \
    && PYTHON="$(/opt/s4d/uv/uv python find --managed-python 3.10.19)" \
    && "$PYTHON" -I /opt/s4d/bundle/package_runtime.py check-base --base-image "$BASE_IMAGE" \
    && "$PYTHON" -I /opt/s4d/bundle/package_runtime.py verify --bundle /opt/s4d/bundle \
    && /opt/s4d/uv/uv sync --project /opt/s4d/bundle --python "$PYTHON" --locked --extra dev \
        --no-build --no-install-package gsplat --no-install-package fused-ssim \
    && /opt/s4d/.venv/bin/python -I /opt/s4d/bundle/package_runtime.py install --bundle /opt/s4d/bundle
RUN grep -Eq '^[0-9]+$' <<< "$USER_ID" && test "$USER_ID" -gt 0 \
    && grep -Eq '^[0-9]+$' <<< "$GROUP_ID" && test "$GROUP_ID" -gt 0 \
    && groupadd --gid "$GROUP_ID" compu \
    && useradd --uid "$USER_ID" --gid "$GROUP_ID" --no-log-init \
        --home-dir /home/compu/kaist/sunho/runtime --shell /bin/bash compu \
    && mkdir -p /workspace/splatter4d /home/compu/kaist/sunho/runtime/cache/{uv,torch,extensions,xdg,tmp,wandb-data,wandb-cache} \
    && chown -R "$USER_ID:$GROUP_ID" /home/compu/kaist/sunho/runtime

ENV PATH=/opt/s4d/.venv/bin:/opt/s4d/uv:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    HOME=/home/compu/kaist/sunho/runtime \
    TMPDIR=/home/compu/kaist/sunho/runtime/cache/tmp \
    XDG_CACHE_HOME=/home/compu/kaist/sunho/runtime/cache/xdg \
    UV_CACHE_DIR=/home/compu/kaist/sunho/runtime/cache/uv \
    TORCH_HOME=/home/compu/kaist/sunho/runtime/cache/torch \
    TORCH_EXTENSIONS_DIR=/home/compu/kaist/sunho/runtime/cache/extensions \
    WANDB_CACHE_DIR=/home/compu/kaist/sunho/runtime/cache/wandb-cache \
    WANDB_DATA_DIR=/home/compu/kaist/sunho/runtime/cache/wandb-data \
    UV_NO_SYNC=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl
USER compu
WORKDIR /workspace/splatter4d
CMD ["python", "--version"]
