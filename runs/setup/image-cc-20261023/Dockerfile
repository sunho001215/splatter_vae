# Derived campaign image (2026-10-23): the accepted runtime image s4d-runtime-03cca5218096dc88 (ID
# sha256:c3ffea7f7c114dd12755f9d537f760a6d5b31c96c923233185b775e0324799b7, verified before the build) plus a C compiler,
# which Triton needs to build its CUDA helper at first launch (the Chamfer nearest-neighbour kernel). Nothing else changes:
# same Python environment, packages, user and entry; running containers are not touched.
ARG BASE=s4d-runtime-03cca5218096dc88:latest
FROM ${BASE}
USER root
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/* \
    && gcc --version | head -1 && dpkg-query -W gcc libc6-dev
USER compu
