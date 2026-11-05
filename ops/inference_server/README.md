# Real-world inference server (RAM-only streaming)

Preprocessing and model inference for real-world episodes run on a separate GPU host; the local host only reads the raw
episode, streams bounded chunks, saves results and assembles the training cache. Configuration:
`configs/realworld/inference_server.yaml` (host) and `configs/realworld/<episode>.yaml` (pipeline).

## Host layout (everything under `~/shkim` on the inference host)

| Path | Content |
|---|---|
| `~/shkim/src/splatter4d.git`, `~/shkim/src/splatter4d` | bare repository receiving `git push`, and the checkout the container runs (pinned to the local commit at every run) |
| `~/shkim/upstream/{map-anything,alltracker,sam3,dinov2}` | upstream checkouts at the commits pinned in `configs/realworld/upstream_pins.yaml` |
| `~/shkim/envs/{mapanything,alltracker,sam3,s4d}`, `~/shkim/envs/python` | isolated environments (`setup_envs.sh`), uv-managed CPython |
| `~/shkim/assets/validation-20261031` | robot URDF + meshes, mounted read-only at the same absolute path the local configs use |
| `~/shkim/bin/{uv,micromamba}` | uv; static micromamba (only for the `tools` env: conda-forge git, which `s4d.third_party` calls) |

## Container

One persistent container `shkim_rw` (`nvidia/cuda:12.8.0-base-ubuntu22.04`), created once:

```
docker run -d --name shkim_rw --restart unless-stopped \
  --device nvidia.com/gpu=GPU-9bcf38b6-8240-e910-6548-ca715643dc11 \
  --read-only --tmpfs /ram:rw,exec,size=26g,mode=1777 --tmpfs /tmp:rw,exec,size=2g,mode=1777 --shm-size=8g \
  --memory=56g --memory-swap=56g --cpus=16 --pids-limit=4096 --security-opt no-new-privileges --user $(id -u):$(id -g) \
  -v ~/shkim/envs:/opt/envs -v ~/shkim/upstream:/opt/upstream -v ~/shkim/src:/opt/src -v ~/shkim/bin:/opt/bin:ro \
  -v ~/shkim/assets/validation-20261031:/home/ws/ws/validation-20261031:ro \
  -e HOME=/ram/home -e HF_HOME=/ram/hf -e TORCH_HOME=/ram/torch -e XDG_CACHE_HOME=/ram/cache -e TMPDIR=/ram/tmp \
  -e UV_CACHE_DIR=/ram/uv-cache -e UV_PYTHON_INSTALL_DIR=/opt/envs/python -e UV_LINK_MODE=copy -e PIP_NO_CACHE_DIR=1 \
  -e PYTHONDONTWRITEBYTECODE=1 -e LC_ALL=C.UTF-8 -e HF_HUB_OFFLINE=1 --label owner=shkim \
  nvidia/cuda:12.8.0-base-ubuntu22.04 sleep infinity
docker exec shkim_rw sh /opt/src/splatter4d/ops/inference_server/setup_envs.sh mapanything s4d alltracker sam3 tools
```

- GPU isolation: the CDI device request names the one GPU UUID; `nvidia-smi -L` inside shows only it (checked by
  `session.Server.verify_container` before every run, together with the read-only root and the bind mounts).
- The root filesystem is read-only, so nothing can be written to the container's writable layer; `/ram` and `/tmp` are
  RAM tmpfs. Python runs with `-I -B` (no bytecode written below `~/shkim`).
- `s4d` env: `uv sync --frozen` of this repository's `uv.lock` (same versions as the training `.venv`); `gsplat` and
  `fused-ssim` are copied from the local `.venv` build (TORCH_CUDA_ARCH_LIST 12.0, Ubuntu 22.04 / glibc 2.35 on both).
- `sam3` env: Triton JIT-builds a C helper for sam3's GPU NMS; the env carries `zig cc` (pinned `ziglang` wheel) behind
  `bin/cc-zig`, selected by the worker env `CC`.

## Authentication and transport

The operator opens an SSH ControlMaster once (`ssh -fN irsl`, password typed interactively; nothing is stored).
Every later call uses `ssh -o BatchMode=yes irsl ...`, so a missing master fails immediately instead of prompting.
Workers run as `docker exec -i ... <env python> -I -B s4d/realworld/workers/<name>_worker.py --stream`: frames
(`s4d/realworld/workers/_framing.py`) carry a sha256 per payload; the server handles one request, returns the result,
and waits for the client's ACK, which the client sends only after `io.ChunkStore.put` has fsync'd the chunk and its
manifest line (backpressure: at most one chunk in flight per worker). The worker refuses to start unless its cwd and
`TMPDIR` are RAM-backed; the client refuses any session that is not, or whose repository commit differs from the
local one. Interrupted connections are retried (same chunk, pure function of its inputs); a restarted command resumes
every stage at its first unsaved chunk.

## Weights (RAM only)

`session.Server.stage_weights` places each configured file in `/ram/weights` (or the DINO cache in `/ram/s4d-cache`)
and checks its pinned sha256: local files are streamed through ssh into the tmpfs; the public MapAnything checkpoint
is downloaded straight into RAM at its pinned Hugging Face revision. Nothing is cached on disk; a container restart
clears them and the next run re-stages them.

## Running an episode (and resuming)

```
cd <local checkout at the commit the server runs>
.venv/bin/python -I ops/inference_server/run_episode.py --pipeline configs/realworld/mug_hanging_rgbd.yaml \
    --server configs/realworld/inference_server.yaml --episode mug-hanging-test-000000
.venv/bin/python -I ops/inference_server/depth_report.py --pipeline configs/realworld/mug_hanging_rgbd.yaml \
    --server configs/realworld/inference_server.yaml --episode mug-hanging-test-000000
```

Re-running the same command is the resume procedure (complete stages are skipped; an interrupted stage continues at
its first unsaved chunk). `--stages` selects stages, `--force-stage NAME` rebuilds one. Each run writes
`<output_root>/runs/<utc>/run.json`: container and source checks, weight staging, per-call timings and memory, and the
persistence audit (`audit_end.ok`: no file below `~/shkim` changed during the run and no new `docker diff` entry).
Do not push to the server checkout while a run is active: the source pin then refuses new sessions.

Without `--stages` a run executes the config's default stages (`s4d.realworld.cli.default_stages`): every pipeline
stage (optional ones such as `track_motion` only when configured) plus exactly the baseline stages its declared
`baselines:` need. Only our method is pretrained on DROID, so `droid_raw_validation.yaml` declares `baselines: []` and
never runs `sam3_masks`/`dino_snerl` (requesting them is refused). The real-world config declares the baselines trained
on it (SNeRL -> SAM 3 + DINO teacher; 3D-MVP, SinCro, ReViWo, DynaMo -> export only); its RGB-D calibration also uses
the SAM 3 robot masks as evidence.

Both depth sources are kept as separate versioned outputs: `mug_hanging_rgbd.yaml` (MapAnything depth for lifting and
export) and `mug_hanging_rgbd_rawdepth.yaml` (raw RealSense depth; its upstream stages are symlinked from the main
root). `depth_compare.py` (same arguments as `depth_report.py`) writes the common-support comparison with plain-language
metric definitions to `<output_root>/reports/<episode>/depth_v2/`.

## MuJoCo in the container

MuJoCo is not used by the real-world preprocessing (robot geometry is URDF forward kinematics in numpy) or by the
episode viewer (`scripts/realworld_viewer.py`: URDF FK + browser mesh rendering). Only simulator tests and Meta-World
rendering need it. The `nvidia/cuda:12.8.0-base-ubuntu22.04` image ships no GLVND loader (`libEGL.so.1`,
`libOpenGL.so.0`); CDI injects only NVIDIA's vendor library, so `MUJOCO_GL=egl` fails at import
(`mujoco_mig_setup`: `Failed to load libEGL.so.1`; plain `import mujoco`: PyOpenGL
`'NoneType' object has no attribute 'eglQueryString'`). The `tools` env carries conda-forge's GLVND libraries; with
`LD_LIBRARY_PATH=/opt/envs/tools/lib __EGL_VENDOR_LIBRARY_DIRS=/usr/share/glvnd/egl_vendor.d` (plus
`S4D_HOST_CONFIG=configs/hosts/inference.yaml`, `CUDA_VISIBLE_DEVICES=<the UUID>`) the guarded EGL path renders
(`mujoco_mig_setup` + `gpu_guard.enforce_mujoco_egl_device` + a Meta-World camera render). An `EGLError` printed at
interpreter exit comes from MuJoCo's context destructor during teardown and does not affect results.
