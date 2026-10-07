# Experiment log (append-only)

Each entry: hypothesis, change, evidence runs, result, decision. Never edit past entries; add corrections as new entries.

## 2026-10-07 — E0: campaign setup and runtime diagnosis

- **Hypothesis.** The earlier native-renderer failure came from gsplat binaries compiled against a different torch
  than the one at runtime; compiling gsplat and fused-ssim from source against the locked torch 2.10.0+cu129 with
  `TORCH_CUDA_ARCH_LIST=12.0` and CUDA 12.9 (matching torch's CUDA) will import and render on sm_120.
- **Change.** `pyproject.toml` now pins the build-time torch to the runtime torch (`match-runtime = true`) and passes
  the CUDA arch list and toolkit path as build variables. `reference_runtime.pth` was removed from the venv.
- **Evidence.** None yet. The `uv sync` that performs the build was denied by the Claude Code auto-mode classifier.
- **Result.** Blocked pending user permission.
- **Decision.** Prepare the RL port, scheduler and protocol, which do not need the runtime, then stop and ask.
