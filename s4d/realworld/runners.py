"""How stage code reaches compute, independent of where it runs.

A runner offers two calls and an acknowledgement:

* ``kernel(task, params, arrays, *, chunk_id) -> (outputs, meta, handle)`` - this package's compute kernels
  (``s4d.realworld.kernels``);
* ``worker(spec, task, params, arrays, *, chunk_id, allow_mock) -> (response, outputs, handle)`` - an isolated
  upstream-model worker (``s4d.realworld.workers``), same response/identity checks as ``call_worker``;
* ``ack(handle)`` - called by the stage AFTER it has durably saved the outputs (``io.ChunkStore.put``), so a remote
  producer can release the chunk.

``LocalRunner`` runs kernels in-process and workers through the file protocol (one subprocess per call). The remote
runner (RAM-only streaming to the inference container) lives in ``ops/inference_server`` - transport and resource
management are not part of the scientific package.
"""

from __future__ import annotations

from pathlib import Path


class LocalRunner:
    kind = "local"

    def __init__(self, workdir: Path):
        self.workdir = Path(workdir)

    def kernel(self, task: str, params: dict, arrays: dict, *, chunk_id: str):
        from s4d.realworld.kernels import run_kernel

        outputs, meta = run_kernel(task, params, arrays)
        return outputs, meta, None

    def worker(self, spec, task: str, params: dict, arrays: dict, *, chunk_id: str, allow_mock: bool = False):
        from s4d.realworld.workers import call_worker

        response, outputs = call_worker(spec, task, params, arrays, self.workdir / spec.name, allow_mock=allow_mock)
        return response, outputs, None

    def ack(self, handle) -> None:
        return None

    def describe(self) -> dict:
        return {"kind": self.kind}
