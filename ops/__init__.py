"""Operational tooling of the splatter4d campaign, kept out of the research package ``s4d``.

- ``ops.remote``: SSH access, file transfer and the download budget; the read-only on-host artifact inspector.
- ``ops.jobs``: the file-based scheduler, its remote-host backend, run provenance and lifecycle (replay) cleanup.
- ``ops.resources``: GPU isolation/admission guard, bounded NVIDIA queries and cross-host equivalence checks.
- ``ops.containers``: the remote ``docker run`` launcher and typed release-verification jobs.

``ops`` may import ``s4d``; ``s4d`` imports ``ops`` only where docs/DEVELOPER.md lists it.
"""
