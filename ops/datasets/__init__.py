"""Resumable internet downloads of the large-scale DROID training data onto the remote data disk (Directive I, 2026-11-10).

Standard library only: the owner process runs on the remote host's own ``python3`` outside the containers, from a copy
deployed to ``/data_8T/kaist/sunho/tools/<sha>/``.

- ``catalog``: pinned official sources (DROID RLDS 1.0.1, PointWorld-DROID, DROID-3D, hand-eye assets), the files the
  pipeline needs from each, their official checksums, and the plan builder.
- ``transfer``: HTTP range downloads with streaming verification (md5 / sha256 / git blob ids) and a shared rate limiter.
- ``state``: atomic JSON state, the single-owner lock and owner liveness.
- ``storage``: free space, the margin, final/peak projections and per-download admission.
- ``health``: host and network signals and the pool/rate controller.
- ``archives``: ``tar.zst`` extraction into staging, verification and atomic publication.
- ``runner``: the owner process (workers, extraction lanes, status).
- ``cli``: the remote command line (``scripts/datasets/droid_datasets.py``).
- ``control``: the local side (deploy, latency probe, remote commands through ``ops.remote.access``); not used remotely.
"""
