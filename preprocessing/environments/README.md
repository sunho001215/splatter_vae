# Offline preprocessing environments

The normal training environment never imports DA3, MegaFlow, or LagerNVS. The
three teachers run from separate Python 3.12 virtual environments created by
`scripts/create_droid_preprocessing_envs.sh`. All use the CUDA 12.8 PyTorch
2.8.0 stack (Blackwell-compatible), while repository and checkpoint revisions
are pinned in the adapters and dataset provenance.

`common-rlds.txt` intentionally includes the CPU-only RLDS decoder stack. Each
stage reads original 320x180 DROID pixels directly and TensorFlow is prevented
from seeing CUDA devices by `dataset.droid.rlds`.

At every non-dry-run top-level launch, each environment's full resolved package
inventory is recorded in `metadata/environments/*.json`; newly executed workers
also include it in `metadata/stages/*.json`. This preserves exact provenance
when a resumed run reuses stage metadata created by an older implementation.
The checked-in input constraints, pinned submodule SHAs, pinned Hugging Face
revisions, and those generated inventories together define the reproducible
environment contract.
