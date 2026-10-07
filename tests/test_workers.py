"""Fork-safe DataLoader iteration (mitigation for the evaluation-loader crash of screen S1).

The crash (a CUDA tensor destructor running in a forked worker, "CUDA error: initialization error") could not be
reproduced deterministically; these tests check the mechanics of the mitigation.
"""

from __future__ import annotations

import gc

import torch
from torch.utils.data import DataLoader, Dataset

from s4d.train.workers import fork_safe_iter


class Indexed(Dataset):
    def __len__(self):
        return 10

    def __getitem__(self, index):
        return torch.tensor([index])


def test_fork_safe_iteration_collects_freezes_only_while_forking_and_yields_the_same_batches(monkeypatch):
    calls = []
    real_iter = DataLoader.__iter__

    def spying_iter(self):
        calls.append(gc.get_freeze_count() > 0)
        return real_iter(self)

    monkeypatch.setattr(DataLoader, "__iter__", spying_iter)
    garbage = {"tensor": torch.ones(8, device="cuda")}
    garbage["self"] = garbage
    del garbage
    loader = DataLoader(Indexed(), batch_size=3, num_workers=2)
    batches = [b.tolist() for b in fork_safe_iter(loader)]
    assert calls == [True], "workers are forked while the parent's collector is frozen"
    assert gc.get_freeze_count() == 0, "the parent's collector is re-enabled afterwards"
    assert batches == [b.tolist() for b in iter(loader)]
    assert not any(isinstance(o, dict) and "tensor" in o and o.get("self") is o for o in gc.get_objects())
