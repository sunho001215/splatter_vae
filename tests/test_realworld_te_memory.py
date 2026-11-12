"""TrackEverything worker CUDA memory policy (agent TEMEM, 2026-11-12): cached blocks are released after every request
(also a failed one, after its traceback was dropped), the allocator cap (required on MIG), and the policy/peaks recorded
in the response. CPU-only: a fake torch."""

from __future__ import annotations

import sys
import weakref
from types import SimpleNamespace

import numpy as np
import pytest

from s4d.realworld.workers import trackeverything_worker as te

GIB = 2**30


class FakeCuda:
    def __init__(self, total=24 * GIB, name="NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition"):
        self.total, self.name = total, name
        self.events = []
        self.allocated, self.reserved = 2 * GIB, 9 * GIB
        self.watch = None  # weakref of a request object that must be dead when the cache is released

    def get_device_properties(self, index):
        assert index == 0
        return SimpleNamespace(total_memory=self.total, name=self.name)

    def set_per_process_memory_fraction(self, fraction, device):
        self.events.append(("fraction", fraction, device))

    def empty_cache(self):
        self.events.append("empty_cache" if self.watch is None or self.watch() is None else "empty_cache_while_alive")
        self.reserved = self.allocated

    def reset_peak_memory_stats(self):
        self.events.append("reset_peak")

    def synchronize(self):
        pass

    def max_memory_allocated(self):
        return 4 * GIB

    def max_memory_reserved(self):
        return 5 * GIB

    def memory_allocated(self):
        return self.allocated

    def memory_reserved(self):
        return self.reserved


@pytest.fixture
def fake_torch(monkeypatch):
    cuda = FakeCuda()
    fake = SimpleNamespace(cuda=cuda, manual_seed=lambda seed: None, __version__="2.8.0")
    monkeypatch.setattr(te, "_MEMORY", {})
    monkeypatch.setitem(sys.modules, "torch", fake)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-00000000-0000-0000-0000-000000000000")  # a whole GPU
    return fake


def _request():
    t, h, w = 2, 16, 32
    inputs = {"video": np.zeros((t, h, w, 3), np.uint8), "depth": np.ones((t, h, w), np.float32),
              "K": np.broadcast_to(np.eye(3), (t, 3, 3)).copy(), "w2c": np.broadcast_to(np.eye(4), (t, 4, 4)).copy(),
              "queries": np.array([[0, 3.5, 4.5], [1, 8.5, 2.5]], np.float32)}
    return {"task": "track_queries", "params": {"mode": "sparse", "num_iters": 3}, "repo_dir": "r", "checkpoint": "c"}, \
        inputs


def test_policy_without_cap_only_releases(fake_torch, monkeypatch):
    monkeypatch.delenv(te.CUDA_MEMORY_CAP_ENV, raising=False)
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    policy = te.memory_policy(fake_torch)
    assert policy == {"release_cached_after_request": True, "cuda_memory_cap_gib": None, "memory_fraction": None,
                      "device_total_memory": 24 * GIB, "mig_instance": False,
                      "pytorch_cuda_alloc_conf": "expandable_segments:True"}
    assert fake_torch.cuda.events == []  # no allocator cap


def test_cap_is_applied_once_as_a_fraction_of_device_memory(fake_torch, monkeypatch):
    monkeypatch.setenv(te.CUDA_MEMORY_CAP_ENV, "6")
    first = te.memory_policy(fake_torch)
    assert first["cuda_memory_cap_gib"] == 6.0 and first["memory_fraction"] == pytest.approx(0.25)
    assert te.memory_policy(fake_torch) == first  # cached per process: applied exactly once
    assert fake_torch.cuda.events == [("fraction", pytest.approx(0.25), 0)]


def test_cap_above_device_memory_is_clamped_to_the_device(fake_torch, monkeypatch):
    monkeypatch.setenv(te.CUDA_MEMORY_CAP_ENV, "80")
    assert te.memory_policy(fake_torch)["memory_fraction"] == 1.0


@pytest.mark.parametrize("raw", ["0", "-2", "abc", "nan", "inf"])
def test_invalid_cap_fails_closed(fake_torch, monkeypatch, raw):
    monkeypatch.setenv(te.CUDA_MEMORY_CAP_ENV, raw)
    with pytest.raises(ValueError, match=te.CUDA_MEMORY_CAP_ENV):
        te.memory_policy(fake_torch)
    assert te._MEMORY == {} and fake_torch.cuda.events == []  # nothing applied; the next request fails again


def test_handle_releases_cache_and_records_policy_and_peaks(fake_torch, monkeypatch):
    monkeypatch.setenv(te.CUDA_MEMORY_CAP_ENV, "12")
    monkeypatch.setattr(te, "_model", lambda repo, ckpt, device: ("tracker", {"name": "TrackEverything"}))
    monkeypatch.setattr(te, "runtime_provenance", lambda torch: {"flags": {}})
    out = {"xyz_base": np.zeros((2, 2, 3), np.float32), "vis_logit": np.zeros((2, 2), np.float32),
           "birth_t": np.array([0, 1])}
    calls = []

    def infer(tracker, video, depth, K, w2c, queries, num_iters):
        calls.append(fake_torch.cuda.events[-1])  # inference starts after the peak reset
        return out, 1.5

    monkeypatch.setattr(te, "infer_queries", infer)
    identity, arrays, meta = te.handle(*_request())
    assert identity == {"name": "TrackEverything"} and arrays is out and calls == ["reset_peak"]
    assert fake_torch.cuda.events[-1] == "empty_cache"  # after inference, before the response
    assert meta["max_memory_allocated"] == 4 * GIB and meta["max_memory_reserved"] == 5 * GIB
    assert meta["memory"]["release_cached_after_request"] is True and meta["memory"]["cuda_memory_cap_gib"] == 12.0
    assert meta["memory"]["allocated_after_release"] == meta["memory"]["reserved_after_release"] == 2 * GIB
    assert "allocated_after_release" not in te._MEMORY  # per-request values never leak into the cached policy


class RequestTensor:  # stands in for a CUDA tensor that only the failing frame references
    pass


def test_failed_request_releases_cache_after_its_tensors_are_dead(fake_torch, monkeypatch):
    monkeypatch.delenv(te.CUDA_MEMORY_CAP_ENV, raising=False)
    monkeypatch.setattr(te, "_model", lambda repo, ckpt, device: ("tracker", {}))

    def oom(*args, **kwargs):
        activations = RequestTensor()
        fake_torch.cuda.watch = weakref.ref(activations)
        raise RuntimeError("CUDA out of memory (fake)")

    monkeypatch.setattr(te, "infer_queries", oom)
    with pytest.raises(te.RequestFailed, match=r"RuntimeError: CUDA out of memory \(fake\)") as err:
        te.handle(*_request())
    assert "in oom" in str(err.value)  # the original traceback travels as text
    assert fake_torch.cuda.events[-1] == "empty_cache"  # not "empty_cache_while_alive"
    assert err.value.__context__ is None and err.value.__cause__ is None  # no chained exception keeps frames alive


def test_model_load_failure_also_releases_cache(fake_torch, monkeypatch):
    monkeypatch.delenv(te.CUDA_MEMORY_CAP_ENV, raising=False)

    def load(repo, ckpt, device):
        partial = RequestTensor()
        fake_torch.cuda.watch = weakref.ref(partial)
        raise RuntimeError("CUDA out of memory while loading (fake)")

    monkeypatch.setattr(te, "_model", load)
    with pytest.raises(te.RequestFailed, match="while loading"):
        te.handle(*_request())
    assert fake_torch.cuda.events == ["empty_cache"]  # nothing else ran; nothing kept alive


@pytest.mark.parametrize("visible,name", [("MIG-0257ebe5-58b1-5a24-8564-bd96af094cbf", "NVIDIA RTX PRO 6000"),
                                          ("0", "NVIDIA RTX PRO 6000 Blackwell MIG 1g.24gb")])
def test_cap_is_required_on_a_mig_instance(fake_torch, monkeypatch, visible, name):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    fake_torch.cuda.name = name
    monkeypatch.delenv(te.CUDA_MEMORY_CAP_ENV, raising=False)
    with pytest.raises(ValueError, match="required on a MIG instance"):
        te.memory_policy(fake_torch)
    assert te._MEMORY == {}
    monkeypatch.setenv(te.CUDA_MEMORY_CAP_ENV, "8")
    policy = te.memory_policy(fake_torch)
    assert policy["mig_instance"] is True and policy["cuda_memory_cap_gib"] == 8.0
    assert fake_torch.cuda.events == [("fraction", pytest.approx(8 / 24), 0)]


def test_runtime_flags_record_allocator_configuration(monkeypatch, tmp_path):
    props = SimpleNamespace(name="GPU X", uuid="GPU-abc", major=12, minor=0, total_memory=24 * GIB)
    cudnn = SimpleNamespace(version=lambda: 91002, enabled=True, benchmark=False, deterministic=False, allow_tf32=True)
    backends = SimpleNamespace(cudnn=cudnn, cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False)))
    fake = SimpleNamespace(cuda=SimpleNamespace(get_device_properties=lambda i: props),
                           hub=SimpleNamespace(get_dir=lambda: str(tmp_path)), _C=SimpleNamespace(),
                           version=SimpleNamespace(cuda="12.9"), __version__="2.8.0", backends=backends,
                           get_float32_matmul_precision=lambda: "highest",
                           are_deterministic_algorithms_enabled=lambda: False)
    monkeypatch.setattr(te, "_RUNTIME", {})
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    monkeypatch.setenv(te.CUDA_MEMORY_CAP_ENV, "8")
    flags = te.runtime_provenance(fake)["flags"]
    assert flags["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True" and flags[te.CUDA_MEMORY_CAP_ENV] == "8"
