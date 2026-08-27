from __future__ import annotations

from types import SimpleNamespace

from dataset.droid.rlds import _canonical_read_config


def test_canonical_rlds_read_config_uses_sequential_deterministic_shards() -> None:
    class Options:
        deterministic = None

    class FakeTFDS:
        @staticmethod
        def ReadConfig(**kwargs):
            return SimpleNamespace(**kwargs)

    fake_tf = SimpleNamespace(data=SimpleNamespace(Options=Options))
    config = _canonical_read_config(fake_tf, FakeTFDS)

    assert config.options.deterministic is True
    assert config.interleave_cycle_length == 1
    assert config.interleave_block_length == 1
    assert config.num_parallel_calls_for_interleave_files == 1
    assert config.num_parallel_calls_for_decode == 1
    assert config.try_autocache is False
    assert config.skip_prefetch is True
