"""A row with pin_host is never moved by the host-neutral pool (matched campaigns keep both arms of a task on one host)."""
from ops.jobs import pool_placement as pool


def _row(rid, **extra):
    return {'id': rid, 'host': 'local', 'script': 'scripts/train.py', 'args': ['--name', rid, '--output-root',
            'runs/pretrain-pin-test-does-not-exist'], **extra}


def test_pinned_rows_are_not_movable():
    free, pinned = _row('pin-test-free-200k'), _row('pin-test-pinned-200k', pin_host='local')
    by_id = {r['id']: r for r in (free, pinned)}
    assert pool.movable(free, {}, by_id)
    assert not pool.movable(pinned, {}, by_id)


def test_rows_pinned_remote_wait_for_their_move():
    pinned_remote = _row('pin-test-remote-200k', pin_host='remote')
    assert pool.movable(pinned_remote, {}, {pinned_remote['id']: pinned_remote})  # the pool may move it to the remote
