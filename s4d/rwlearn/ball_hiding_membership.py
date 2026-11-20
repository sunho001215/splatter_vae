"""BH3's hash-bound membership decision. Pure selection; never mutates queues or recordings."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AMENDMENT = ROOT / 'configs/realworld/gate_amendments/ball-hiding-99-20261121.json'
SPLIT = ROOT / 'configs/realworld/tasks/ball-hiding-split99.json'


def amendment():
    return json.loads(AMENDMENT.read_text())


def excluded_ids():
    return {'ball-hiding-success-' + Path(n).stem for n in amendment()['excluded']}


def eligible_jobs(jobs):
    # Identity equality only; a suffix or substring must never hide another job's failure.
    return [j for j in jobs if j.get('episode') not in excluded_ids()]


def verify_rejection(path):
    """Prove the exact raw bytes and recorder flags, without invoking any repair path."""
    import h5py
    import numpy as np
    from s4d.rwlearn.pnpv2_arrival import sha256

    path = Path(path)
    record = amendment()['excluded']['ball-hiding/' + path.name]
    if sha256(path) != record['raw_sha256']:
        raise ValueError('BH exclusion raw sha256 differs')
    with h5py.File(path, 'r') as f:
        if record['camera'] not in f['cameras']:
            raise ValueError('BH exclusion camera evidence missing')
        for camera in f['cameras']:
            for flag in ('t_valid', 'stereo/t_valid', 'stereo/depth_ok'):
                a = f['cameras'][camera][flag][()]
                wanted = [record['frame_index']] if (camera, flag) == (record['camera'], record['flag']) else []
                if np.flatnonzero(a == 0).tolist() != wanted:
                    raise ValueError('BH exclusion recorder evidence differs')
                if camera == record['camera'] and len(a) != record['frames']:
                    raise ValueError('BH exclusion frame count differs')
    return {'status': 'invalid', 'reasons': [record['reason']], 'raw_sha256': record['raw_sha256']}


def validate_membership(m):
    """Require the committed 99 hashes, documented rejection and recomputed split for every consumer."""
    import numpy as np

    a = amendment()
    split = json.loads(SPLIT.read_text())
    ids = sorted('ball-hiding-success-' + Path(n).stem for n in a['episodes'])
    order = np.random.default_rng([0, int(hashlib.sha256(b'ball-hiding').hexdigest()[:8], 16)]).permutation(len(ids))
    val = {ids[int(i)] for i in order[:int(np.ceil(.1 * len(ids))) ]}
    expected = {s: [e for e in ids if (e in val) == (s == 'validation')] for s in ('train', 'validation')}
    if ({s: split[s] for s in expected} != expected or split['amendment_sha256'] != hashlib.sha256(AMENDMENT.read_bytes()).hexdigest()
            or m.get('amendment') != a
            or {r['filename']: r['sha256'] for r in m['episodes']} != {Path(n).name: h for n, h in a['episodes'].items()}
            or {r['filename']: r['sha256'] for r in m.get('excluded', [])} !=
               {Path(n).name: r['raw_sha256'] for n, r in a['excluded'].items()}
            or m['split'] != {s: split[s] for s in ('train', 'validation')}):
        raise ValueError('BH requires the evidence-bound 99-episode membership and canonical 89/10 split')
    return m

