"""Read-only dependency barriers for already preprocessed S/SF artifacts. No robot I/O.

Bulk preparation uses the established dispatcher and generated prep-plan.json.
These cheap queue rows audit its outputs; they never duplicate completed prep.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if __name__ == '__main__':
    from s4d.rwlearn.dsreset_runtime import install
    install()

from s4d.rwlearn.baseline_data import check_gate
from s4d.rwlearn.dsreset import validate_manifest
from s4d.rwlearn.identity import store_identities
from s4d.rwlearn.store import open_store


def check(args):
    m = json.loads(Path(args.manifest).read_text())
    validate_manifest(m)
    split = json.loads(Path(args.split).read_text())
    outcomes = json.loads(Path(args.outcomes).read_text())
    report = json.loads(Path(args.gate_report).read_text())
    store = open_store(args.store)
    ids = store_identities(store)
    by_sha = {r['sha256']: r for r in m['episodes']}
    if len(ids) != len(by_sha) or {r['raw_sha256'] for r in ids.values()} != set(by_sha):
        raise ValueError('store raw identities do not equal the manifest')
    for ep, r in ids.items():
        row = by_sha[r['raw_sha256']]
        if r['task'] != m['task'] or type(outcomes.get(ep)) is not bool or outcomes[ep] != (row['label'] == 'success'):
            raise ValueError('store task/outcome differs from directory manifest')
        if (ep in split[row['split']]) is not True:
            raise ValueError('split differs from paired manifest')
    consumer = 'ours' if args.stage == 'calib' else 'dynamo'
    result = check_gate(store, report, steps=None, split=split, target=m['task'], consumer=consumer)
    if args.stage == 'qa':
        for c in ('ours', 'sincro', 'snerl', 'mvp3d', 'reviwo', 'dynamo', 'e2e_cnn'):
            check_gate(store, report, steps=None, split=split, target=m['task'], consumer=c)
    return {'schema':'dsreset-readiness/v1', 'stage':args.stage, 'task':m['task'], 'variant':m['variant'],
            'episodes':len(ids), 'passed':True, 'gate':result}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=('prep','calib','qa'), required=True)
    for flag in ('manifest','split','outcomes','gate-report','store','out'):
        p.add_argument('--'+flag, required=True)
    a = p.parse_args()
    result = check(a)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name('.'+out.name+'.partial')
    tmp.write_text(json.dumps(result,indent=2)+'\n')
    os.replace(tmp,out)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
