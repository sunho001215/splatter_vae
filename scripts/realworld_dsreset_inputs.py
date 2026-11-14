"""Publish manifest-scoped learning inputs from established per-episode prep/QA records.

No preprocessing or training is launched. Run on the host with the compact stores.
The ordinary dispatcher must have recorded the raw validation, stage completions,
per-family QA and accepted calibration in the supplied ingest snapshot.
"""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if __name__ == '__main__':
    from s4d.rwlearn.dsreset_runtime import install
    install()

import yaml
from s4d.rwlearn.dsreset import learning_inputs
from s4d.rwlearn.store import open_store


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--ingest-manifest', type=Path, required=True)
    p.add_argument('--store', type=Path, help='optional descriptor; default selects exact raw hashes from ingest records')
    p.add_argument('--out-dir', type=Path, required=True)
    a = p.parse_args()
    m, ingest = json.loads(a.manifest.read_text()), json.loads(a.ingest_manifest.read_text())
    if a.store:
        descriptor = yaml.safe_load(a.store.read_text())
    else:
        paths = []
        for r in m['episodes']:
            matches = [e for e in ingest['episodes'].values()
                       if e.get('task') == m['task'] and e.get('sha256') == r['sha256']]
            if len(matches) != 1:
                raise ValueError(f"{r['id']}: no unique ingest record")
            paths.append(matches[0]['preprocessing']['stages']['train_data']['outputs'])
        descriptor = {'kind': 'compact', 'paths': paths}
    docs = learning_inputs(m, ingest, open_store(descriptor))
    a.out_dir.mkdir(parents=True, exist_ok=False)
    (a.out_dir / 'store.yaml').write_text(yaml.safe_dump(descriptor, sort_keys=False))
    for name, doc in docs.items():
        (a.out_dir / name).write_text(json.dumps(doc, indent=2) + '\n')
    print(json.dumps({'out': str(a.out_dir), 'task': m['task'], 'variant': m['variant'], 'launched': False}))


if __name__ == '__main__':
    main()
