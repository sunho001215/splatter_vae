"""Prepare immutable RGB-only S inputs locally; dry-run by default, no network."""
import argparse
import json
import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from s4d.rwlearn.rgb_inputs import prepare
from ops.jobs.local_phase_a import MAIN, PROCESSED
import yaml

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--ingest', type=Path, default=Path('/home/ws/ws/realworld-ops/ingest/manifest.json'))
    p.add_argument('--processed', type=Path, default=PROCESSED)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    print(json.dumps(prepare(a.manifest,a.ingest,a.processed,a.out,execute=a.execute,
                     queue=yaml.safe_load((MAIN/'experiments/queue.yaml').read_text())),indent=2))
