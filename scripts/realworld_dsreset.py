"""Build immutable paired manifests and prep configs; never run jobs or modify a queue."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yaml
from s4d.rwlearn.dsreset import TASKS, available_preprocessing, paired_manifests, prep_bundle, raw_roots, reuse_from_ingest, verified_success_from_ingest, write_bundle


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', choices=TASKS, required=True)
    p.add_argument('--raw-root', type=Path, default=Path('/home/ws/data/real_world'))
    p.add_argument('--template', type=Path, required=True, help='accepted task prep template; supplies SAM prompts/workers')
    p.add_argument('--rig', required=True, help='per-task rig path, pending until calibration is accepted')
    p.add_argument('--rig-map', type=Path, help='JSON episode-id -> per-session rig path (PNP)')
    p.add_argument('--prep-only-available', action='store_true', help='provisional prep plan only; no training manifests or split; reports raw QA rejections')
    p.add_argument('--reuse-ingest', type=Path, help='discover success output reuse by exact hash from ingest snapshot')
    p.add_argument('--reuse', type=Path, help='JSON id -> {raw_sha256, train_data}; success outputs only')
    p.add_argument('--amendment', type=Path, help='existing documented hash-bound success gate amendment; never infers exclusions')
    p.add_argument('--success-split', type=Path, help='legacy split keyed by episode filenames, exact partition required')
    p.add_argument('--processed-root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--success-count', type=int, default=100)
    p.add_argument('--failure-count', type=int, default=50)
    a = p.parse_args()
    def read(path):
        return json.loads(path.read_text()) if path else None
    rejected = []
    if a.prep_only_available:
        manifests, rejected = available_preprocessing(a.task, raw_roots(a.raw_root,a.task))
    else:
        manifests = paired_manifests(a.task, raw_roots(a.raw_root, a.task), seed=a.seed,
                                expected={'success': a.success_count, 'failure': a.failure_count},
                                success_split=read(a.success_split), amendment=read(a.amendment),
                                verified_success=verified_success_from_ingest(read(a.reuse_ingest),a.task)
                                if a.reuse_ingest else None)
    template = yaml.safe_load(a.template.read_text())
    template['stages']['calibrate']['rig_calibration'] = a.rig
    reuse = reuse_from_ingest(manifests['SF'], read(a.reuse_ingest)) if a.reuse_ingest else {}
    reuse.update(read(a.reuse) or {})
    prep = prep_bundle(manifests, template, a.processed_root, rig_by_episode=read(a.rig_map), reuse=reuse)
    write_bundle(a.out, {} if a.prep_only_available else manifests, prep)
    if a.prep_only_available:
        (a.out/'provisional-inventory.json').write_text(json.dumps({'training_ready':False,
            'episodes':manifests['SF']['episodes'],'rejected':rejected},indent=2)+'\n')
    print(json.dumps({'task': a.task, 'episodes': len(prep), 'out': str(a.out), 'launched': False}))


if __name__ == '__main__':
    main()
