"""Two-success equivalence gate. Fixed thresholds; failures never authorize registration."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import yaml

from ops.jobs import local_phase_a as a

# Declared before inference; no fitting thresholds to observed differences.
TOLERANCES = {'depth_mean_abs_m': 0.002, 'depth_p99_abs_m': 0.01,
              'depth_p99_rel': 0.01, 'depth_valid_iou_min': 0.995,
              'mask_mean_iou_min': 0.99, 'mask_min_iou_min': 0.95,
              'mask_score_max_abs': 0.005, 'mask_box_max_abs_px': 1.0}
REFERENCES = (
    ('covering', 'covering-000000', 'episode_000000'),
    ('pick-and-place', 'pick-and-place-000002', 'episode_000002'),
)


def metrics_depth(reference, candidate):
    a.require(reference.shape == candidate.shape and reference.dtype == candidate.dtype, 'depth shape/dtype differs')
    a.require(np.isfinite(reference).all() and np.isfinite(candidate).all(), 'nonfinite depth')
    rv, cv = reference > 0, candidate > 0
    both = rv & cv
    union = np.count_nonzero(rv | cv)
    delta = np.abs(candidate[both].astype(np.float64) - reference[both])
    a.require(delta.size > 0, 'no common valid depth pixels')
    return {'valid_iou': float(np.count_nonzero(both) / max(1, union)),
            'common_valid_pixels': int(delta.size), 'mean_abs_m': float(delta.mean()),
            'max_abs_m': float(delta.max()), 'p99_abs_m': float(np.quantile(delta, .99)),
            'p99_rel': float(np.quantile(delta / reference[both], .99)),
            'bitwise': bool(np.array_equal(reference, candidate))}


def metrics_masks(reference, candidate):
    a.require(reference.shape == candidate.shape and reference.dtype == candidate.dtype, 'mask shape/dtype differs')
    a.require(np.isin(candidate, [0, 1, 2, 3]).all(), 'unknown semantic class')
    rows = []
    # Absent-in-both classes do not inflate IoU. Report their counts separately.
    absent = 0
    for frame in range(len(reference)):
        for cls in range(1, 4):
            r, c = reference[frame] == cls, candidate[frame] == cls
            union = np.count_nonzero(r | c)
            if not union:
                absent += 1
                continue
            rows.append(float(np.count_nonzero(r & c) / union))
    a.require(rows, 'no foreground classes to compare')
    return {'mean_iou': float(np.mean(rows)), 'min_iou': min(rows), 'comparisons': len(rows),
            'absent_both': absent, 'bitwise': bool(np.array_equal(reference, candidate))}


def compare_instances(reference, candidate):
    """Raw SAM outputs: identity/counts exact, instance IoU and score/box bounds."""
    a.require(set(reference['chunks']) == set(candidate['chunks']), 'SAM chunk inventory differs')
    ious, score_error, box_error = [], 0., 0.
    for name in reference['chunks']:
        r, c = reference['chunks'][name], candidate['chunks'][name]
        a.require(set(r) == set(c), 'SAM raw datasets differ')
        for key in ('frame', 'prompt_index', 'instance_id', 'count'):
            a.require(np.array_equal(r[key][:], c[key][:]), f'SAM {key} differs')
        a.require(r['masks'].shape == c['masks'].shape, 'SAM instance count/shape differs')
        for i in range(len(r['masks'])):
            x, y = r['masks'][i], c['masks'][i]
            union = np.count_nonzero(x | y)
            ious.append(float(np.count_nonzero(x & y) / union) if union else 1.)
        for key, limit in (('scores', 'mask_score_max_abs'), ('boxes_xyxy', 'mask_box_max_abs_px')):
            x, y = r[key][:], c[key][:]
            a.require(x.shape == y.shape and np.isfinite(x).all() and np.isfinite(y).all(), f'SAM {key} invalid')
            delta = float(np.max(np.abs(x-y), initial=0.))
            if key == 'scores': score_error = max(score_error, delta)
            else: box_error = max(box_error, delta)
            a.require(delta <= TOLERANCES[limit], f'SAM {key} tolerance exceeded: {delta}')
    a.require(ious, 'no SAM instances compared')
    result = {'mean_iou': float(np.mean(ious)), 'min_iou': min(ious), 'instances': len(ious),
              'max_score_abs': score_error, 'max_box_abs_px': box_error}
    a.require(result['mean_iou'] >= TOLERANCES['mask_mean_iou_min'] and
              result['min_iou'] >= TOLERANCES['mask_min_iou_min'], 'SAM instance-mask IoU exceeded')
    return result


def h5_arrays(path):
    result = {}
    with h5py.File(path, 'r') as f:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                result[name] = (obj.shape, str(obj.dtype))
        f.visititems(visit)
    return result


def exact_h5(reference, candidate):
    a.require(h5_arrays(reference) == h5_arrays(candidate), f'HDF5 dataset inventory differs: {candidate}')
    compared = 0
    with h5py.File(reference, 'r') as r, h5py.File(candidate, 'r') as c:
        for key in (None, *h5_arrays(reference)):
            x, y = (r, c) if key is None else (r[key], c[key])
            a.require(set(x.attrs) == set(y.attrs) and all(np.array_equal(x.attrs[k], y.attrs[k]) for k in x.attrs),
                      f'HDF5 attributes differ: {key}')
            if key is None:
                continue
            if not x.shape:
                a.require(np.array_equal(x[()], y[()]), f'{key} differs')
            else:
                for start in range(0, x.shape[0], 8):
                    a.require(np.array_equal(x[start:start+8], y[start:start+8]), f'{candidate}:{key} differs')
            compared += x.size
    return compared


def compare_episode(task, old_ep, number, root, bundle, stages=a.STAGES):
    reference = Path('/home/ws/ws/realworld-20261104/processed/tasks') / task / 'base/episodes' / old_ep
    ep = f'{task}-success-{number}'
    candidate = Path(root) / 'episodes' / ep
    proof = {'task': task, 'label': 'success', 'episode': ep, 'reference': str(reference),
             'candidate': str(candidate), 'stages': {}, 'passed': True, 'evidence_files': {}}
    for stage in stages:
        tail = Path('normalized') if stage == 'ingest' else Path('derived') / stage
        r, c = reference / tail, candidate / tail
        try:
            rd, cd = [json.loads((p / '_COMPLETE.json').read_text()) for p in (r, c)]
            a.require(rd['version'] == cd['version'], f'{stage}: computation version differs')
            files = sorted(r.glob('*.h5')) if stage != 'ingest' else sorted((r / 'frames').glob('*.h5'))
            a.require(files, f'{stage}: missing reference files')
            stage_results = []
            proof['stages'][stage] = {'passed': False, 'files': stage_results}
            for rf in files:
                cf = c / rf.relative_to(r)
                if stage in ('ingest', 'sensor_depth'):
                    elements = exact_h5(rf, cf)
                    stage_results.append({'file': rf.name, 'bitwise': True, 'elements': elements,
                                          **({'max_abs_m': 0.0, 'max_rel': 0.0, 'valid_iou': 1.0}
                                             if stage == 'sensor_depth' else {})})
                else:
                    with h5py.File(rf, 'r') as x, h5py.File(cf, 'r') as y:
                        if stage == 'sam3_masks':
                            raw_metrics = compare_instances(x, y)
                            stage_results.append({'file': rf.name, 'raw_instances': raw_metrics})
                        key = 'depth' if stage == 'foundation_stereo_depth' else 'semantic'
                        a.require(x[key].shape == y[key].shape, f'{stage}: full frame count differs')
                        # Frame by frame bounds memory and exposes worst-frame deviations.
                        vals = [metrics_depth(x[key][i:i+1], y[key][i:i+1]) if key == 'depth' else
                                metrics_masks(x[key][i:i+1], y[key][i:i+1]) for i in range(len(x[key]))]
                        stage_results.append({'file': rf.name, 'frames': len(vals), 'metrics': vals})
                        if key == 'depth':
                            a.require(all(v['mean_abs_m'] <= TOLERANCES['depth_mean_abs_m'] and
                                          v['p99_abs_m'] <= TOLERANCES['depth_p99_abs_m'] and
                                          v['p99_rel'] <= TOLERANCES['depth_p99_rel'] and
                                          v['valid_iou'] >= TOLERANCES['depth_valid_iou_min'] for v in vals), 'depth tolerance exceeded')
                            a.require(np.array_equal(y['valid'][:], y['depth'][:] > 0), 'depth validity inconsistent')
                        else:
                            a.require(all(v['mean_iou'] >= TOLERANCES['mask_mean_iou_min'] and
                                          v['min_iou'] >= TOLERANCES['mask_min_iou_min'] for v in vals), 'mask tolerance exceeded')
                            a.require(np.array_equal(x['class_present'][:], y['class_present'][:]), 'mask class presence differs')
                for p in (rf, cf):
                    proof['evidence_files'][str(p)] = a.sha(p)
            if stage == 'ingest':
                with np.load(r / 'arrays.npz') as x, np.load(c / 'arrays.npz') as y:
                    a.require(set(x.files) == set(y.files) and all(np.array_equal(x[k], y[k]) for k in x.files),
                              'normalized trajectory/timestamps differ')
                # JSON identity/path changes are explicitly mapped, all other contract fields exact.
                x, y = [json.loads((p / 'episode.json').read_text()) for p in (r, c)]
                a.require([(s['sha256'], s['bytes']) for s in x['provenance']['source_files']] ==
                          [(s['sha256'], s['bytes']) for s in y['provenance']['source_files']], 'ingest raw provenance differs')
                x['episode_id'] = y['episode_id']
                x['provenance']['source_files'] = y['provenance']['source_files']
                x['provenance']['created_utc'] = y['provenance']['created_utc']
                a.require(x == y, 'normalized contract differs beyond episode/path identity')
                for directory in (r, c):
                    for name in ('arrays.npz', 'episode.json'):
                        proof['evidence_files'][str(directory/name)] = a.sha(directory/name)
            proof['stages'][stage] = {'passed': True, 'files': stage_results}
            for p in (r / '_COMPLETE.json', c / '_COMPLETE.json'):
                proof['evidence_files'][str(p)] = a.sha(p)
        except (OSError, ValueError, KeyError) as exc:
            proof['passed'] = False
            proof['stages'].setdefault(stage, {})
            proof['stages'][stage].update(passed=False, error=str(exc))
    return proof


def gate_jobs(bundle):
    bundle = Path(bundle)
    jobs = []
    for task, old_ep, number in REFERENCES:
        ref = Path('/home/ws/ws/realworld-20261104/processed/tasks') / task / 'base/episodes' / old_ep
        rec = json.loads((ref / 'normalized/_COMPLETE.json').read_text())
        rawproof = json.loads((ref / 'normalized/episode.json').read_text())['provenance']['source_files'][0]
        ep = f'{task}-success-{number}'
        cfg = {'adapter': {'name': rec['config']['adapter'], 'config': rec['config']['adapter_config']},
               'baselines': ['snerl', 'mvp3d', 'sincro', 'reviwo', 'dynamo'], 'workers': {}, 'stages': {}}
        raw = Path('/home/ws/data/real_world') / task / f'{number}.h5'
        cfg['adapter']['config']['episodes'] = {ep: str(raw)}
        for st in a.STAGES[1:]:
            d = json.loads((ref / 'derived' / st / '_COMPLETE.json').read_text())
            cfg['stages'][st] = {k: v for k, v in d['config'].items() if k not in ('worker_identity', 'kernel', 'worker')}
        masks = ref/'derived/sam3_masks'
        with h5py.File(next(masks.glob('*.h5')), 'r') as f:
            order = json.loads(f.attrs['prompts'])
        prompts = cfg['stages']['sam3_masks']['prompts']
        cfg['stages']['sam3_masks']['prompts'] = {name: prompts[name] for name in order}
        cfg['stages']['calibrate'] = {'robot_mask_stage': 'sam3_masks'}
        cfg['output_root'] = str(bundle / 'gate' / task)
        job = {'episode': ep, 'raw': str(raw), 'raw_sha256': rawproof['sha256'], 'pipeline': cfg,
               'envelope_sha256': a.sha(bundle / 'envelope.json')}
        path = bundle / 'jobs' / f'gate-{task}.json'
        if path.exists():
            a.require(json.loads(path.read_text()) == job, 'gate input changed')
        else:
            a.atomic(path, job)
        jobs.append(path)
    return jobs


def write_report(bundle, rows, coverage):
    result = {'schema': 'local-phase-a-equivalence/1', 'coverage': coverage,
              'qualified': coverage == 'full-two-episodes' and all(r['passed'] for r in rows),
              'envelope_sha256': a.sha(Path(bundle) / 'envelope.json'), 'comparison_code_sha256': a.sha(__file__), 'tolerances': TOLERANCES, 'episodes': rows}
    resources = [json.loads((Path(bundle)/'gate'/r['task']/'local-phase-a.json').read_text()).get('footprints', {})
                 for r in rows if (Path(bundle)/'gate'/r['task']/'local-phase-a.json').exists()]
    result['footprints'] = {'measured': len(resources) == 2 and all(r.get('measured') for r in resources),
                            'gpu_peak_reserved_gib': max((r.get('gpu_peak_reserved_gib', 0) for r in resources), default=0),
                            'rss_peak_gib': max((r.get('rss_peak_gib', 0) for r in resources), default=0),
                            'disk_gib': max((r.get('disk_gib', 0) for r in resources), default=0),
                            'gpu_declaration_gib': 16, 'ram_declaration_gib': 48, 'cpu_threads': 4, 'disk_declaration_gib': 32}
    a.atomic(Path(bundle) / 'equivalence.json', result)
    lines = ['# Local phase-A equivalence', '', f"Qualified: **{result['qualified']}**. Coverage: {coverage}.", '',
             'Depth limits: each frame mean abs <= 0.002 m, p99 abs <= 0.01 m, p99 rel <= 0.01; valid-mask IoU >= 0.995.',
             'Raw SAM frame/prompt/instance IDs and counts must match. Raw instance-mask IoU uses the same limits; score abs <= 0.005, box abs <= 1 px.',
             'Semantic foreground class/frame IoU: mean >= 0.99 and minimum >= 0.95; class presence exact.',
             'Absent-in-both classes are counted separately. Ingest arrays and sensor depth must be bitwise equal.',
             'HDF5 container bytes and completion IDs/times are not compared. Ingest contract mapping changes only episode_id, raw path and provenance.created_utc; raw SHA/size and every other field remain exact.', '',
             '| Episode | Stage | Result | Numbers |', '|---|---|---|---|']
    for row in rows:
        for stage, st in row['stages'].items():
            numbers = st.get('error', 'bitwise payloads equal')
            if stage == 'sensor_depth' and st['passed']:
                numbers = 'bitwise equal; max abs=0 m; max rel=0; valid IoU=1'
            vals = [v for f in st.get('files', []) for v in f.get('metrics', [])]
            if vals and stage == 'foundation_stereo_depth':
                numbers = f"worst frame mean abs={max(v['mean_abs_m'] for v in vals):.8g} m; p99 abs={max(v['p99_abs_m'] for v in vals):.8g} m; p99 rel={max(v['p99_rel'] for v in vals):.8g}; valid IoU min={min(v['valid_iou'] for v in vals):.8g}"
            elif vals:
                numbers = f"mean IoU={np.mean([v['mean_iou'] for v in vals]):.8g}; minimum IoU={min(v['min_iou'] for v in vals):.8g}"
            lines.append(f"| {row['episode']} | {stage} | {'PASS' if st['passed'] else 'BLOCKED/FAIL'} | {numbers} |")
    (Path(bundle) / 'EQUIVALENCE.md').write_text('\n'.join(lines) + '\n')
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', required=True, type=Path)
    p.add_argument('--cpu-only', action='store_true', help='ingest/sensor only; never qualifies bulk')
    p.add_argument('--compare-only', action='store_true')
    args = p.parse_args(argv)
    doc = a.verify_envelope(args.bundle, a.sha(args.bundle / 'envelope.json'))
    jobs = gate_jobs(args.bundle)
    stages = a.STAGES[:2] if args.cpu_only else a.STAGES
    rows = []
    for job, (task, old_ep, number) in zip(jobs, REFERENCES):
        if not args.compare_only:
            frozen = Path(doc['frozen']) / 'source'
            code = ('import sys;sys.path.insert(0,sys.argv.pop(1));from ops.jobs.local_phase_a import execute;'
                    'execute(sys.argv[1],sys.argv[2],sys.argv[3],stages=sys.argv[4].split(","))')
            subprocess.run([sys.executable, '-I', '-B', '-c', code, str(frozen), str(args.bundle), str(job), a.sha(job),
                            ','.join(stages)], cwd=frozen, check=False)
        row = compare_episode(task, old_ep, number, args.bundle / 'gate' / task, args.bundle, a.STAGES)
        rows.append(row)
        write_report(args.bundle, rows, 'cpu-only' if args.cpu_only else 'full-two-episodes')
    result = write_report(args.bundle, rows, 'cpu-only' if args.cpu_only else 'full-two-episodes')
    print(json.dumps({'qualified': result['qualified'], 'report': str(args.bundle / 'EQUIVALENCE.md')}))
    return 0 if result['qualified'] else 3


if __name__ == '__main__':
    raise SystemExit(main())
