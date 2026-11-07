"""Four-arm campaign (arm4): condenses the local smoke (outputs/arm4_smoke/<prefix>/smoke.json) into
runs/setup/arm4-smoke-20261106.json, adding the step-0 comparison restricted to METRIC keys (val/<metric>@s<k> that are
not loss terms; loss terms differ by recipe by construction) between each arm's smoke run and its matched-initialisation
reference run (B: i2-m3d-seed0-hammer; A, C, D: cst50-seed0-hammer-200k), and the render-wrapper check if present.

    python -I arm4_smoke_summary_20261106.py [--smoke-root outputs/arm4_smoke]
"""
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arm4_common as C  # noqa: E402

root = C.ROOT / (sys.argv[sys.argv.index('--smoke-root') + 1] if '--smoke-root' in sys.argv else 'outputs/arm4_smoke')
LOSS = ('total', 'render', 'motion', 'infonce', 'consistency', 'rgb_', 'coverage_t', 'depth_l1', 'chamfer', 'depth_grad',
        'depth_hard', 'visibility', 'synth_render', 'occlusion', 'ramp')


def step0(path):
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if r.get('step') == 0 and 'val/psnr@s2' in r:
            return r
    return {}


out = {'recorded_at': datetime.now().astimezone().isoformat(), 'host': 'local', 'smoke_root': str(root),
       'not_campaign_runs': True, 'arms': {}}
for arm, spec in C.ARMS.items():
    p = root / spec['prefix'] / 'smoke.json'
    if not p.is_file():
        continue
    rec = json.loads(p.read_text())
    run = root / spec['prefix'] / f'arm4smoke-{spec["prefix"]}-hammer'
    a, r = step0(run / 'metrics.jsonl'), step0(C.matched_init_reference(arm, 'hammer') / 'metrics.jsonl')
    keys = sorted(k for k in a if k.startswith('val/') and k in r and not k[4:].startswith(LOSS))
    diffs = {k: abs(a[k] - r[k]) for k in keys}
    # sampled diagnostics: the M3D point sample (val/m3d_*) follows the random stream, which the Chamfer target sampling
    # (present in only one of the two runs) advances first; every deterministic metric must be bitwise equal
    sampled = {k for k in keys if k[4:].startswith('m3d_')}
    pre = rec['pretrain']
    out['arms'][arm] = {
        'prefix': spec['prefix'], 'gpu': rec['gpu'],
        'pretrain': {k: pre.get(k) for k in ('exit_code', 'wall_seconds', 'median_step_time_s', 'median_gpu_time_s',
                                             'peak_gpu_mem_gb', 'peak_session_pss_bytes', 'finite', 'loss_terms',
                                             'config_sha256', 'last_logged')},
        'step0_metrics_vs_matched_init_reference': {
            'reference_run': str(C.matched_init_reference(arm, 'hammer').relative_to(C.ROOT)), 'metric_keys': len(keys),
            'bitwise_equal': sum(v == 0 for v in diffs.values()), 'max_abs_diff': max(diffs.values(), default=None),
            'deterministic_keys': len(keys) - len(sampled),
            'deterministic_bitwise_equal': sum(diffs[k] == 0 for k in keys if k not in sampled),
            'differing_non_sampled': sorted(k for k in keys if k not in sampled and diffs[k] != 0),
            'largest': sorted(diffs.items(), key=lambda kv: -kv[1])[:5],
            'psnr_s2': [a.get('val/psnr@s2'), r.get('val/psnr@s2')],
            'heldout_psnr_s2': [a.get('val/heldout_psnr@s2'), r.get('val/heldout_psnr@s2')]},
        'export': {k: rec.get('export', {}).get(k) for k in ('exit_code', 'check')},
        'drm': {k: rec.get('rl', {}).get(k) for k in ('exit_code', 'wall_seconds', 'peak_session_pss_bytes', 'final_json',
                                                     'train_records', 'snapshots')},
        'evaluator': {k: rec.get('eval', {}).get(k) for k in ('exit_code', 'wall_seconds', 'peak_session_pss_bytes',
                                                              'records', 'steps')}}
out['render_wrapper'] = {}
for model in ('orig', 'cst50'):
    done = root / 'render' / model / 'done.json'
    if done.is_file():
        d = json.loads(done.read_text())
        out['render_wrapper'][model] = {k: d[k] for k in ('model', 'checkpoint', 'checkpoint_sha256', 'seconds', 'outputs',
                                                         'reproduction_of_chl_ref200')}
evalcheck = C.ROOT / 'runs/setup/arm4-evalcheck-20261106.json'
if evalcheck.is_file():
    out['evaluator_layout_check'] = json.loads(evalcheck.read_text())
dst = C.ROOT / 'runs/setup/arm4-smoke-20261106.json'
dst.write_text(json.dumps(out, indent=1, default=str) + '\n')
print(json.dumps({a: {'pre_exit': v['pretrain']['exit_code'], 's_per_it': v['pretrain']['median_step_time_s'],
                      'pss_GB': round((v['pretrain']['peak_session_pss_bytes'] or 0) / 1e9, 2),
                      'cfg_hash_equal': (v['pretrain']['config_sha256'] or {}).get('equal'),
                      'step0': {k: v['step0_metrics_vs_matched_init_reference'][k] for k in ('metric_keys', 'bitwise_equal', 'deterministic_keys', 'deterministic_bitwise_equal', 'differing_non_sampled', 'psnr_s2')},
                      'export': v['export']['exit_code'], 'drm': v['drm']['exit_code'], 'eval': v['evaluator']['exit_code'],
                      'eval_records': v['evaluator']['records']} for a, v in out['arms'].items()}, indent=1, default=str))
