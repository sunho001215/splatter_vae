"""Publish an encoder copy or validate a completed offline policy. No robot I/O."""
from __future__ import annotations
import argparse
import json
import os
import shutil
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def publish(source, out, kind):
    import torch
    from s4d.rwlearn.dsreset import sha256
    source, out = Path(source), Path(out)
    payload = torch.load(source, map_location='cpu', weights_only=False)
    if kind == 'encoder':
        if (not isinstance(payload, dict) or payload.get('format') not in
                {f'splatter4d-baseline-{m}-v1' for m in ('dynamo','reviwo','sincro','snerl','mvp3d')}
                or not payload.get('state_dict')):
            raise ValueError('not a baseline encoder export')
        if int(payload.get('step', -1)) <= 0:
            raise ValueError('refusing an untrained encoder export')
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name('.' + out.name + '.partial')
        shutil.copyfile(source, tmp)
        os.replace(tmp, out)
        return {'sha256': sha256(out), 'source_sha256': sha256(source)}
    if not isinstance(payload,dict):
        raise ValueError('not an offline FQL policy')
    if not isinstance(payload.get('agent'), dict) or not isinstance(payload.get('meta'), dict):
        raise ValueError('not an offline FQL policy')
    # The policy contains the final step and its agent state; deployment/actuation
    # eligibility is deliberately outside this offline artifact check.
    if int(payload.get('step', -1)) < 1_000_000:
        raise ValueError('FQL budget incomplete')
    metrics = source.parent / 'metrics.jsonl'
    records = [json.loads(line) for line in metrics.read_text().splitlines() if line.strip()]
    validation = [r for r in records if any(k.startswith('validation/') for k in r)]
    validation = [r for r in validation if r.get('step') == payload['step']]
    if not validation:
        raise ValueError('missing held-out offline validation')
    report = {'schema': 'dsreset-offline-eval/v1', 'policy_sha256': sha256(source),
              'step': payload['step'], 'final_validation': validation[-1],
              'robot_io': False, 'actuation_eligible': False}
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name('.' + out.name + '.partial')
    tmp.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, out)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True)
    p.add_argument('--kind', required=True, choices=('encoder', 'policy'))
    p.add_argument('--out', required=True)
    a = p.parse_args()
    print(json.dumps(publish(a.source, a.out, a.kind)))


if __name__ == '__main__':
    main()
