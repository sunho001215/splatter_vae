"""Append only SAM3-independent drawer-v2 S consumers. Dry-run default; no network/launch."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True

SOURCE = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(SOURCE),str(Path(__file__).resolve().parent)]
import register_ours_ext_20261113 as R
from ops.jobs.dsreset_plan import build_rows as paired_rows
from ops.jobs.local_phase_a import sha, require
from s4d.rwlearn.rgb_inputs import METHODS, TASK, TOOL, PAUSE, canonical, hybrid_ids, verify


def build_rows(bundle, manifest, release):
    bundle = Path(bundle)
    require(bundle.is_absolute() and bundle.name == 'S', 'publication path must be absolute and end in /S')
    canonical(manifest)
    stages = {s:dict(script='scripts/realworld_dsreset_check.py',args=['unused'],required_results=['unused'])
              for s in ('prep','calib','qa')}
    rows = paired_rows(TASK,str(bundle.parent),release,stage_rows=stages,rig='',methods=METHODS,variants=('S',))
    barriers = {r['id'] for r in rows[:3]}
    rows = rows[3:]
    checksum = sha(bundle/'publication.json') if (bundle/'publication.json').is_file() else 'NOT-PREPARED'
    for row in rows:
        method = next(m for m in METHODS if '-'+m+'-'+TASK in row['id'])
        stage = row['id'].split('-')[2]
        row.update(registration_tool=TOOL,local_placement_pause=PAUSE,realworld_variant='S',rgb_method=method,
                   rgb_bundle=str(bundle),rgb_publication_sha256=checksum,priority=-3 if stage in ('pre','export') else -1)
        row['deps'] = [d for d in row['deps'] if d not in barriers]
        if stage in ('pre','fqldata'):
            row['deps'] += hybrid_ids(manifest)
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,default=R.MAIN)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--commit',required=True)
    p.add_argument('--image',required=True)
    p.add_argument('--execute',action='store_true')
    a = p.parse_args(argv)
    import re
    require(re.fullmatch('[0-9a-f]{40}',a.commit) and re.fullmatch('sha256:[0-9a-f]{64}',a.image), 'full release pin required')
    m = canonical(json.loads(a.manifest.read_text()))
    release = dict(commit=a.commit,image=a.image,source='local',evidence_manifest=R.LOCAL_EVIDENCE)
    rows = build_rows(a.bundle,m,release)
    ready = (a.bundle/'publication.json').is_file()
    manifest_sha = sha(a.manifest)
    if ready:
        verify(a.bundle,sha(a.bundle/'publication.json'))
        require(json.loads((a.bundle/'publication.json').read_text())['manifest_sha256'] == manifest_sha,
                'registration manifest differs from RGB publication')
    record = dict(schema='drawer-rgb-registration/1',ready=ready,registered=False,rows=rows,release=release)
    from ops.jobs import realworld_local as local
    for row in rows:
        local.needed_inputs(row)
    if a.execute:
        require(Path.cwd().resolve() == SOURCE == a.repo.resolve() == R.MAIN, 'only adopted main may register')
        require(ready, 'prepare all 98 qualified hybrids and RGB contracts first')
        with R.locks(a.repo):
            require(R.local_release_pin(a.repo,a.commit,a.image) == release, 'release changed')
            before = (a.repo/'experiments/queue.yaml').read_bytes()
            queue = R.queue_document(before)
            known = {r['id'] for r in queue['jobs']} | R.registry_ids((a.repo/'experiments/registry.jsonl').read_bytes())
            from ops.jobs import host_roles
            known |= set(host_roles.read_ownership())
            require(not known.intersection(r['id'] for r in rows), 'rows already registered/attempted/owned')
            require(set(hybrid_ids(m)) <= {r['id'] for r in queue['jobs']}, 'hybrid dependencies missing')
            require(all(not (a.repo/'runs'/r['id']).exists() for r in rows), 'outputs already exist')
            for row in rows:
                local.placement_scope(row,root=a.repo)
            pausepath = a.repo/'runs/setup/admission-pauses.json'
            pauses = json.loads(pausepath.read_text())
            require(any(h.get('active') and h['name']=='s-only-cancelled-20261115-a9594590'
                        for h in pauses['pauses']), 'SF cancellation hold missing')
            require(not any(h['name']==PAUSE for h in pauses['pauses']), 'placement hold exists; inspect transaction')
            proof = a.repo/'runs/setup/drawer-rgb-registration-20261115.json'
            require(not proof.exists(), 'registration proof exists')
            require(sha(a.manifest) == manifest_sha, 'registration manifest changed')
            verify(a.bundle,rows[0]['rgb_publication_sha256'])
            pauses['pauses'].append(R.local_hold(rows,PAUSE))
            R.atomic_write(pausepath,R.json_bytes(pauses))  # fail closed if interrupted before queue append
            R.atomic_write(a.repo/'experiments/queue.yaml',R.queue_candidate(before,rows))
            record.update(registered=True)
            R.atomic_write(proof,R.json_bytes(record))
    print(json.dumps(record,indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
