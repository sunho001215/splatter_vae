"""Immutable drawer-v2 S RGB publication; no global gate or ingest mutations."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from ops.jobs.local_phase_a import identity, inventory, readonly, require, sha
from ops.jobs.local_phase_a_hybrid import verify_output
from s4d.rwlearn.dsreset import inspect_raw, validate_manifest

TASK = 'drawer-open-v2'
METHODS = ('reviwo', 'dynamo', 'e2e_cnn', 'e2e_cnn_noaug')
TOOL = 'runs/setup/tools/register_drawer_rgb_20261115.py'
PAUSE = 'drawer-rgb-await-placement-20261115'
FILES = ('store.yaml', 'split_seed0.json', 'outcomes.json', 'gate_report.json')


def canonical(manifest):
    from s4d.rwlearn import dsreset_gate as gate
    validate_manifest(manifest)
    approved = gate.load_amendments()[TASK]['record']
    require(manifest['task'] == TASK and manifest['variant'] == 'S', 'drawer SUCCESS-only')
    require(manifest.get('amendment') == approved and len(manifest['episodes']) == 98,
            'canonical 98-success amendment required')
    require({r['filename']: r['sha256'] for r in manifest['episodes']} ==
            {Path(k).name: v for k, v in approved['episodes'].items()}, 'canonical membership differs')
    require({r['filename']: r['sha256'] for r in manifest.get('excluded', [])} ==
            {Path(k).name: v['raw_sha256'] for k, v in approved['excluded'].items()}, 'exclusions differ')
    return manifest


def hybrid_ids(manifest):
    return ['rwv2-20261113-prep-ours-'+r['id']+'-hybridA' for r in canonical(manifest)['episodes']]


def prepare(manifest_path, ingest_path, processed, out, *, execute=False, queue=None):
    """Default reads/validates only. --execute creates a new local publication after hybrid closeout."""
    from s4d.realworld import io as rio
    from s4d.realworld.pipeline import camera_views, load_meta
    from s4d.realworld.train_store import CompactStore
    from s4d.realworld.rgb_train_store import write_episode
    from s4d.rwlearn import dsreset_gate as gate
    from s4d.rwlearn.dsreset_identity import recording_label, store_identities
    from s4d.rwlearn.store import sync_index
    from ops.inference_server.contract_checks import singleview_family

    manifest_path, ingest_path, processed, out = map(Path, (manifest_path, ingest_path, processed, out))
    require(out.is_absolute() and '..' not in out.parts and
            not any(p.is_symlink() for p in (out,*out.parents)), 'real absolute output required')
    require(not out.is_relative_to(processed) and not out.is_relative_to('/home/ws/data/real_world'),
            'RGB publication must stay outside immutable raw/hybrid roots')
    m = canonical(json.loads(manifest_path.read_text()))
    ingest = json.loads(ingest_path.read_text())
    require(not out.exists(), 'publication exists; use a new output directory')
    raw_root = Path(m['episodes'][0]['path']).parent
    require(Path(str(raw_root)+'.VERIFIED').is_file(), 'local raw transfer verification marker missing')
    require({p.name for p in raw_root.glob('episode_*.h5')} ==
            {r['filename'] for r in m['episodes']+m['excluded']}, 'raw directory membership changed')
    for r in m['excluded']:
        require(sha(r['path']) == r['sha256'], 'excluded raw SHA drift')
        matches = [e for k,e in ingest['episodes'].items() if e.get('task') == TASK
                   and Path(e.get('source',{}).get('path') or e.get('filename') or k).name == r['filename']
                   and recording_label(e.get('source',{}).get('path') or e.get('filename') or k) == 'success']
        require(not matches or (len(matches) == 1 and matches[0].get('validation',{}).get('status') == 'invalid'
                and matches[0].get('sha256') in (None,r['sha256'])), 'excluded raw must remain invalid')
    # Validate the released source/qualification and durable output of every registered hybrid.
    from ops.jobs import local_phase_a as a, local_phase_a_hybrid as h, realworld_local as local
    queue = queue or {}
    by_id = {r['id']: r for r in queue.get('jobs', [])}
    sources, entries, verified_bundles = {}, {}, set()
    for row, rid in zip(m['episodes'], hybrid_ids(m)):
        require(rid in by_id, 'hybrid row not registered: '+rid)
        registered = by_id[rid]
        args = registered['args']
        require(len(args) == 8 and args[::2] == ['--bundle','--job','--job-sha256','--gate-sha256'],
                'invalid registered hybrid arguments')
        bundle, jobfile = Path(args[1]), Path(args[3])
        require(jobfile.resolve().is_relative_to(bundle.resolve()/'jobs') and sha(jobfile) == args[5],
                'hybrid job SHA/path differs')
        job = json.loads(jobfile.read_text())
        require(registered.get('registration_tool') == 'runs/setup/tools/register_local_phase_a_20261115.py'
                and registered['id'] == rid and job['episode'] == row['id'] and h.is_hybrid(job),
                'only registered qualified three-stage hybrids may publish RGB')
        cache_key = (str(bundle), args[7], job['envelope_sha256'])
        if cache_key not in verified_bundles:
            a.scheduler_inputs(registered)  # expensive frozen weights/environment/qualification closure ONCE per bundle
            verified_bundles.add(cache_key)
        else:
            require(sha(bundle/'envelope.json') == job['envelope_sha256'] and sha(bundle/h.REPORT) == args[7],
                    'qualified bundle changed')
            require(job['pipeline']['adapter']['config']['episodes'] == {job['episode']:job['raw']}
                    and sha(job['raw']) == job['raw_sha256'], 'hybrid raw identity drift')
        root = processed.joinpath(*identity(row['id']))
        require(Path(job['pipeline']['output_root']) == root and job['raw_sha256'] == row['sha256'],
                'hybrid root/raw differs')
        proof = h.verify_output(root, row['id'])
        require(proof['raw_sha256'] == row['sha256'] and proof['envelope_sha256'] == job['envelope_sha256'],
                'hybrid output/raw/envelope differs')
        meta = load_meta(root,row['id'])
        require(meta.robot.asset_sha256.get('urdf') and
                sha(meta.robot.urdf_path) == meta.robot.asset_sha256['urdf'], 'normalized robot URDF SHA drift')
        close = local._closeout(rid, local.FINISHED_DIR)
        completion = a.MAIN/'runs'/rid/'completion.json'
        require(close['files'].get('completion.json') == sha(completion), 'hybrid closeout differs')
        matches = [e for e in ingest['episodes'].values() if e.get('task') == TASK and e.get('sha256') == row['sha256']]
        require(not matches or (len(matches) == 1 and not matches[0].get('flags')
                and not matches[0].get('override') and matches[0].get('validation',{}).get('status') == 'valid'),
                'existing ingest has blocked/invalid raw identity')
        entries[row['id']] = inspect_raw(row['path'])
        sources[rid] = dict(root=str(root), manifest_sha256=sha(root/'hybrid-stage-manifest.json'),
                            completion_sha256=sha(completion))
    preview = dict(schema='drawer-rgb-inputs/1', prepared=False, hybrid_sources=sources,
                   manifest_sha256=sha(manifest_path), ingest_sha256=sha(ingest_path), out=str(out))
    if not execute:
        return preview
    out.mkdir(parents=True, exist_ok=False)
    paths, evidence = [], {}
    for row in m['episodes']:
        ep = row['id']
        root = Path(sources['rwv2-20261113-prep-ours-'+ep+'-hybridA']['root'])
        # No writes under the hybrid root: its three-stage inventory remains immutable.
        dest = out/'episodes'/ep
        write_episode(root, ep, dest, task=TASK, semantic=False, rgb_only=True, max_sync_ns=20_000_000)
        store = CompactStore([dest])
        key = store.episodes()[0]
        meta = load_meta(root, ep)
        expected = sorted(v.rig_id for v in camera_views(meta))
        sv = singleview_family(store, key, expected_cameras=expected)
        sets = sync_index(store, key, geometry=False)
        require(sv['passed'] and len(sets.times) > 0, ep+': RGB/action/M5 contracts failed')
        store.close()
        require(h.verify_output(root,ep)['raw_sha256'] == row['sha256'], 'hybrid changed during export')
        paths.append(str(dest))
        evidence[ep] = dict(singleview=sv, m5=sets.counts)
        evidence[ep]['raw_validation'] = entries[ep]
    # A local gate report records local hash/contract verification, without inventing a remote transfer record.
    store = CompactStore(paths)
    report = local_report(m,store,evidence,preview)
    require(all(gate.task_passed(report,TASK,c) for c in ('reviwo','dynamo','e2e_cnn')), 'RGB gate failed')
    require(not any(gate.task_passed(report,TASK,c) for c in ('ours','sincro','mvp3d','snerl')), 'geometry escaped')
    report['dataset_manifest_sha256'] = hashlib.sha256(json.dumps(m,sort_keys=True).encode()).hexdigest()
    report['rgb_only_projection'] = preview
    keys = {store.raw_sha256(ep): ep for ep in store.episodes()}
    docs = {'manifest.json':m, 'gate_report.json':report, 'contracts.json':evidence,
            'split_seed0.json':{s:[keys[r['sha256']] for r in m['episodes'] if r['split']==s]
                                for s in ('train','validation')},
            'outcomes.json':{ep:True for ep in store.episodes()}}
    store.close()
    for name, doc in docs.items():
        rio.atomic_write_json(out/name, doc)
    (out/'store.yaml').write_text(yaml.safe_dump(dict(kind='compact',paths=paths)))
    require(sha(manifest_path) == preview['manifest_sha256'] and sha(ingest_path) == preview['ingest_sha256'],
            'membership/ingest changed during export; discard incomplete publication')
    payload = {**preview, 'prepared':True, 'files':inventory(out)}
    rio.atomic_write_json(out/'publication.json', payload)
    readonly(out)
    return payload


def local_report(manifest, store, evidence, provenance):
    """Gate only the exact canonical S set after local raw, RGB/action and M5 validation."""
    from s4d.rwlearn import dsreset_gate as gate
    from s4d.rwlearn.dsreset_identity import raw_identity, store_identities
    canonical(manifest)
    ids = [r['id'] for r in manifest['episodes']]
    require(set(evidence) == set(ids), 'missing per-episode RGB QA')
    require(all(evidence[e]['singleview']['passed'] is True and evidence[e]['m5']['accepted'] > 0
                and evidence[e].get('raw_validation') for e in ids), 'local raw/RGB/action/M5 gate failed')
    families = {f: dict(passed=f in ('singleview','multiview_sets'),
                       failed_count=0 if f in ('singleview','multiview_sets') else 98,
                       failed_episodes=[] if f in ('singleview','multiview_sets') else ids)
                for f in gate.REPORT_FAMILIES}
    identities = {r['id']: {**raw_identity(TASK,r['path'],r['sha256']), 'store_episode':None}
                  for r in manifest['episodes']}
    report = dict(schema='drawer-local-rgb-gate/1',passed=True,geometry_passed=False,
        verification_source='local VERIFIED raw directory + rehashed canonical raw + qualified hybrid + CPU contracts',
        required_stages=['normalized','rgb_train_data','singleview_contract','m5_sets'],
        tasks={TASK:dict(passed=True,target=98,valid_and_preprocessed=98,episodes=ids,store_episodes=[],
                         episode_identities=identities,excluded=manifest['excluded'],problems=[],problem_count=0)},
        families={TASK:families},calibration_hold={TASK:dict(passed=False,validated=0,episodes={},
            problems=['RGB publication carries no accepted camera geometry'],problem_count=1)},
        rgb_only_projection=provenance)
    return gate.bind_store_episodes(report,{TASK:store.episodes()},store_identities(store))


def verify(bundle, expected):
    bundle = Path(bundle)
    require(sha(bundle/'publication.json') == expected, 'RGB publication SHA drift')
    doc = json.loads((bundle/'publication.json').read_text())
    require(doc['schema'] == 'drawer-rgb-inputs/1' and doc['prepared'] is True
            and doc['out'] == str(bundle), 'RGB publication incomplete or relocated')
    require(not any(p.is_symlink() for p in (bundle,*bundle.parents)), 'symlink publication')
    actual = inventory(bundle)
    actual.pop('publication.json')
    require(actual == doc['files'], 'RGB publication inventory drift')
    canonical(json.loads((bundle/'manifest.json').read_text()))
    from s4d.rwlearn import dsreset_gate as gate
    report = json.loads((bundle/'gate_report.json').read_text())
    require(all(gate.task_passed(report,TASK,c) for c in ('reviwo','dynamo','e2e_cnn')), 'RGB gate failed')
    require(not any(gate.task_passed(report,TASK,c) for c in ('ours','sincro','snerl','mvp3d')), 'geometry escaped')
    return {str(bundle/p): h for p,h in doc['files'].items()} | {str(bundle/'publication.json'):expected}


def bundle_paths(job):
    """Narrow resolver opt-in; scientific configuration still goes through the ordinary resolver."""
    if 'rgb_bundle' not in job:
        return set()
    require(job.get('registration_tool') == TOOL and job.get('realworld_target') == TASK
            and job.get('realworld_variant') == 'S' and job.get('rgb_method') in METHODS,
            'invalid RGB-only row')
    require(job.get('script') in ('scripts/train_reviwo.py','scripts/train_dynamo.py',
            'scripts/train_realworld_fql.py','scripts/realworld_fql_dataset.py',
            'scripts/dsreset_train_reviwo.py','scripts/dsreset_train_dynamo.py',
            'scripts/dsreset_train_realworld_fql.py','scripts/dsreset_realworld_fql_dataset.py',
            'scripts/realworld_dsreset_artifact.py'), 'RGB consumer not allowed')
    bundle = Path(job['rgb_bundle'])
    require(bundle.is_absolute() and '..' not in bundle.parts, 'absolute RGB publication required')
    return {str(bundle/name) for name in FILES}
