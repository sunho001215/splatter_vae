"""Thin PREP-only ball-hiding adapter for the unchanged pnp-v2 scientific pipeline.

Call bootstrap once, before importing pnp modules. No training/registration entry is exposed.
"""
from pathlib import Path


def bootstrap():
    from s4d.rwlearn.pnpv2_spec import select_task
    select_task('ball-hiding')
    from ops.jobs import pnpv2_plan
    pnpv2_plan.STAGES = {**pnpv2_plan.STAGES,
        'normalized': ('external', ('ingest',)),
        'rgb-publication': ('external', ('normalized',)),
        'wave:rgb': ('queue', ('check-data', 'ingest', 'normalized', 'rgb-publication'))}
    # All dynamic offline hooks still reject BH overrides: there is no committed BH override.
    from s4d.rwlearn.pnpv2_runtime import install
    install()


def split_jobs(jobs, remote_normalize):
    """Identical configs/settings; ingest once, then original Phase A reuses its validated ingest cache.

    Per-episode external dependencies permit geometry prep to stream as normalized episodes complete.
    RGB publication waits only for the complete 100-episode normalized collection.
    """
    import copy
    normalize, phase_a = copy.deepcopy(jobs), copy.deepcopy(jobs)
    for n, a in zip(normalize, phase_a):
        n['id'] = n['id'].replace('-prepA-', '-prepN-')
        n['phase'] = 'N'
        # Ingest performs no model inference; reserve only the launcher's minimum GPU charge.
        n['gpu_gib'], n['ram_gib'] = 1, 8
        argv = n['steps'][0]['argv']
        argv[argv.index('--stages') + 1] = 'ingest'
        a['external_states'] = {f'{remote_normalize}/state/state.json': [n['id']]}
    return normalize, phase_a


def phase_a_remote(spec, a, argv):
    """Stage two independent bundles; the operator starts both controllers. No jobs are launched here."""
    import json
    import shutil
    from scripts import pick_and_place_v2 as producer
    from scripts.realworld_prep_register import compile_plans
    from ops.jobs import rwprep_remote as remote
    from s4d.rwlearn.pnpv2_spec import STEM, PREP_DISK_GIB, values

    p = values(spec)
    producer._check_data(spec, a)
    producer._remote_raw_verified(spec)
    root = producer.ATTACH_ROOT / f'{STEM}-compile'
    if root.exists():
        raise ValueError(f'{root} exists; inspect the previous staging attempt')
    compiled = compile_plans([Path(p['prep_plan'])], root, p['remote_processed'], remote.RAW, stem=f'{STEM}-prepA')
    for job in compiled:
        job['disk_gib'] = PREP_DISK_GIB
    normal_name = f'{STEM}-normalize'
    remote_normalize = f'{remote.RW}/configs/{normal_name}'
    normalized, phase_a = split_jobs(compiled, remote_normalize)
    result = []
    # Copy the compiler artifacts so both queues are independently inspectable and hash-bound.
    for name, link, jobs in ((normal_name, p['rgb_source'], normalized),
                             (producer.PHASE_A_BUNDLE, p['phase_a'], phase_a)):
        out, alias = producer._attachment(link, name)
        out.mkdir(parents=True)
        shutil.copytree(root / 'configs', out / 'configs')
        plan = json.loads((root / 'prep-plan.json').read_text())
        plan = remote.rewrite(plan, root, out)
        remote.atomic(out / 'prep-plan.json', plan)
        jobs = remote.rewrite(jobs, root, out)
        staged = remote.stage(jobs, out, remote.RemoteClient(), release=STEM, name=name,
                              image=result[0]['image'] if result else None)
        producer._link(out, alias)
        staged['image'] = json.loads((out / 'dispatcher.json').read_text())['image']
        result.append(staged)
    print(json.dumps({'bundles': result, 'launched': False}))
