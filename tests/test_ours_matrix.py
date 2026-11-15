"""The new scratch matrix is distinct from the live continuation campaign."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

from s4d.config import load_config
from s4d.train.schedule import lr_function

pytestmark = pytest.mark.local_host_data

ROOT=Path(__file__).resolve().parents[1]
TOOLS=ROOT/'runs/setup/tools'
sys.path.insert(0,str(TOOLS))
import register_ours_matrix_20261115 as reg
import place_ours_matrix_loop_20261115 as loop
from ops.jobs.realworld_local import needed_inputs
from ops.jobs import scheduler


@pytest.fixture
def rows():
    current=dict(jobs=json.loads((ROOT/'tests/fixtures/ours400k_templates.json').read_text()))
    before=copy.deepcopy(current)
    release=dict(commit='a'*40,image='sha256:'+'b'*64,source='local',evidence_manifest=reg.R.LOCAL_EVIDENCE)
    result=reg.build_rows(current,release)
    assert current==before
    return result


@pytest.mark.parametrize('task',reg.TASKS)
@pytest.mark.parametrize('variant',['S','SF'])
def test_only_approved_science_changes_and_single_schedule(task,variant):
    reg.validate_recipes(ROOT)
    orig=load_config([ROOT/f'{reg.CONFIG}/{task}-{variant}-orig-scratch400k.yaml'])
    cham=load_config([ROOT/f'{reg.CONFIG}/{task}-{variant}-ch01-scratch400k.yaml'])
    expected=copy.deepcopy(orig);expected['loss']['chamfer']=.1
    assert cham==expected
    assert all(cham['loss'][k]>0 for k in ('depth_l1','depth_grad','depth_hard','visibility'))
    assert orig['train']['steps']==400000 and orig['train']['warmup_steps']==10000
    assert orig['train']['lr']==.0005 and orig['train']['min_lr']==.00001
    assert not any(k in orig['train'] for k in ('restarts','original_steps','stop_step'))
    lr=lr_function(orig)
    assert lr(0)==pytest.approx(.0001)
    assert lr(10000)==pytest.approx(1)
    assert lr(200000)>.02 and lr(400000)==pytest.approx(.02)
    assert all(lr(s)>lr(s+10000) for s in range(10000,400000,10000))


def test_complete_graph_protocol_paths_and_rewards_entrypoints(rows):
    assert len(rows)==84 and len({r['id'] for r in rows})==84
    by_id={r['id']:r for r in rows}
    for row in rows:
        needed_inputs(row)
        assert row['host']=='remote' and row['gpu_profile']=='1g.24gb'
        assert row['local_placement_pause']==reg.LOCAL_PAUSE
        assert row['priority']==(-3 if '-pre-' in row['id'] else -1)
        assert not any('0200000.pt' in a or '0300000.pt' in a for a in row['args'])
        assert not any('pre400k' in d for d in row['deps'])
        deferred=row['realworld_variant']=='SF' or row['realworld_target']=='drawer-open-v2'
        assert row['script'].startswith('scripts/dsreset_')==deferred
        assert not any('0'*64 in str(v) for v in row.values())
        if '-fql-' in row['id']:
            seed=int(row['id'][-1]);assert f'train.seed={seed}' in row['args']
            assert {'agent.alpha=10.0','agent.normalize_q_loss=true','train.offline_steps=1000000'}<=set(row['args'])
            hook=row['deployment_hook'];assert hook['seed']==seed and hook['evaluation_id'] in by_id
            assert row['id'] in by_id[hook['evaluation_id']]['deps']
        if '-pre-' in row['id']:
            assert '--source-checkpoint' not in row['args'] and '--resume' in row['args']
            assert ('learning_input_publication' in row)==deferred
            if deferred:
                assert row['learning_input_publication'] in row['deps']
                assert any(d.startswith('rwv2-20261113-prep-') for d in row['deps'])
    pre=[r for r in rows if '-pre-' in r['id']]
    state={d:dict(status='done',host='local') for r in pre if not r.get('learning_input_publication') for d in r['deps']}
    ready=scheduler.eligible(pre,state)
    assert len(ready)==4
    assert {(r['realworld_target'],r['realworld_variant']) for r in ready}=={('covering','S'),('pick-and-place','S')}
    # The publication+statistics completions alone automatically make the other 8 eligible.
    for row in pre:
        for dep in row['deps']:
            state[dep]=dict(status='done',host='local')
    assert len(scheduler.eligible(pre,state))==12


def test_loop_does_not_touch_unrelated_continuations_and_recovers_lift(rows):
    queue=dict(jobs=rows+[dict(id='unrelated',host='remote')])
    ids=loop.pending_ids(queue,{},dict(pauses=[]))
    assert set(ids)=={r['id'] for r in rows}
    placed=copy.deepcopy(rows[0]);placed['host']='local'
    queue=dict(jobs=[placed])
    owned={placed['id']:dict(owner_host='local')}
    assert loop.pending_ids(queue,owned,dict(pauses=[]))==[]
    pauses=dict(pauses=[dict(name=reg.LOCAL_PAUSE,active=True,ids=[placed['id']])])
    assert loop.pending_ids(queue,owned,pauses)==[placed['id']]


def test_recipe_drift_is_rejected(tmp_path):
    import shutil
    shutil.copytree(ROOT/'configs',tmp_path/'configs')
    path=tmp_path/f'{reg.CONFIG}/covering-S-ch01-scratch400k.yaml'
    path.write_text(path.read_text().replace('depth_hard: 0.5','depth_hard: 0.0'))
    with pytest.raises(RuntimeError,match='unapproved recipe drift'):
        reg.validate_recipes(tmp_path)


def test_dsreset_matrix_mode_publishes_dependencies_without_new_continuations(tmp_path):
    import importlib.util
    spec=importlib.util.spec_from_file_location('matrix_dsreset_publication',TOOLS/'register_dsreset_20261113.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    (tmp_path/'experiments').mkdir()
    (tmp_path/'experiments/queue.yaml').write_text('jobs: []\n')
    (tmp_path/'experiments/registry.jsonl').write_text('')
    cfg=dict(methods=['ours'],ours_input_barriers_only=True,tasks={task:dict(
        inputs=f'/home/compu/kaist/sunho/realworld/learning_inputs/dsreset/{task}',
        inputs_copy='/not-published',rig=f'/rig/{task}.json') for task in reg.TASKS})
    path=tmp_path/'spec.json';path.write_text(json.dumps(cfg))
    release=dict(commit='a'*40,image='sha256:'+'b'*64,source='local',evidence_manifest=reg.R.LOCAL_EVIDENCE)
    record=module.prepare(tmp_path,cfg,path,tmp_path/'ingest.json',release,preview=True)
    assert record['ours_input_barriers_only'] and not record['registered']
    rows=record['rows']
    assert len(rows)==15  # 3 task barriers + 2 stats, per task; S reuse removes 2 stats in the real spec
    assert all(r['id'].split('-')[2] in ('prep','calib','qa') for r in rows)
    assert all(r['priority']==-4 for r in rows)
    ids={r['id'] for r in rows}
    templates=dict(jobs=json.loads((ROOT/'tests/fixtures/ours400k_templates.json').read_text()))
    matrix=reg.build_rows(templates,release)
    deferred={d for r in matrix if r.get('learning_input_publication') and '-pre-' in r['id'] for d in r['deps']}
    assert deferred<=ids
    assert record['status']=='preview'
