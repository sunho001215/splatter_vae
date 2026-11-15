"""Place exact S publication IDs with the established local planner; never launch.

Default previews. --execute releases only IDs with verified local ownership.
Repeat as producers close out; unavailable/dependency-blocked rows remain held.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

SOURCE=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(SOURCE))
from ops.jobs import realworld_local as L,host_roles
from stage_dsreset_s_inputs_20261115 import registrar


def validate_ids(ids, rows):
    if not ids or len(ids)!=len(set(ids)):raise ValueError('unique exact IDs required')
    for identity in ids:
        row=rows[identity]
        if (not identity.startswith('rwv2-20261113-') or '-SF' in identity
                or row.get('registration_tool')!='runs/setup/tools/register_dsreset_20261113.py'
                or row.get('local_placement_pause')!=registrar().PAUSE):
            raise ValueError(f'not an S publication row: {identity}')


def lift(root,ids):
    r=registrar()
    if not ids:return []
    with r.publication_locks(root):
        rows={j['id']:j for j in r.R.queue_document((root/'experiments/queue.yaml').read_bytes())['jobs']}
        validate_ids(ids,rows)
        owned=host_roles.read_ownership()
        for identity in ids:
            row,owner=rows[identity],owned[identity]
            r.R.require(row['host']=='local' and owner.get('kind')=='role_placement'
                and owner.get('state')=='placed' and owner.get('owner_host')=='local'
                and owner.get('execution_scope',{}).get('source_commit')==row['planned_commit'],
                'exact local ownership/pin required before lifting placement hold')
        path=root/'runs/setup/admission-pauses.json';doc=json.loads(path.read_bytes())
        holds=[p for p in doc['pauses'] if p.get('name')==r.PAUSE]
        r.R.require(len(holds)==1 and not holds[0].get('prefixes'),'exact-ID S hold required')
        hold=holds[0];hold['ids']=[i for i in hold['ids'] if i not in ids];hold['active']=bool(hold['ids'])
        r.R.atomic_write(path,r.R.json_bytes(doc))
    return ids


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('ids',nargs='+');p.add_argument('--execute',action='store_true')
    a=p.parse_args();r=registrar();root=r.MAIN
    rows={j['id']:j for j in r.R.queue_document((root/'experiments/queue.yaml').read_bytes())['jobs']}
    validate_ids(a.ids,rows)
    if not a.execute:
        from ops.jobs import scheduler
        result=L.plan({'jobs':list(rows.values())},scheduler.read_registry(),host_roles.load_roles(),
            paused_fn=__import__('ops.jobs.admission_pause',fromlist=['active_pauses']).active_pauses,
            allow_pauses=[r.PAUSE])
        print(json.dumps([d for d in result if d['id'] in a.ids],indent=2));return
    r.R.require(SOURCE==root,'execute only the adopted coordinator copy')
    result=L.apply(a.ids,'S dsreset publication: adopted local release',allow_pauses=[r.PAUSE],blocking=True)
    moved=result['moved']
    # Recover a prior interrupted place→unpause without moving or rewriting the row.
    owned=host_roles.read_ownership()
    recover=[i for i in a.ids if rows[i].get('host')=='local'
             and owned.get(i,{}).get('owner_host')=='local']
    result['unpaused']=lift(root,list(dict.fromkeys([*moved,*recover])))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
