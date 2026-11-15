"""Official local placement + exact moved-ID hold release; dry-run default, no launch."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT),str(Path(__file__).resolve().parent)]
import register_drawer_rgb_20261115 as reg
from ops.jobs import host_roles, realworld_local as local
from s4d.rwlearn.rgb_inputs import PAUSE, TOOL


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('ids',nargs='+')
    p.add_argument('--execute',action='store_true')
    a = p.parse_args()
    reg.require(Path.cwd().resolve() == ROOT == reg.R.MAIN, 'placement only from adopted main')
    queue = reg.R.queue_document((ROOT/'experiments/queue.yaml').read_bytes())
    by_id = {r['id']:r for r in queue['jobs']}
    reg.require(all(by_id[i].get('registration_tool') == TOOL and by_id[i].get('realworld_variant') == 'S'
                    for i in a.ids), 'unexpected or non-S row')
    if not a.execute:
        return subprocess.run([sys.executable,'-m','ops.jobs.realworld_local','place','--dry-run',
                               '--allow-pause',PAUSE,*a.ids]).returncode
    result = local.apply(a.ids,'Drawer-v2 S RGB inputs',allow_pauses=[PAUSE],blocking=True)
    with reg.R.locks(ROOT):
        by_id = {r['id']:r for r in reg.R.queue_document((ROOT/'experiments/queue.yaml').read_bytes())['jobs']}
        ownership = host_roles.read_ownership()
        path = ROOT/'runs/setup/admission-pauses.json'
        pauses = json.loads(path.read_text())
        holds = [h for h in pauses['pauses'] if h['name'] == PAUSE]
        reg.require(len(holds) == 1 and not holds[0].get('prefixes'), 'unexpected placement hold')
        moved = {i for i in a.ids if by_id[i]['host'] == 'local' and ownership.get(i,{}).get('owner_host') == 'local'
                 and ownership[i].get('kind') == 'role_placement' and ownership[i].get('state') == 'placed'}
        hold = holds[0]
        hold['ids'] = [i for i in hold['ids'] if i not in moved]
        hold['rows'],hold['active'] = list(hold['ids']),bool(hold['ids'])
        reg.R.atomic_write(path,reg.R.json_bytes(pauses))
    print(json.dumps(dict(placement=result,released=sorted(moved),launched=False),indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
