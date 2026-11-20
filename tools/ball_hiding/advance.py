"""Claude-only independent BH branch advancer. Prepare first; never register, place or alter running jobs.

--detach starts RGB and geometry branches in separate sessions. STOP or failed/unknown states hold that branch.
Every action is append/stage/fetch/upload through the established producers. No scheduler daemon is started.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

MAIN=Path(__file__).resolve().parents[2]  # code root = this checkout (isolated bh worktree; main keeps accepted source)
WORK=Path('/home/ws/ws/realworld-ops/ballhiding-20261120')
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from s4d.rwlearn.ball_hiding_membership import excluded_ids, amendment

PY=Path('/home/ws/ws/splatter4d/.venv/bin/python')  # shared venv; code root stays MAIN
TOOLS=MAIN/'tools'/'ball_hiding'


def run(args):
    while True:
        check_stop()
        p=subprocess.run(['nice','-n','19','ionice','-c','3',str(PY),'-I',*map(str,args)],cwd=MAIN, capture_output=True, text=True)
        print(p.stdout, end='', flush=True);print(p.stderr, end='', file=sys.stderr, flush=True)
        if p.returncode==20 or (p.returncode and 'Remote operation exited 20' in p.stderr):
            time.sleep(30)  # admission deferral only, no edits to controller or queue state
            continue
        if p.returncode:raise RuntimeError(f'HELD ({p.returncode}): {args}')
        return


def entry(name,*args):run([MAIN/'scripts/ball_hiding.py','--entry',name,*args])
def tick(bundle):run([TOOLS/'start_tick.py',WORK/bundle])
def check_stop():
    if (WORK/'STOP').exists():raise SystemExit('STOP present')


def done(bundle):
    check_stop()
    if (WORK/bundle/'state/STOP').exists():raise RuntimeError(f'HELD: {bundle}/state/STOP')
    path=WORK/bundle/'state/state.json'
    if not path.exists():return False
    states=json.loads(path.read_text())['jobs']
    if bundle in ('normalize', 'phaseA'):
        stage = 'prepN' if bundle == 'normalize' else 'prepA'
        ignored = {f'bh-20261120-{stage}-{ep}' for ep in excluded_ids()}
        required = {f'bh-20261120-{stage}-ball-hiding-success-' + Path(n).stem
                    for n in amendment()['episodes']}
        unexpected = set(states) - required - ignored
        if unexpected:
            raise RuntimeError(f'HELD: {bundle} unexpected jobs: {sorted(unexpected)}')
        statuses = [(states.get(i) or {}).get('status', 'pending') for i in required]
    else:
        statuses=[s['status'] for s in states.values()]
    if any(x in ('failed','unknown') for x in statuses):raise RuntimeError(f'HELD: {bundle} failed/unknown')
    return bool(statuses) and all(s=='done' for s in statuses)


def wait(bundle,*upstreams):
    while not done(bundle):
        for parent in upstreams:done(parent)  # propagate failed/unknown/STOP; no cross-branch holds
        time.sleep(30)


def rgb():
    if not (WORK/'rgb2-remote/dispatcher.json').exists():entry('rgb-remote')
    tick('rgb2-remote');wait('rgb2-remote','normalize')
    if not (WORK/'publication/S/rgb/store.yaml').exists():entry('rgb-fetch')
    if not (WORK/'publication/S/rgb.UPLOADED.json').exists():entry('upload','--','--view','rgb')
    run([MAIN/'scripts/ball_hiding.py','status'])
    print('RGB branch published; coordinator may proceed with its separate training workflow',flush=True)


def geometry():
    if not (WORK/'calib2/dispatcher.json').exists():entry('calib-remote')
    tick('calib2');wait('calib2','normalize','phaseA')
    if not (Path('/home/ws/data/realworld-processed/bh-20261120/calibration/ball-hiding')/'rig-ball-hiding-accepted.json').exists():
        entry('rig-fetch')
    # phase-b-remote independently checks automatic acceptance; a held rig never proceeds.
    if not (WORK/'phaseB/dispatcher.json').exists():entry('phase-b-remote')
    tick('phaseB');wait('phaseB','phaseA','calib2')
    if not (WORK/'geometry-remote/dispatcher.json').exists():entry('geometry-remote')
    tick('geometry-remote');wait('geometry-remote','phaseB')
    if not (WORK/'publication/S/geometry.FETCHED.json').exists():entry('geometry-fetch')
    if not (WORK/'publication/S/geometry.UPLOADED.json').exists():entry('upload','--','--view','geometry')
    run([MAIN/'scripts/ball_hiding.py','status'])
    print('Geometry branch published; no registration or placement performed',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--branch',choices=('rgb','geometry'));p.add_argument('--detach',action='store_true');a=p.parse_args()
    for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
    os.environ['CUDA_VISIBLE_DEVICES']=''
    if a.detach:
        assert a.branch is None
        (WORK/'logs').mkdir(parents=True,exist_ok=True)
        for branch in ('rgb','geometry'):
            with (WORK/'logs'/f'advance-{branch}.log').open('ab') as log:
                child=subprocess.Popen(['nice','-n','19','ionice','-c','3',str(PY),'-I',str(TOOLS/'advance.py'),'--branch',branch],cwd=MAIN,
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
            (WORK/'logs'/f'advance-{branch}.pid').write_text(str(child.pid)+'\n')
            print(branch,child.pid)
    elif a.branch:
        import fcntl
        (WORK/'logs').mkdir(parents=True,exist_ok=True)
        with (WORK/'logs'/f'advance-{a.branch}.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            {'rgb':rgb,'geometry':geometry}[a.branch]()
    else:p.error('choose --branch or --detach')


if __name__=='__main__':main()
