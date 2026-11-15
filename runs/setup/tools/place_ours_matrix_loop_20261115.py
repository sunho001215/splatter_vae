"""Finite exact-ID matrix placement loop. Dry-run once unless --execute is explicit.

Extends the established continuation placement pattern without replacing a live
controller. Refusals defer only that row, allowing independent ready chains to fit.
A STOP file drains the loop between passes; no training process is signalled.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'s4d').is_dir())
sys.path[:0] = [str(ROOT),str(Path(__file__).resolve().parent)]
import register_ours_matrix_20261115 as reg
import place_ours_matrix_20261115 as place


def pending_ids(queue, owned, pauses):
    """Only this registrar's exact IDs; recover a crash between placement and lift."""
    held={i for h in pauses.get('pauses',[]) if h.get('name')==reg.LOCAL_PAUSE and h.get('active')
          for i in h.get('ids',[])}
    rows=[r for r in queue['jobs'] if r.get('registration_tool')==reg.TOOL]
    by_id={r['id']:r for r in rows}
    return [r['id'] for r in rows if place.valid_id(r['id'],by_id) and
            (r.get('host')!='local' or r['id'] in held) and
            (owned.get(r['id'],{}).get('owner_host')!='local' or r['id'] in held)]


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--loop',type=int,default=0)
    p.add_argument('--stop-file',type=Path,default=reg.MAIN/'runs/setup/STOP-OURS-MATRIX-PLACE')
    a=p.parse_args(argv)
    reg.R.require(Path.cwd().resolve()==ROOT==reg.MAIN,'placement requires released main')
    while not a.stop_file.exists():
        queue=reg.R.queue_document((ROOT/'experiments/queue.yaml').read_bytes())
        pauses=json.loads((ROOT/'runs/setup/admission-pauses.json').read_bytes())
        ids=pending_ids(queue,place.host_roles.read_ownership(),pauses)
        print(json.dumps(dict(pending=ids,execute=a.execute)),flush=True)
        if not ids:
            return 0
        # Official helper processes each requested row under the reviewed admission locks.
        for identity in ids:
            if a.stop_file.exists():
                return 0
            command=[sys.executable,str(ROOT/'runs/setup/tools/place_ours_matrix_20261115.py'),identity]
            if a.execute:
                command+=['--execute']
            result=subprocess.run(command,check=False)
            if result.returncode:
                return result.returncode
        if not a.execute or not a.loop:
            return 0
        deadline=time.monotonic()+a.loop
        while time.monotonic()<deadline and not a.stop_file.exists():
            time.sleep(min(10,max(0,deadline-time.monotonic())))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
