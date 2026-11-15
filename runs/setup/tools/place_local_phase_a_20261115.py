"""Established local placement followed by exact-ID hold release; dry-run default.
RAM capacity comes from realworld_local and the shared runtime memory_budget control."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 's4d').is_dir())
sys.path[:0] = [str(ROOT), str(Path(__file__).resolve().parent)]
from ops.jobs import host_roles, realworld_local as rw  # noqa: E402
import register_local_phase_a_20261115 as reg  # noqa: E402


@reg.input_verification_pass()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('ids', nargs='+')
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    reg.a.require(Path.cwd().resolve() == ROOT == reg.a.MAIN, 'placement requires adopted main')
    by_id = {j['id']: j for j in reg.reviewed.queue_document((ROOT/'experiments/queue.yaml').read_bytes())['jobs']}
    reg.a.require(all(by_id[i].get('registration_tool') == reg.TOOL for i in args.ids), 'unexpected row')
    if not args.execute:
        return subprocess.run([sys.executable, '-I', 'ops/jobs/realworld_local.py', 'place', '--dry-run',
                               '--allow-pause', reg.PAUSE, *args.ids]).returncode
    result = rw.apply(args.ids, 'Qualified campaign-container local phase A', allow_pauses=[reg.PAUSE], blocking=True)
    with reg.reviewed.locks(ROOT):
        by_id = {j['id']: j for j in reg.reviewed.queue_document((ROOT/'experiments/queue.yaml').read_bytes())['jobs']}
        owned = host_roles.read_ownership()
        path = ROOT / 'runs/setup/admission-pauses.json'
        pauses = json.loads(path.read_text())
        holds = [h for h in pauses['pauses'] if h['name'] == reg.PAUSE]
        reg.a.require(len(holds) == 1 and not holds[0].get('prefixes'), 'unexpected placement hold')
        moved = {i for i in args.ids if by_id[i]['host'] == 'local' and owned.get(i, {}).get('owner_host') == 'local'
                 and owned[i].get('kind') == 'role_placement' and owned[i].get('state') == 'placed'}
        hold = holds[0]
        hold['ids'] = [i for i in hold['ids'] if i not in moved]
        hold['rows'] = list(hold['ids'])
        hold['active'] = bool(hold['ids'])
        reg.a.atomic(path, pauses)
    print(json.dumps({'placement': result, 'released': sorted(moved), 'launched': False}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
