"""Machine-readable restart state for the local outage of 2026-10-31 23:00 KST: paused local jobs (intentional stops
with the shutdown reason) with checkpoint/replay paths and resume commands, local jobs left pending, remote jobs
running at handoff, the remote runner plan summary, controllers stopped, worktrees/branches. Writes
runs/setup/restart-state-20261031.json and uploads it (and docs/RESTART_BRIEFING_20261031.md if present) to
~/kaist/sunho/remote_runner/.

    python -I write_restart_state_20261031.py [--no-upload]
"""
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path('/home/ws/ws/splatter4d')
sys.path.insert(0, str(ROOT))
from scripts import jobs  # noqa: E402

REASON = 'local_shutdown_20261031_maintenance'
queue = jobs.load_queue()
by_id = {j['id']: j for j in queue['jobs']}
state = jobs.read_registry()


def git(*args):
    return subprocess.run(['git', '-C', str(ROOT), *args], capture_output=True, text=True).stdout.strip()


paused, local_active, remote_active = [], [], []
for jid, cur in state.items():
    stops = [s for s in cur.get('intentional_stops', []) if s.get('reason') == REASON]
    if stops and cur.get('attempt') == stops[-1].get('attempt') and cur.get('host', 'local') == 'local':
        job = by_id.get(jid, {})
        run = ROOT / 'runs' / jid
        paused.append({
            'id': jid, 'status_in_registry': cur['status'], 'attempt': cur['attempt'],
            'checkpoint': stops[-1].get('checkpoint'), 'checkpoint_step': stops[-1].get('checkpoint_step'),
            'checkpoint_sha256': stops[-1].get('checkpoint_sha256'),
            'replay': str((run / 'replay').relative_to(ROOT)) if (run / 'replay').exists() else None,
            'script': job.get('script'), 'args': job.get('args'), 'commit_at_launch': cur.get('commit'),
            'resume': 'automatic: a crashed row with an intentional stop and failures <= max_restarts is retried by '
                      'the local driver, whose command line carries --resume auto (pretraining) / resumes from '
                      'checkpoints/latest.pt with the replay snapshot (FlashSAC)',
            'manual_command': f".venv/bin/python -I {job.get('script')} {' '.join(map(str, job.get('args', [])))}"})
    if cur.get('host', 'local') == 'local' and cur.get('status') in jobs.ACTIVE:
        local_active.append(jid)
    if cur.get('host') == 'remote' and cur.get('status') in jobs.ACTIVE:
        remote_active.append({'id': jid, 'container': cur.get('container'), 'gpu': cur.get('gpu'), 'commit': cur.get('commit')})
plan_path = ROOT / 'runs/remote/autonomy-20261031/plan.json'
plan = json.loads(plan_path.read_text()) if plan_path.is_file() else {'items': []}
worktrees = git('worktree', 'list')
doc = {
    'written': datetime.now().astimezone().isoformat(timespec='seconds'),
    'main_checkout': {'branch': git('rev-parse', '--abbrev-ref', 'HEAD'), 'commit': git('rev-parse', 'HEAD'),
                      'origin_splatter4d': git('rev-parse', 'refs/remotes/origin/splatter4d')},
    'paused_local_jobs': paused,
    'local_jobs_still_active_at_write': local_active,
    'remote_jobs_active_at_write': remote_active,
    'remote_runner': {'dir': '/home/compu/kaist/sunho/remote_runner', 'plan_items': len(plan['items']),
                      'first_items': [i['id'] for i in plan['items'][:12]],
                      'gpu_load_limit_default': 5.0, 'stop_file': 'remote_runner/STOP',
                      'start_command': 'cd ~/kaist/sunho/remote_runner && setsid nohup python3 remote_runner.py '
                                       '>> runner.out 2>&1 < /dev/null &',
                      'events': 'remote_runner/events.jsonl', 'heartbeat': 'remote_runner/heartbeat.json',
                      'reconcile': 'runs/setup/tools/reconcile_remote_runner_20261031.py --stop; then --apply'},
    'admission_pauses_active': [p['name'] for p in json.loads((ROOT / 'runs/setup/admission-pauses.json').read_text())['pauses']
                                if p.get('active')],
    'local_holds': ['experiments/HOLD (since 2026-10-16)', 'local driver loops stopped 2026-10-31 19:48'],
    'worktrees': worktrees.splitlines(),
}
out = ROOT / 'runs/setup/restart-state-20261031.json'
out.write_text(json.dumps(doc, indent=1) + '\n')
print(json.dumps({'paused': [p['id'] for p in paused], 'local_active': local_active,
                  'remote_active': len(remote_active)}, indent=1))
if '--no-upload' not in sys.argv:
    from s4d.remote_access import RemoteClient  # noqa: E402
    client = RemoteClient()
    client.rsync_upload(out, '~/kaist/sunho/remote_runner/restart-state-20261031.json')
    brief = ROOT / 'docs/RESTART_BRIEFING_20261031.md'
    if brief.is_file():
        client.rsync_upload(brief, '~/kaist/sunho/remote_runner/RESTART_BRIEFING_20261031.md')
    print('uploaded')
