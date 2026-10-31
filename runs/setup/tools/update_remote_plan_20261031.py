"""Regenerate the remote autonomy plan and swap it into the running remote runner atomically (the runner reloads
plan.json when its mtime changes): the bundle is unpacked into remote_runner/staging, the new commands directory is
renamed into place, then plan.json is replaced by a rename. Already-launched items are recognised by their container
names, so a regenerated plan never launches an attempt twice.

    python -I update_remote_plan_20261031.py
"""
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path('/home/ws/ws/splatter4d')
sys.path.insert(0, str(ROOT))
from s4d.remote_access import RemoteClient  # noqa: E402

OUT = ROOT / 'runs/remote/autonomy-20261031'
gen = subprocess.run([str(ROOT / '.venv/bin/python'), '-I', str(ROOT / 'runs/setup/tools/gen_remote_autonomy_plan_20261031.py')],
                     capture_output=True, text=True, timeout=1500)
print(gen.stdout[-600:], gen.stderr[-1500:])
if gen.returncode:
    raise SystemExit(gen.returncode)
bundle = OUT / 'update.tgz'
with tarfile.open(bundle, 'w:gz') as tar:
    tar.add(OUT / 'plan.json', arcname='plan.json')
    tar.add(OUT / 'commands', arcname='commands')
client = RemoteClient()
client.rsync_upload(bundle, '~/kaist/sunho/remote_runner/update.tgz')
out = client.checked_ssh('set -e; cd /home/compu/kaist/sunho/remote_runner; rm -rf staging; mkdir staging; '
                         'tar xzf update.tgz -C staging; rm -rf commands.old; mv commands commands.old; '
                         'mv staging/commands commands; mv staging/plan.json plan.json.new; mv plan.json.new plan.json; '
                         'rm -rf commands.old staging; ls commands | wc -l; python3 -c "import json;print(len(json.load(open(\'plan.json\'))[\'items\']))"',
                         timeout=300)
print(out.stdout)
