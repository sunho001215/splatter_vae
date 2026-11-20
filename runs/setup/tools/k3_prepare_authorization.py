"""Prepare K3's exact-nine grant in its private evidence directory; never publish."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import k0_isolated_release as release

ROOT = Path('/home/ws/ws/realworld-ops/k0-20261120/k3')


def main():
    inventory_path = ROOT / 'evidence/exact-nine-targeted-evidence.json'
    inventory = json.loads(inventory_path.read_bytes())
    # Inventory records measured evidence, not the runtime's full-suite schema.
    inventory.pop('deployment_authorization', None)
    inventory['deployment_executed'] = False
    inventory['reason'] = 'K/K2 targeted evidence inventory. Separate exact-ID adapter acceptance; no full-suite claim.'
    inventory_path.write_text(json.dumps(inventory, indent=2, sort_keys=True) + '\n')
    snapshot_path = ROOT / 'releases' / release.COMMIT / 'snapshot.json'
    adapter = Path(release.__file__).resolve()
    remote = [i for i, spec in inventory['rows'].items() if spec['row']['host'] != 'local']
    if remote:
        raise ValueError('BLOCKED: three registered remote rows need explicit placement preparation approval; '
                         'no operational authorization published: ' + ', '.join(remote))
    grant = {'kind': 'k0-isolated-targeted-v1', 'active': True, 'commit': release.COMMIT,
             'host': 'local', 'rows': inventory['rows'],
             'accepted_by': 'policy owner decision quoted in targeted evidence inventory',
             'scope': 'exactly nine rows; immutable science plus separately hash-pinned operational entry adapter',
             'adapter_sha256': release.digest(adapter.read_bytes()),
             'worker_sha256': release.digest(adapter.with_name('k0_isolated_worker.py').read_bytes()),
             'snapshot': str(snapshot_path), 'snapshot_sha256': release.digest(snapshot_path.read_bytes()),
             'targeted_evidence': str(inventory_path),
             'targeted_evidence_sha256': release.digest(inventory_path.read_bytes()),
             'execution_environment': '/home/ws/ws/splatter4d/.venv/bin/python; same venv as k2-native-01',
             'full_suite_claimed': False, 'published': False}
    path = ROOT / 'evidence/authorization.json'
    path.write_text(json.dumps(grant, indent=2, sort_keys=True) + '\n')
    release.load_authorization(path)
    print(json.dumps({'prepared': str(path), 'sha256': release.digest(path.read_bytes()), 'ids': len(release.IDS)}))


if __name__ == '__main__':
    main()
