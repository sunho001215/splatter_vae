"""Run one admitted local phase-A episode from its immutable execution bundle."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ops.jobs.local_phase_a import execute  # noqa: E402


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ('bundle', 'job', 'job-sha256', 'gate-sha256'):
        p.add_argument('--' + name, required=True)
    a = p.parse_args(argv)
    # The scheduler launches this wrapper in adopted main; the child imports only
    # frozen sources. The wrapper's entry and the manifest are release pinned.
    from ops.jobs.local_phase_a import verified_job
    import subprocess
    job, envelope = verified_job(a.bundle, a.job, a.job_sha256)
    frozen = Path(envelope['frozen']) / 'source'
    code = ('import sys;sys.path.insert(0,sys.argv.pop(1));'
            'from ops.jobs.local_phase_a import execute;execute(*sys.argv[1:4],gate_sha=sys.argv[4])')
    result = subprocess.run([sys.executable, '-I', '-B', '-c', code, str(frozen), a.bundle, a.job,
                           a.job_sha256, a.gate_sha256], cwd=frozen).returncode
    if result == 0:
        from ops.jobs.local_phase_a import atomic
        import json
        proof = json.loads((Path(job['pipeline']['output_root'])/'local-phase-a.json').read_text())
        atomic(Path('runs')/f"rwv2-20261113-prep-ours-{job['episode']}-localA"/'completion.json', proof)
    return result


if __name__ == '__main__':
    raise SystemExit(main())
