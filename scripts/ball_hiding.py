"""Ball-hiding PREP CLI: status/check-data or --entry (no registration/placement/launch)."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None):
    from s4d.rwlearn.ball_hiding import bootstrap
    bootstrap()
    argv = list(sys.argv[1:] if argv is None else argv)
    if '--entry' in argv:
        from scripts import pick_and_place_v2 as producer
        from s4d.rwlearn.ball_hiding import phase_a_remote
        producer.ENTRIES = {**producer.ENTRIES, 'phase-a-remote': phase_a_remote}
        return producer.main(argv, cpu_only=True)
    if any(x in argv for x in ('register', '--execute')):
        raise SystemExit('BH1 is PREP-only: registration is a separate coordinator task')
    from scripts import pnpv2
    return pnpv2.main(argv)


if __name__ == '__main__':
    raise SystemExit(main())
