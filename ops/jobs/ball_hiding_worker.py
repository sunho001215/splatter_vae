"""Dedicated remote prep publisher entry; explicitly select BH before importing pnp-v2 workers."""
import sys


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in ('rgb', 'geometry'):
        raise SystemExit('expected rgb or geometry')
    from s4d.rwlearn.ball_hiding import bootstrap
    bootstrap()
    if argv[0] == 'rgb':
        from ops.jobs.pnpv2_rgb import main as entry
    else:
        from ops.jobs.pnpv2_geometry import main as entry
    return entry(['remote', *argv[1:]])


if __name__ == '__main__':
    raise SystemExit(main())
