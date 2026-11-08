"""Run several worker invocations in one container: python run_many.py MODULE JOBS.json [PARALLEL]
(JOBS.json: list of argument lists for MODULE.main(argv)); PARALLEL > 1 runs that many processes concurrently
(spawned, each loads its own model)."""
import importlib
import json
import multiprocessing as mp
import sys


def _run(args):
    mod_name, argv = args
    print(json.dumps({"run": argv[-6:]}), flush=True)
    importlib.import_module(mod_name).main(argv)


if __name__ == "__main__":
    jobs = json.loads(open(sys.argv[2]).read())
    par = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    if par <= 1:
        for argv in jobs:
            _run((sys.argv[1], argv))
    else:
        with mp.get_context("spawn").Pool(par) as pool:
            pool.map(_run, [(sys.argv[1], a) for a in jobs], chunksize=1)
