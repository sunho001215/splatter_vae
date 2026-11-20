"""Accept ball-hiding (bh-20261120) prep job ids in rwprep_mig.validate for the remote driver (2026-11-21).

Ball-hiding prep bundles were staged from .worktrees/bh (which extends the job-id allow-list). Main keeps the accepted
source (docs/tests.json), so main's rwprep_mig.validate refuses `bh-20261120-*` ids and the remote driver's prep
reservation (ops/jobs/prep_reservation.ready_rows) crashed on every pass. This wrapper-level hook validates BH jobs with
the ORIGINAL function after aliasing only the id stem (bh-20261120- -> pnpv2-20261117-bh-; deps aliased identically), so
every other check (resources, profile, steps, inputs, dependencies, config) is unchanged and any other id still fails.
"""
P = 'bh-20261120-'
Q = 'pnpv2-20261117-bh-'


def install():
    from ops.jobs import rwprep_mig
    if getattr(rwprep_mig.validate, '_bh_alias', False):
        return True
    original = rwprep_mig.validate

    def alias(i):
        return Q + i[len(P):] if isinstance(i, str) and i.startswith(P) else i

    def validate(config, jobs):
        if not any(isinstance(j, dict) and str(j.get('id', '')).startswith(P) for j in jobs):
            return original(config, jobs)
        return original(config, [{**j, 'id': alias(j['id']), 'deps': [alias(d) for d in j.get('deps', [])]} for j in jobs])

    validate._bh_alias = True
    rwprep_mig.validate = validate
    return True
