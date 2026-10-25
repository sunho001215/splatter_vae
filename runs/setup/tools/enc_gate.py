"""Encoder-pilot admission gate (user directive 2026-10-25: new rows use free capacity only, strictly below every
registered row). Encoder-pilot rows (enc-*) may take capacity only when no earlier-registered admissible row is still
pending: the rows of the registrations the local baseline driver admits (the active campaign registrations, the same
list as baseline_local_driver.py; the pool moves their unstarted rows between hosts). A row counts while it is pending
(or crashed with restarts left) and none of its dependencies failed. Rows elsewhere in the queue that no driver admits
(superseded or held baselines, gated specifications) never block. Used by remote_campaign_driver.py; the local baseline
driver applies the same rule inline.
"""
import json
from pathlib import Path

SETUP = Path('/home/ws/ws/splatter4d/runs/setup')
REGISTRATIONS = ['baselines-sanity-registration-20261016.json', 'baselines-eight-task-registration-20261017.json',
                 'baselines-snerl-eight-task-registration-20261020.json', 'afk2-registration-hammer-pick-place-20261020.json',
                 'afk2-registration-bin-picking-coffee-push-door-open-peg-insert-side-peg-unplug-side-stick-push-20261020.json',
                 'afk2rerun-seed0-bin-picking-200k-registration-20261021.json',
                 'chl-registration-hammer-pick-place-20261021.json', 'chl-registration-eight-task-20261022.json',
                 'mpf-registration-20261023.json', 'vision-only-registration-20261023.json',
                 'chlrm-registration-eight-task-20261024.json']
ONLY = {'baselines-sanity-registration-20261016.json': {'bxs-dynamo-hammer-rand-s1000-eval'}}


def earlier_pending(state: dict, max_restarts: dict | None = None) -> list[str]:
    """Ids of earlier-registered admissible rows that are still waiting (state: the scheduler registry view)."""
    out = []
    for name in REGISTRATIONS:
        path = SETUP / name
        if not path.is_file():
            continue
        for job in json.loads(path.read_text())['jobs']:
            if name in ONLY and job['id'] not in ONLY[name]:
                continue
            current = state.get(job['id'], {})
            status = current.get('status', 'pending')
            restartable = status == 'crashed' and current.get('failures', 0) <= int(job.get('max_restarts', 1))
            if (status == 'pending' or restartable) and not any(
                    state.get(d, {}).get('status') == 'failed' for d in job.get('deps', [])):
                out.append(job['id'])
    return out
