"""Live intentional admission pauses. Unreadable evidence fails closed.

Long-lived consumers can take one ``snapshot()`` per admission pass. Ordinary
calls read current evidence, including callers imported before a pause changes.
"""
import json
from pathlib import Path

FILE = Path('/home/ws/ws/splatter4d/runs/setup/admission-pauses.json')
PREP_PRIORITY = Path('/home/ws/ws/splatter4d/runs/setup/rwprep-mig-priority.json')


def snapshot() -> list[dict]:
    spec = json.loads(FILE.read_text()) if FILE.is_file() else {'pauses': []}
    pauses = [p for p in spec['pauses'] if p.get('active') is True]
    if PREP_PRIORITY.exists():
        prep = json.loads(PREP_PRIORITY.read_text())
        if prep['active'] is True:
            pauses.append({'name':'rwprep-mig-priority','prefixes':['']})
    return pauses


def active_pauses(job_id: str, *, fresh: bool = True, pauses=None) -> list[str]:
    """Match the row and its trainer base; ``fresh`` remains accepted for compatibility.

    There is no import-time cache. Pass an explicit snapshot for a coherent tick.
    """
    base = job_id.removesuffix('-eval')
    return [p.get('name', '') for p in (snapshot() if pauses is None else pauses)
            if any(i in p.get('ids', []) or any(i.startswith(x) for x in p.get('prefixes', []))
                   for i in (job_id, base))]


def paused(job_id: str, *, pauses=None) -> bool:
    return bool(active_pauses(job_id, pauses=pauses))
