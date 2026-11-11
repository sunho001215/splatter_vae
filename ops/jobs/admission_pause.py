"""Intentional admission pauses (user directive 2026-10-28): rows listed in runs/setup/admission-pauses.json (by id or
id prefix, pause `active`) are never admitted, retried, relaunched or moved by the campaign drivers. Fails closed: an
unreadable pause file raises, so the driver pass admits nothing.

    from admission_pause import paused
    candidates = [j for j in candidates if not paused(j['id'])]
"""
import json
from pathlib import Path

FILE = Path('/home/ws/ws/splatter4d/runs/setup/admission-pauses.json')
_SPEC = json.loads(FILE.read_text()) if FILE.is_file() else {'pauses': []}
_ACTIVE = [p for p in _SPEC['pauses'] if p.get('active') is True]


def paused(job_id: str) -> bool:
    base = job_id.removesuffix('-eval')
    return any(i in p.get('ids', []) or any(i.startswith(x) for x in p.get('prefixes', []))
               for p in _ACTIVE for i in (job_id, base))


def active_pauses(job_id: str, *, fresh: bool = False) -> list[str]:
    """Names of active pauses matching the id or its trainer base, exactly as ``paused`` matches them.

    Placement may explicitly allow named pauses without lifting them: the admission drivers still hold these rows.
    An unnamed pause cannot be bypassed by naming another pause. Placement uses ``fresh`` under its locks so a
    coordinator pause added since the read-only plan is also respected; ordinary callers share paused()'s snapshot.
    """
    base = job_id.removesuffix('-eval')
    pauses = [p for p in json.loads(FILE.read_text())['pauses'] if p.get('active') is True] \
        if fresh and FILE.is_file() else ([] if fresh else _ACTIVE)
    return [p.get('name', '') for p in pauses
            if any(i in p.get('ids', []) or any(i.startswith(x) for x in p.get('prefixes', []))
                   for i in (job_id, base))]
