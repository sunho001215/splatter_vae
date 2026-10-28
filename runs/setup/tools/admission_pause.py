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
