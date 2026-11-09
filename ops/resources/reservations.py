"""Read transient RAM reservations without reserving resources at import time."""

from pathlib import Path


def validation_reservation():
    import json as _json
    import time as _time
    try:
        doc = _json.loads((Path('/home/ws/ws/splatter4d/runs/setup/local-validation-reservation.json')).read_text())
    except FileNotFoundError:
        return 0
    return int(doc['ram_bytes']) if _time.time() < float(doc['expires_unix']) else 0

