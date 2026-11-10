"""Free space, the safety margin and the space projections (final footprint and peak) of a download plan.

Footprint per group once complete: plain files keep their size; archives are replaced by their extracted members
(``archive.expanded``), so the final growth of an archive item is ``expanded`` and its archive is transient. The depth
package is one stream: its parts stay on disk until the whole stream is extracted and verified, so the depth peak is
parts + members. The margin is a fraction of the file-system size that must stay available (``f_bavail``).
"""

from __future__ import annotations

import math
import os

from ops.datasets.state import FINISHED


def fs_usage(path) -> dict:
    stat = os.statvfs(path)
    return {"size": stat.f_blocks * stat.f_frsize, "avail": stat.f_bavail * stat.f_frsize}


def margin_bytes(size: int, fraction: float) -> int:
    return math.ceil(size * fraction)


def selected_fraction(item: dict, selection: dict | None) -> float:
    """Share of an archive's members kept by an episode selection (flow shards: exact counts; other packages: the
    corpus-wide share)."""
    if not selection or "archive" not in item:
        return 1.0
    shard = item["archive"].get("shard")
    if shard is not None:
        members = selection["per_shard"].get(shard, 0)
        return members / max(1, item["archive"]["members"])
    return selection["fraction"]


def final_bytes(item: dict, selection: dict | None = None) -> int:
    if "archive" not in item:
        return int(item["size"])
    return math.ceil(item["archive"]["expanded"] * selected_fraction(item, selection))


def projection(items: list[dict], groups, selection: dict | None = None) -> dict:
    """Bytes per group: download, final footprint, and the peak the schedule needs (all parts of the depth stream
    plus its members; one flow shard archive per extraction slot)."""
    out = {}
    for group in groups:
        members = [i for i in items if i["group"] == group]
        download = sum(int(i["size"]) for i in members)
        final = sum(final_bytes(i, selection) for i in members)
        archives = [int(i["size"]) for i in members if "archive" in i]
        if any(i.get("archive", {}).get("kind") == "stream" for i in members):
            peak = final + sum(archives)
        else:
            peak = final + max(archives, default=0)
        out[group] = {"files": len(members), "download": download, "final": final, "peak": peak,
                      "estimate": any(not i.get("archive", {}).get("exact", True) for i in members)}
    return out


def plan_fit(items: list[dict], groups, usage: dict, margin_fraction: float, state_items: dict | None = None,
             selection: dict | None = None) -> dict:
    """Does the remaining work of ``groups`` fit above the margin? Finished items are already in ``usage``."""
    state_items = state_items or {}
    pending = [i for i in items if state_items.get(i["id"], {}).get("status") not in FINISHED]
    proj = projection(pending, groups, selection)
    margin = margin_bytes(usage["size"], margin_fraction)
    usable = usage["avail"] - margin
    final = sum(p["final"] for p in proj.values())
    peak = sum(p["final"] for p in proj.values()) + max(
        [p["peak"] - p["final"] for p in proj.values()], default=0)
    return {"usable": usable, "margin": margin, "final_growth": final, "peak_growth": peak,
            "final_fits": final <= usable, "peak_fits": peak <= usable, "deficit": max(0, final - usable),
            "groups": proj}


def admissible(avail: int, outstanding: int, growth: int, margin: int) -> bool:
    """Admission of one more download/extraction: the space every admitted unfinished step may still claim plus this
    step's claim must leave the margin available."""
    return avail - outstanding - growth >= margin
