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

DEFAULT_MARGIN = 0.15
# A lower free-space floor is honoured only on these mount points (user decision 2026-11-10: 8.5 % for the dedicated
# /data_8T data disk only); anywhere else the default applies whatever the configuration says.
LOW_MARGIN_MOUNTS = ("/data_8T",)
# Block rounding measured on released members (2026-11-10): cameras JSON 2,051 B/file (REMOTE, 42,935 files), flows
# 2,777 and depth 2,414 B/file (LOCAL sample, 367/294 files), multi-GB files 6-8 KB; the projection uses one block.
FS_OVERHEAD_PER_FILE = 4096


def effective_margin(fraction: float, mount_point: str) -> float:
    if fraction < DEFAULT_MARGIN and mount_point not in LOW_MARGIN_MOUNTS:
        return DEFAULT_MARGIN
    return fraction


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
             selection: dict | None = None, on_disk: dict | None = None, staged: dict | None = None) -> dict:
    """Does the remaining work of ``groups`` fit above the margin? Finished items are already in ``usage``; so are
    the bytes of unfinished items already on disk (``on_disk``: partial files and held archives, which become final
    files or are deleted) and members already staged (``staged`` per group); they are credited."""
    state_items, on_disk, staged = state_items or {}, on_disk or {}, staged or {}
    pending = [i for i in items if state_items.get(i["id"], {}).get("status") not in FINISHED]
    proj = projection(pending, groups, selection)
    for group, value in proj.items():
        value["on_disk"] = sum(on_disk.get(i["id"], 0) for i in pending if i["group"] == group) + staged.get(group, 0)
        value["final"] -= value["on_disk"]
    margin = margin_bytes(usage["size"], margin_fraction)
    usable = usage["avail"] - margin
    final = sum(p["final"] for p in proj.values())
    peak = sum(p["final"] for p in proj.values()) + max(
        [p["peak"] - p["final"] for p in proj.values()], default=0)
    return {"usable": usable, "margin": margin, "final_growth": final, "peak_growth": peak,
            "final_fits": final <= usable, "peak_fits": peak <= usable, "deficit": max(0, final - usable),
            "groups": proj}


def peak_table(items: list[dict], groups, usage: dict, margin_fraction: float, state_items: dict, on_disk: dict,
               staged: dict, stream_published: bool, pool_max: int, flow_members: int, selection=None) -> dict:
    """The space table of the remaining work, term by term (bytes, relative to the current free space):

    - final growth of every enabled group (credits for bytes already on disk), plus block rounding
      (``FS_OVERHEAD_PER_FILE`` per file still to be written: plain files, flow and depth members);
    - transients: the depth parts held until the stream is published, ``pool_max`` in-flight flow shard archives
      (temp files of plain files become the final files themselves; depth parts are already counted as held), and
      one more flow shard archive held during its extraction (its staged members are final bytes);
    - ``peak_unordered``: everything finishes before the depth stream is published (all parts held at the end);
    - ``peak_depth_first``: the enforced order (depth parts first; the depth stream claims its remaining parts and
      expansion from the first part on, so other work is admitted only into the space left beside it): the larger of
      the moment the stream is published (all non-flow groups complete, all parts held, at most ``pool_max`` flow
      shards with expansion beside them) and the end (final + ``pool_max`` + 1 shard archives).
    The runner enforces the floor at every admission whatever the order; this table shows the order completes."""
    fit = plan_fit(items, groups, usage, margin_fraction, state_items, selection, on_disk, staged)
    pending = [i for i in items if state_items.get(i["id"], {}).get("status") not in FINISHED and i["group"] in groups]
    files = sum(1 for i in pending if "archive" not in i) + sum(i["archive"].get("members", 0) for i in pending
                                                               if i["group"] == "pw_flows")
    depth = [i for i in pending if i["group"] == "pw_depth"]
    if depth and not stream_published:
        files += flow_members
    overhead = files * FS_OVERHEAD_PER_FILE
    shard_archives = [int(i["size"]) for i in pending if i["group"] == "pw_flows"]
    shard_growth = max((int(i["size"]) + int(i["archive"]["expanded"]) for i in pending if i["group"] == "pw_flows"),
                       default=0)
    held_parts = sum(int(i["size"]) for i in depth) if not stream_published else 0
    inflight = pool_max * max(shard_archives, default=0)
    extraction = max(shard_archives, default=0)
    final = fit["final_growth"] + overhead
    # until the stream is published: every group but the flows may complete (bound), the depth parts are all held,
    # and at most pool_max flow shards (archive + expansion) are admitted beside them
    others = sum(v["final"] for g, v in fit["groups"].items() if g != "pw_flows")
    at_publish = others + held_parts + pool_max * shard_growth + overhead if depth and not stream_published else 0
    at_end = final + (pool_max + 1) * max(shard_archives, default=0)
    terms = {"final_growth": fit["final_growth"], "fs_overhead": overhead, "held_depth_parts": held_parts,
             "inflight_shard_archives": inflight, "shard_archive_during_extraction": extraction}
    peaks = {"peak_unordered": final + held_parts + inflight + extraction,
             "peak_depth_first": max(at_publish, at_end), "depth_first_at_stream_publish": at_publish,
             "depth_first_at_end": at_end}
    usable = fit["usable"]
    return {"usable": usable, "margin": fit["margin"], "avail": usage["avail"], **terms,
            "final_with_overhead": final, **peaks, "fits": {k: v <= usable for k, v in
                                                            {"final": final, **peaks}.items()},
            "headroom": {k: usable - v for k, v in {"final": final, **peaks}.items()}, "groups": fit["groups"]}


def admissible(avail: int, outstanding: int, growth: int, margin: int) -> bool:
    """Admission of one more download/extraction: the space every admitted unfinished step may still claim plus this
    step's claim must leave the margin available."""
    return avail - outstanding - growth >= margin
