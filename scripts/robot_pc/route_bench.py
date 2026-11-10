"""Development host (LOCAL): ONE command for the robot-PC latency benchmark over the real route, run ON the robot PC
through the existing ``robotpc`` ssh master (BatchMode, never authenticates), with a read-only GPU snapshot of the 5090
before and after. Never commands a robot: the robot PC runs scripts/robot_pc/bench.py (recorded replay -> server ->
validation -> MockSink).

    .venv/bin/python -I scripts/robot_pc/route_bench.py --label contended      # or: --label uncontended
    # -> /home/ws/ws/realworld-ops/serving/rob-bench/<label>-<YYYYmmdd-HHMMSS>/{summary.json,table.md,<policy>.json}

REPLAY ONLY (no camera is opened): per policy (default: one per condition x task, deployable or restricted; 6 today)
15 Hz paced + max-rate replay of 150 frames of a recording ON the robot PC (10 warm-up steps discarded), about 3-5 min
in total. Refuses before starting and before every policy when the robot PC runs a recorder / teleop /
switch_controller process, holds a camera device or has the collector IPC (``devices.require_free`` there), PING round trips robot PC -> server, and once a LOCAL ->
server PING set (the difference estimates the robot PC <-> LOCAL relay hop). ``--label`` must say whether the GPU was
shared: the snapshot lists every process on the GPU (foreign = not the policy server) so the label can be checked;
contended and uncontended results always go to separate directories.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

ROUTE = ("robotpc 127.0.0.1:47100 -> reverse forward over the robotpc ssh master -> LOCAL 127.0.0.1:47100 -> irsl ssh "
         "master -> shkim_rw:47100 (RTX 5090 GPU-9bcf38b6)")
GPU = "GPU-9bcf38b6-8240-e910-6548-ca715643dc11"
ROBOT_ROOT = "/home/irsl/workspace/s4d-robot-pc"
DATA = "/home/irsl/workspace/panda-teleop/teleop_collector/data"
EPISODES = {"drawer-open": f"{DATA}/drawer-open/episode_000000.h5", "mug-hanging": f"{DATA}/mug-hanging/episode_000001.h5"}
LOCAL_TOKEN = Path("/tmp/claude-0/-home-ws/738aaa26-ed64-46bb-9068-b4ef624cd2ac/scratchpad/irsl-serving.token")
SSH = ["ssh", "-o", "BatchMode=yes"]


def ssh(host: str, command: str, timeout: float) -> str:
    r = subprocess.run([*SSH, host, command], capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"ssh {host} failed ({r.returncode}): {r.stderr.strip()[-500:]}")
    return r.stdout


def robot_pc_idle() -> None:
    """Refuse (SystemExit) unless the robot PC runs no recorder / teleop / switch_controller process, holds no camera
    device and has no collector IPC (``ops.robot_pc.devices.require_free`` on the robot PC, read-only /proc scan):
    the benchmark must never load the robot control machine while it controls or records."""
    cmd = (f"cd {ROBOT_ROOT} && venv/bin/python -I -c 'import sys; sys.path.insert(0, \"src\"); "
           "from ops.robot_pc import devices; devices.require_free(); print(\"IDLE\")'")
    r = subprocess.run([*SSH, "robotpc", cmd], capture_output=True, text=True, timeout=60)
    if r.returncode != 0 or "IDLE" not in r.stdout:
        raise SystemExit(f"REFUSED: the robot PC is busy (recording / teleop / devices): {r.stderr.strip()[-600:]}")


def gpu_snapshot() -> dict:
    """Read-only: utilisation / memory of the serving GPU and every compute process on it (with its host user)."""
    q = ssh("irsl", f"nvidia-smi -i {GPU} --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,"
                    "nounits; echo ---; nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory "
                    "--format=csv,noheader,nounits", 60)
    head, apps = q.split("---")
    util, used, total = (x.strip() for x in head.strip().split(","))
    procs = []
    for line in apps.strip().splitlines():
        uuid, pid, mem = (x.strip() for x in line.split(","))
        if uuid != GPU:
            continue
        info = ssh("irsl", f"ps -o user=,etime=,args= -p {int(pid)} || true", 30).strip()
        user, _, rest = info.partition(" ")
        procs.append({"pid": int(pid), "used_mib": int(mem), "user": user, "ps": rest.strip()[:160],
                      "policy_server": "ops/policy_serving/server.py" in rest})
    return {"t": time.strftime("%Y-%m-%d %H:%M:%S %z"), "gpu": GPU, "util_pct": int(util), "mem_used_mib": int(used),
            "mem_total_mib": int(total), "processes": procs,
            "foreign_processes": sum(not p["policy_server"] for p in procs)}


def local_pings(policy: str, task: str, n: int) -> dict:
    """LOCAL -> server PING round trips in one short session (no step), for the hop decomposition."""
    import argparse as ap

    from ops.robot_pc import cli
    from ops.robot_pc import config as C
    from ops.robot_pc.loop import percentiles

    cfg = C.load(root=REPO)
    args = ap.Namespace(policy=policy, condition=None, seed=None, task=task, host=None, port=None,
                        token_file=str(LOCAL_TOKEN), step_timeout_s=None)
    _, _, spec, _ = cli.choose_policy(cfg, args)
    cam = spec["trained_camera_orders"][0][0]
    client = cli.connect(cfg, args, policy, {"native_hw": {cam: [480, 640]}, "camera_id": cam}, "route-bench-local")
    try:
        return percentiles([client.ping() for _ in range(n)])
    finally:
        client.close()


def default_policies(offered: dict) -> list:
    """One seed-0 (else lowest seed) policy per condition x task with a robot-PC recording, deployable or restricted
    (restricted = dry-run / replay only, e.g. DynaMo; benchmarking is replay)."""
    best = {}
    for pid, e in sorted(offered.items()):
        info = e.get("info") or {}
        key = (info.get("condition"), info.get("task"))
        if info.get("status") not in ("deployable", "restricted") or info.get("alias_of") or key[1] not in EPISODES:
            continue
        if key not in best or (info.get("seed") or 0) < best[key][1]:
            best[key] = (pid, info.get("seed") or 0)
    return [v[0] for _, v in sorted(best.items())]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="route_bench.py", description=__doc__.split("\n")[0])
    ap.add_argument("--label", choices=("contended", "uncontended"), required=True)
    ap.add_argument("--policies", default=None, help="comma-separated ids (default: one per condition x task)")
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--pings", type=int, default=30)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args(argv)
    out = args.out_dir or Path("/home/ws/ws/realworld-ops/serving/rob-bench") / \
        f"{args.label}-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=False)
    if "OK" not in ssh("robotpc", f"test -s {ROBOT_ROOT}/secrets/token && echo OK || echo MISSING", 30):
        raise SystemExit(f"REFUSED: {ROBOT_ROOT}/secrets/token is missing on the robot PC (user action)")
    robot_pc_idle()
    from ops.robot_pc import cli
    from ops.robot_pc import config as C

    welcome = cli.welcome(C.load(root=REPO), argparse.Namespace(host=None, port=None))
    policies = args.policies.split(",") if args.policies else default_policies(welcome["policies"])
    summary = {"label": args.label, "route": ROUTE, "robot_commands_sent": 0, "policies": policies,
               "gpu_before": gpu_snapshot(), "runs": {}}
    first = welcome["policies"][policies[0]]["info"]
    summary["local_ping_ms"] = local_pings(policies[0], first["task"], args.pings)
    for pid in policies:
        robot_pc_idle()  # re-checked before every policy: stop as soon as a recording / teleop session starts
        task = (welcome["policies"][pid].get("info") or {}).get("task")
        cmd = (f"cd {ROBOT_ROOT} && nice -n 10 venv/bin/python -I src/scripts/robot_pc/bench.py "
               f"--episode {shlex.quote(EPISODES[task])} --policy {shlex.quote(pid)} --task {shlex.quote(task)} "
               f"--steps {args.steps} --warmup {args.warmup} --pings {args.pings} --route {shlex.quote(ROUTE)}")
        t0 = time.time()
        try:
            text = ssh("robotpc", cmd, 1800)
            run = json.loads(text[text.index("{"):])
        except (RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
            run = {"error": str(exc)[-800:]}
        run["wall_s"] = round(time.time() - t0, 1)
        (out / f"{pid}.json").write_text(json.dumps(run, indent=1) + "\n")
        summary["runs"][pid] = run
    summary["gpu_after"] = gpu_snapshot()
    if args.label == "uncontended" and (summary["gpu_before"]["foreign_processes"] or
                                        summary["gpu_after"]["foreign_processes"]):
        summary["label_check"] = "WARNING: foreign GPU processes present: these results are CONTENDED"
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    (out / "table.md").write_text(table(summary) + "\n")
    print(out)
    print(table(summary))
    return 0


def _p(d, k):
    v = (d or {}).get(k) or {}
    return "-" if not v else f"{v['p50']:.1f}/{v['p95']:.1f}/{v['p99']:.1f}"


def table(s: dict) -> str:
    g0, g1 = s["gpu_before"], s["gpu_after"]
    lines = [f"# robot-PC route benchmark ({s['label'].upper()})", "", f"route: {s['route']}", "",
             f"GPU before: util {g0['util_pct']} %, {g0['mem_used_mib']} MiB, foreign processes "
             f"{g0['foreign_processes']}; after: util {g1['util_pct']} %, foreign {g1['foreign_processes']}",
             f"LOCAL -> server PING p50/p95/p99 ms: {_p({'x': s['local_ping_ms']}, 'x')}", "",
             "| policy | mode | resident before | warm-up discarded | ok/sent | Hz | e2e (preproc->action) p50/p95/p99 | "
             "capture->command p50/p95/p99 | misses e2e / capture | robotPC PING | capture age at send | acquire "
             "(decode+align) | preprocess | serialize | transport | server queue / encoder / FQL / total | "
             "validation | bytes first / steady / response |", "|" + "---|" * 17]
    for pid, run in s["runs"].items():
        if "error" in run:
            lines.append(f"| {pid} | ERROR | {run['error'][:120]} |")
            continue
        for r in run["runs"]:
            c, sv, pb = r["client_ms"], r["server_ms"], r["payload_bytes"]
            res = (r.get("catalog_info") or {}).get("resident")
            lines.append(
                f"| {pid} | {r['pace']} | {res} | {r['warmup_steps_discarded']} | {r['ok']}/{r['steps_sent']} | "
                f"{(r['achieved_hz'] or 0):.1f} | {_p(c, 'e2e_ms')} | {_p(c, 'capture_to_command_ms')} | "
                f"{r['missed_deadline']['e2e_ms']} / {r['missed_deadline']['capture_to_command_ms']} | "
                f"{_p({'x': r.get('ping_ms')}, 'x')} | {_p(c, 'capture_age_at_send_ms')} | {_p(c, 'acquire_ms')} | "
                f"{_p(c, 'preprocess_ms')} | {_p(c, 'serialize_ms')} | {_p(c, 'network_ms')} | "
                f"{_p(sv, 'queue_ms')} / {_p(sv, 'encoder_ms')} / {_p(sv, 'fql_ms')} / {_p(sv, 'total_ms')} | "
                f"{_p(c, 'validation_ms')} | {pb['first_request']} / {pb['steady_request_median']} / "
                f"{pb['response_median']} |")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
