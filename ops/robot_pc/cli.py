"""Shared command-line plumbing of the scripts/robot_pc entry points: arguments, policy selection from the server's
WELCOME, recorded / live session setup. Never constructs a live sink (only ``execute.py`` asks for one, and it is
refused)."""

from __future__ import annotations

import argparse
import json
import socket
from pathlib import Path

from ops.robot_pc import config as C
from ops.robot_pc import selection
from ops.robot_pc.safety import admit_endpoint, safe_output

SAFETY = {"robot_commands_sent": 0, "controller_binding": "message-format-only", "execution_permitted": False,
          "note": "dry run / replay only: no robot, arm, gripper or controller was contacted"}


def add_common(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--config", type=Path, default=None, help="robot-PC config (default: the repo's / setup copy)")
    ap.add_argument("--root", default=None, help="robot-PC folder (default: config root)")
    ap.add_argument("--policy", default=None, help="policy id offered by the server (or --condition/--seed)")
    ap.add_argument("--condition", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--task", default=None, help="task of the policy (default: the server's catalog info)")
    ap.add_argument("--camera", default=None, help="single-view camera serial (default: first trained camera)")
    ap.add_argument("--identity-depth", action="store_true",
                    help="3D-MVP: no native depth correction (explicit, labelled); default = rig sensor_depth scale")
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--token-file", default=None)
    ap.add_argument("--step-timeout-s", type=float, default=None)
    ap.add_argument("--noise-seed", type=int, default=None, help="validation-mode FQL noise (seed base + step)")
    ap.add_argument("--no-pad", action="store_true", help="refuse run starts instead of the training padding")


def load_config(args) -> dict:
    path = args.config
    if path is None:
        root = Path(args.root or "~/workspace/s4d-robot-pc").expanduser()
        path = root / "config/robot_pc.yaml" if (root / "config/robot_pc.yaml").is_file() else C.DEFAULT_PATH
    return C.load(path, root=args.root)


def welcome(cfg: dict, args) -> dict:
    """The server's WELCOME (offered policies + contracts); no session is opened."""
    from ops.policy_serving import protocol as P

    s = cfg.get("server") or {}
    host, port = args.host or s.get("host", "127.0.0.1"), int(args.port or s.get("port", 47100))
    admit_endpoint(host, port)
    sock = socket.create_connection((host, port), timeout=float(s.get("connect_timeout_s", 3.0)))
    wire = P.Wire(sock)
    try:
        msg = wire.recv(float(s.get("connect_timeout_s", 3.0)))
    finally:
        wire.close()
    if msg is None or msg.type != P.WELCOME:
        raise RuntimeError(f"no WELCOME from {host}:{port}")
    return msg.header


def choose_policy(cfg: dict, args) -> tuple[str, str, dict, dict]:
    """(policy id, task, unbound input spec, catalog info): explicit id or the unique deployable match; the task from
    the catalog info unless given."""
    offered = welcome(cfg, args)["policies"]
    pid = selection.select_policy(offered, args.policy, condition=args.condition, task=args.task, seed=args.seed)
    task = args.task or (offered[pid].get("info") or {}).get("task")
    if task is None:
        raise SystemExit(f"{pid}: the server gives no task; pass --task")
    C.task_entry(cfg, task)
    return pid, task, offered[pid]["input_spec"], dict(offered[pid].get("info") or {})


def rig_document(cfg: dict, task: str) -> Path | None:
    doc = C.task_entry(cfg, task).get("rig_document")
    return C.path(cfg, doc)


def connect(cfg: dict, args, pid: str, choose, name: str):
    """The persistent session; a server that requires the shared token while the token file is missing fails closed
    with the path to fill (the token is never read from anywhere else, printed or passed on a command line)."""
    from ops.policy_serving.client import ServingError

    client = C.client(cfg, pid, choose, name=name, token_file=args.token_file, host=args.host, port=args.port,
                      step_timeout_s=args.step_timeout_s)
    try:
        client.connect()
    except ServingError as exc:
        client.close()
        if exc.code == "auth":
            tok = C.path(cfg, args.token_file or (cfg.get("server") or {}).get("token_file"))
            raise SystemExit(f"REFUSED: the policy server requires the shared token; token file {tok} is "
                             f"{'present but rejected' if tok is not None and tok.is_file() else 'MISSING'} "
                             "(user action: copy it there, mode 0600)") from exc
        raise
    return client


def write_json(path: Path | None, obj) -> None:
    text = json.dumps(obj, indent=1, allow_nan=False, default=str)
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        safe_output(path).write_text(text + "\n")
    print(text)


class JsonlRows:
    def __init__(self, path: Path | None):
        self.f = None
        if path is not None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self.f = open(safe_output(path), "w")

    def __call__(self, row: dict) -> None:
        if self.f is not None:
            self.f.write(json.dumps(row, allow_nan=False, default=float) + "\n")

    def close(self) -> None:
        if self.f is not None:
            self.f.close()
