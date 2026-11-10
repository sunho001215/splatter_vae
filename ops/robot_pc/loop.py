"""One policy step of the robot PC (shared by recorded replay and the live dry run) and the latency summary.

``PolicyLoop.step``: measured state bound to the observation (training rule) -> ``PolicyClient.step`` over the
persistent link -> ``validation.validate_action`` -> the sink (MockSink / DryRunSink; never a live sink here). Every
row carries the client-clock timings: client preprocessing, serialisation, transport (round trip - server time),
server prep / queue / encoder / FQL, action validation, preprocessing->action (``e2e_ms``), capture->action and
capture->validated command, and the payload bytes.
"""

from __future__ import annotations

import collections
import time

import numpy as np

from ops.robot_pc.robot_state import StateUnavailable, require_measured_gripper
from ops.robot_pc.validation import ActionRejected, BoundRequest, validate_action

BUDGET_MS = 1000.0 / 15.0  # one policy step at the trained 15 Hz
CLIENT_KEYS = ("e2e_ms", "capture_to_action_ms", "capture_to_command_ms", "capture_age_at_send_ms", "acquire_ms",
               "preprocess_ms", "serialize_ms", "send_ms", "rtt_ms", "network_ms", "decode_ms", "validation_ms")
SERVER_KEYS = ("prep_ms", "queue_ms", "validation_ms", "encoder_ms", "fql_ms", "policy_ms", "infer_ms", "total_ms")


class PolicyLoop:
    def __init__(self, client, state, limits, sink, *, clock=time.monotonic_ns, pad_run_start: bool = True,
                 executor=None):
        self.client, self.state, self.limits, self.sink = client, state, limits, sink
        self.clock, self.pad_run_start, self.executor = clock, bool(pad_run_start), executor
        self.counts = collections.Counter()

    def step(self, end_ids: dict, observation_time_ns: int, reference_id: str, *, noise_seed: int | None = None,
             acquire_ms: float | None = None) -> dict:
        row = {"reference_id": reference_id, "end_ids": dict(end_ids), "observation_time_ns": int(observation_time_ns),
               "t_step_ns": time.monotonic_ns(), "state_source": self.state.kind, "sent_to_robot": False}
        try:
            st = self.state.bound_state(observation_time_ns)
            require_measured_gripper(st)
        except StateUnavailable as exc:
            self.counts["no_state"] += 1
            return {**row, "status": "skipped", "code": "no_state", "detail": str(exc)}
        row["gripper_origin"] = st.gripper_origin
        req = BoundRequest(reference_id, dict(end_ids), int(observation_time_ns), st.pose7, st.pose_time_ns,
                           st.gripper_register, st.gripper_time_ns)
        try:
            r = self.client.step(end_ids, reference_id, list(st.pose7), st.gripper_register,
                                 pose_time_ns=st.pose_time_ns, gripper_time_ns=st.gripper_time_ns,
                                 pad_run_start=self.pad_run_start, noise_seed=noise_seed)
        except ValueError as exc:  # client-side training binding refusal: nothing was sent
            self.counts["binding_refused"] += 1
            return {**row, "status": "skipped", "code": "binding_refused", "detail": str(exc)}
        t_result = self.clock()
        row.update(status=r.status, code=r.code, detail=r.detail, frames_sent=r.frames_sent,
                   request_bytes=r.request_bytes, response_bytes=r.response_bytes, server_ms=dict(r.server_ms),
                   client_ms=dict(r.timings_ms), terminal=r.terminal)
        if acquire_ms is not None:
            row["client_ms"]["acquire_ms"] = acquire_ms
        t0 = self.clock()
        try:
            cmd = validate_action(r, req, self.limits, session_id=self.client.session_id, now_ns=t0)
        except ActionRejected as exc:
            self.counts[f"rejected_{exc.code}"] += 1
            row.update(command=None, rejected=exc.code, rejected_detail=exc.detail)
            cmd = None
        t1 = self.clock()
        if r.ok:
            c = row["client_ms"]
            c["validation_ms"] = (t1 - t0) / 1e6
            # on the loop clock (= validation clock: wall clock live / realtime replay, frame clock at max pace)
            c["capture_to_action_ms"] = (t_result - int(observation_time_ns)) / 1e6
            c["capture_to_command_ms"] = (t1 - int(observation_time_ns)) / 1e6
            c["capture_age_at_send_ms"] = c["capture_to_action_ms"] - c["rtt_ms"] - c["decode_ms"]
            o = r.outputs
            row.update(action_norm=np.asarray(o["action_norm"]).tolist(), padded=o["metadata"].get("padded"),
                       flange_pose=np.asarray(o["flange_pose"]).tolist(),
                       execution_flange_pose=np.asarray(o.get("execution_flange_pose", o["flange_pose"])).tolist(),
                       gripper_closure=float(o["gripper_closure"]), measured_pose7=list(st.pose7))
        if cmd is not None:
            self.sink.send(cmd)
            if self.executor is not None:
                self.executor.accept(cmd)
            row["command"] = cmd.to_json()
            self.counts["commands"] += 1
        self.counts[r.status if r.ok else (r.code or r.status)] += 1
        return row


def percentiles(values) -> dict | None:
    v = np.asarray([x for x in values if x is not None], np.float64)
    if not len(v):
        return None
    return {"n": int(len(v)), "p50": float(np.percentile(v, 50)), "p95": float(np.percentile(v, 95)),
            "p99": float(np.percentile(v, 99)), "max": float(v.max()), "mean": float(v.mean())}


def summarize(rows: list, *, route: str, mode: str, budget_ms: float = BUDGET_MS, warmup: int = 0) -> dict:
    """Latency / throughput summary of measured rows (after ``warmup`` sent steps)."""
    sent = [r for r in rows if r.get("status") in ("ok", "error", "timeout")]
    measured = sent[warmup:]
    ok = [r for r in measured if r["status"] == "ok"]
    t = [r["observation_time_ns"] for r in ok]
    starts = [r.get("t_step_ns") for r in measured if r.get("t_step_ns") is not None]
    hz = (len(starts) - 1) / ((starts[-1] - starts[0]) / 1e9) if len(starts) > 1 and starts[-1] > starts[0] else None
    steady = [r["request_bytes"] for r in ok if r.get("frames_sent") is not None][1:]
    return {
        "route": route, "mode": mode, "warmup_steps_discarded": min(warmup, len(sent)), "steps_sent": len(measured),
        "ok": len(ok),
        "commands": sum(r.get("command") is not None for r in measured),
        "rejected": dict(collections.Counter(r["rejected"] for r in measured if r.get("rejected"))),
        "errors": dict(collections.Counter(r.get("code") or r["status"] for r in measured if r["status"] != "ok")),
        "skipped": dict(collections.Counter(r["code"] for r in rows if r.get("status") == "skipped")),
        "achieved_hz": hz, "observation_span_s": (t[-1] - t[0]) / 1e9 if len(t) > 1 else None,
        "budget_ms": budget_ms,
        "missed_deadline": {"e2e_ms": sum(r["client_ms"]["e2e_ms"] > budget_ms for r in ok),
                            "capture_to_command_ms": sum(r["client_ms"]["capture_to_command_ms"] > budget_ms
                                                         for r in ok)},
        "client_ms": {k: percentiles(r["client_ms"].get(k) for r in ok) for k in CLIENT_KEYS},
        "server_ms": {k: percentiles(r["server_ms"].get(k) for r in ok) for k in SERVER_KEYS},
        "payload_bytes": {"first_request": sent[0]["request_bytes"] if sent else None,
                          "steady_request_median": float(np.median(steady)) if steady else None,
                          "response_median": float(np.median([r["response_bytes"] for r in ok])) if ok else None},
    }
