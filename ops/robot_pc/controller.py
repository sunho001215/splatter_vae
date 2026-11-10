"""The commander side of switch_controller (execution mode only), exactly as the robot PC's teleop collector does it
(teleop_collector/robot.py:87-160, app.py:305-363 / :785-797; docs/robot_pc.md "LIVE integration design"):

* bind the ControllerCommand PUB (``:28401``) and the RobotAction PUB (``:28402``) on 127.0.0.1 - switch_controller's
  subscribers connect to them (its panda_real.yaml ``zmq_setup``). A collector running at the same time already holds
  these ports, so the bind fails and execution is refused (two commanders can never coexist);
* ``wait_subscribers``: both controller subscribers connected (ZMQ monitor ACCEPTED events) before anything is sent;
* ``engage``: repeat mode 3 requests every ~0.1 s during the collector's ``CONTROLLER_SETTLE_S`` 0.5 s. Requests for
  the active mode are idempotent; RobotState has no mode field (the operator confirms acceptance on the UI);
* ``send_flange_pose``: one RobotAction (flange goal -> controller ee, ``target_dt`` 0.05 s: without a fresher action the
  controller holds the measured pose);
* ``release``: a final action at the fresh MEASURED ee pose as received (byte-identical to the collector's hold),
  wait until the robot is still (|dq| small, bounded wait), request IDLE (mode 1) if this link requested a mode.

Constructed only with an ``ExecutionAuthorization``. pyzmq / robot_msg imported here only.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from ops.robot_pc import switch_controller as sc
from ops.robot_pc.safety import ExecutionAuthorization, ExecutionRefused

TASK_IMPEDANCE, IDLE = 3, 1
CONTROLLER_SETTLE_S = 0.5  # teleop_collector app.py:53
log = logging.getLogger(__name__)


class ControllerLink:
    def __init__(self, authorization: ExecutionAuthorization, *, command_bind: str = sc.COMMAND_BIND,
                 action_bind: str = sc.ACTION_BIND, zmq_module=None, robot_msg=None, target_dt: float = sc.TARGET_DT_S):
        if not isinstance(authorization, ExecutionAuthorization):
            raise ExecutionRefused("ControllerLink requires an ExecutionAuthorization (execution mode)")
        if zmq_module is None:
            import zmq as zmq_module
        if robot_msg is None:
            import robot_msg
        self.zmq, self.msg, self.target_dt = zmq_module, robot_msg, float(target_dt)
        self.ctx = zmq_module.Context()
        self.peers = {"command": 0, "action": 0}
        self._monitors, self.socks = {}, {}
        self.mode_requested: int | None = None
        self.actions_sent = 0
        self.mode_requests_sent = 0
        self._disconnected = False
        try:
            for name, ep in (("command", command_bind), ("action", action_bind)):
                s = self.ctx.socket(zmq_module.PUB)
                self.socks[name] = s  # include a socket whose bind/options fail in construction cleanup
                s.setsockopt(zmq_module.LINGER, 0)
                s.setsockopt(zmq_module.SNDHWM, 4)
                self._monitors[name] = s.get_monitor_socket(zmq_module.EVENT_ACCEPTED | zmq_module.EVENT_DISCONNECTED)
                s.bind(ep)  # fails if a teleop collector (or another launcher) is the commander
        except Exception as exc:
            self.close()
            raise ExecutionRefused(f"cannot become the commander ({exc}); is the teleop collector running?") from exc

    def update_peers(self) -> dict:
        from zmq.utils.monitor import recv_monitor_message

        for name, mon in self._monitors.items():
            while mon.poll(0):
                event = recv_monitor_message(mon)["event"]
                if event == self.zmq.EVENT_ACCEPTED:
                    self.peers[name] += 1
                elif event == self.zmq.EVENT_DISCONNECTED:
                    self._disconnected = True
                    self.peers[name] = max(0, self.peers[name] - 1)
        return dict(self.peers)

    def connected(self) -> bool:
        p = self.update_peers()
        return p["command"] > 0 and p["action"] > 0

    def peers_ok(self) -> bool:
        """A disconnect is latched even if a peer reconnects before the next tick."""
        return self.connected() and not self._disconnected

    def wait_subscribers(self, timeout_s: float = 3.0, *, cancel=None) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if cancel is not None and cancel.is_set():
                return False
            if self.connected():
                return True
            time.sleep(0.02)
        return False

    def request_mode(self, mode: int) -> None:
        # A send can fail after handing bytes to the transport; release must still attempt IDLE.
        self.mode_requested = int(mode)
        self.socks["command"].send(self.msg.encode(self.msg.ControllerCommand(ctrl_mode=int(mode))),
                                   self.zmq.NOBLOCK)
        self.mode_requests_sent += 1

    def engage(self, settle_s: float = CONTROLLER_SETTLE_S, *, cancel=None) -> None:
        if not self.connected():
            raise ExecutionRefused("switch_controller subscribers are not connected to the command/action channels")
        deadline = time.monotonic() + settle_s
        while cancel is None or not cancel.is_set():
            self.request_mode(TASK_IMPEDANCE)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            if cancel is None:
                time.sleep(min(0.1, remaining))
            else:
                cancel.wait(min(0.1, remaining))

    def send_flange_pose(self, flange_pose7) -> None:
        self.socks["action"].send(self.msg.encode(sc.robot_action(self.msg, flange_pose7, self.target_dt)),
                                  self.zmq.NOBLOCK)
        self.actions_sent += 1

    def send_measured_ee_hold(self, measured_ee_pose7) -> None:
        """The collector's disengage hold (app.py:360-361): the MEASURED controller-ee pose exactly as received
        (no flange round trip), so the message is byte-identical to the collector's."""
        self.socks["action"].send(self.msg.encode(sc.robot_action_ee(self.msg, measured_ee_pose7, self.target_dt)),
                                  self.zmq.NOBLOCK)
        self.actions_sent += 1

    def release(self, latest_state, *, still=None, wait_still_s: float = 1.0) -> None:
        """Independent hold / bounded stillness / IDLE phases; ``latest_state()`` has host ``t_ns``.

        A cached pose alone is never sufficient authority to send a final hold.
        """
        try:
            if self.mode_requested is None:
                return  # constructed but never engaged: close without publishing an action
            try:
                s = latest_state()
                if s is None or not 0 <= time.monotonic_ns() - s["t_ns"] < 50_000_000:
                    log.warning("release: skipping measured-ee hold: no fresh state (<50 ms)")
                elif np.asarray(s["ee_pose7"]).shape != (7,) or not np.isfinite(s["ee_pose7"]).all():
                    log.warning("release: skipping measured-ee hold: invalid pose")
                else:
                    self.send_measured_ee_hold(s["ee_pose7"])
            except (Exception, KeyboardInterrupt):
                log.exception("release: measured-ee hold failed; continuing to IDLE")
            try:
                # The collector's entire stillness wait is bounded to 1 s, including target_dt.
                start = time.monotonic()
                deadline = start + min(1.0, max(self.target_dt, float(wait_still_s)))
                time.sleep(min(self.target_dt, 1.0))
                while still is not None and time.monotonic() < deadline and not still():
                    time.sleep(min(0.02, max(0.0, deadline - time.monotonic())))
            except (Exception, KeyboardInterrupt):
                log.exception("release: stillness wait failed; continuing to IDLE")
            try:
                if self.mode_requested is not None:
                    self.request_mode(IDLE)
                    time.sleep(0.05)
            except (Exception, KeyboardInterrupt):
                log.exception("release: IDLE request failed")
        finally:
            self.close()

    def close(self) -> None:
        for s in [*self.socks.values(), *self._monitors.values()]:
            try:
                s.close(0)
            except Exception:  # noqa: BLE001 - shutdown
                pass
        self.socks, self._monitors = {}, {}
