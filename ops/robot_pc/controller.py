"""The commander side of switch_controller (execution mode only), exactly as the robot PC's teleop collector does it
(teleop_collector/robot.py:87-160, app.py:305-363 / :785-797; docs/robot_pc.md "LIVE integration design"):

* bind the ControllerCommand PUB (``:28401``) and the RobotAction PUB (``:28402``) on 127.0.0.1 - switch_controller's
  subscribers connect to them (its panda_real.yaml ``zmq_setup``). A collector running at the same time already holds
  these ports, so the bind fails and execution is refused (two commanders can never coexist);
* ``wait_subscribers``: both controller subscribers connected (ZMQ monitor ACCEPTED events) before anything is sent;
* ``engage``: request mode 3 (task impedance; switch_controller ignores a request for the active mode), wait the
  collector's ``CONTROLLER_SETTLE_S`` 0.5 s;
* ``send_flange_pose``: one RobotAction (flange goal -> controller ee, ``target_dt`` 0.05 s: without a fresher action the
  controller holds the measured pose);
* ``release``: a final action at the MEASURED pose (immediate hold), wait until the robot is still (|dq| small, bounded
  wait), request IDLE (mode 1) - only if this link requested the mode.

Constructed only with an ``ExecutionAuthorization``. pyzmq / robot_msg imported here only.
"""

from __future__ import annotations

import time

from ops.robot_pc import switch_controller as sc
from ops.robot_pc.safety import ExecutionAuthorization, ExecutionRefused

TASK_IMPEDANCE, IDLE = 3, 1
CONTROLLER_SETTLE_S = 0.5  # teleop_collector app.py:53


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
        try:
            for name, ep in (("command", command_bind), ("action", action_bind)):
                s = self.ctx.socket(zmq_module.PUB)
                s.setsockopt(zmq_module.LINGER, 0)
                s.setsockopt(zmq_module.SNDHWM, 4)
                self._monitors[name] = s.get_monitor_socket(zmq_module.EVENT_ACCEPTED | zmq_module.EVENT_DISCONNECTED)
                s.bind(ep)  # fails if a teleop collector (or another launcher) is the commander
                self.socks[name] = s
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
                    self.peers[name] = max(0, self.peers[name] - 1)
        return dict(self.peers)

    def connected(self) -> bool:
        p = self.update_peers()
        return p["command"] > 0 and p["action"] > 0

    def wait_subscribers(self, timeout_s: float = 3.0) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.connected():
                return True
            time.sleep(0.02)
        return False

    def request_mode(self, mode: int) -> None:
        self.socks["command"].send(self.msg.encode(self.msg.ControllerCommand(ctrl_mode=int(mode))),
                                   self.zmq.NOBLOCK)
        self.mode_requested = int(mode)

    def engage(self, settle_s: float = CONTROLLER_SETTLE_S) -> None:
        if not self.connected():
            raise ExecutionRefused("switch_controller subscribers are not connected to the command/action channels")
        self.request_mode(TASK_IMPEDANCE)
        time.sleep(settle_s)

    def send_flange_pose(self, flange_pose7) -> None:
        self.socks["action"].send(self.msg.encode(sc.robot_action(self.msg, flange_pose7, self.target_dt)),
                                  self.zmq.NOBLOCK)
        self.actions_sent += 1

    def release(self, measured_flange_pose7=None, *, still=None, wait_still_s: float = 2.0) -> None:
        """Hold at the measured pose, wait (bounded) until ``still()`` is true, then IDLE if we requested a mode."""
        try:
            if measured_flange_pose7 is not None:
                self.send_flange_pose(measured_flange_pose7)
            deadline = time.monotonic() + wait_still_s
            while still is not None and time.monotonic() < deadline and not still():
                time.sleep(0.02)
            if self.mode_requested is not None and self.mode_requested != IDLE:
                self.request_mode(IDLE)
                time.sleep(0.05)
        finally:
            self.close()

    def close(self) -> None:
        for s in [*self.socks.values(), *self._monitors.values()]:
            try:
                s.close(0)
            except Exception:  # noqa: BLE001 - shutdown
                pass
        self.socks, self._monitors = {}, {}
