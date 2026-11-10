"""Robot-PC side of real-world policy deployment (Directive I, 2026-11-10; docs/robot_pc.md). CPU only, no torch.

- ``safety``: execution-mode guard (manual mode needs CLI flag + typed confirmation + config armed; binding UNBOUND);
- ``sinks``: controller-integration interface: ``MockSink`` (replay), ``DryRunSink`` (logs, never sends),
  ``LiveSink`` (refuses without authorization, then raises Unbound);
- ``validation`` / ``poses``: robot-side action validation (finite, contract range, per-step lead, workspace, gripper,
  binding to the observation's measured pose, expiry);
- ``executor``: the local high-rate loop (rate-limited tracking of the bound absolute goal, hold on expiry);
- ``robot_state``: read-only robot state (recorded replay, static latency-only, live UNBOUND skeleton);
- ``cameras`` / ``devices``: RealSense acquisition with the training preprocessing, device-busy guard;
- ``selection``: policy / task / camera selection and session choices from the server's contracts;
- ``loop``: one policy step + latency summary; ``replay`` / ``dryrun``: the two drivers; ``config``: config reader.

Science (preprocessing, history, contracts, M5 rule) comes from ``s4d.rwlearn.deploy``; the link from
``ops.policy_serving.client``. Nothing in this package sends anything to a robot, arm, gripper or controller.
"""
