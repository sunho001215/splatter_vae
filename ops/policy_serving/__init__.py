"""Network serving of real-world FQL policies: robot-side CPU client <-> one persistent GPU inference server.

Infrastructure only. The scientific inference (input contracts, preprocessing, history rules, encoder + FQL, action
conversion) lives in ``s4d.rwlearn.deploy``; this package moves its inputs and outputs over ONE persistent TCP
connection and never reinterprets them.

- ``protocol``: length-prefixed binary frames (u32 header length + JSON header + raw little-endian arrays), message
  types, error codes, socket I/O. numpy + stdlib only.
- ``client``: CPU-only robot-side client (numpy + stdlib; imports without torch), action-expiry helper, optional
  RealSense capture through ``s4d.rwlearn.deploy.realsense``.
- ``server``: persistent policy server (``python -I ops/policy_serving/server.py serve``); the only module that
  loads models.
- ``bench``: sustained 15 Hz paced / max-rate latency benchmark (``python -I ops/policy_serving/bench.py``).
- ``config``: ``configs/realworld/policy_serving.yaml`` reader.

Nothing here commands a robot: actions are returned to the caller, which runs dry (replay demo) by default.
"""
