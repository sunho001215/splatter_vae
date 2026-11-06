"""Real-world learning on top of the preprocessed Panda + Robotiq + RealSense episodes (design: docs/realworld_baselines.md,
docs/realworld_fql.md).

* ``store``: the abstract per-camera ``FrameStore`` every adapter reads (frames stored once per camera and
  timestamp, centre crop 128 px; data-format policy 2026-11-06) and the loader-side indexes (cross-camera sync, runs,
  histories); ``ProcessedEpisodeStore`` is a reference implementation over the preprocessing outputs (tests).
* ``baseline_data``: real-world data adapters for the representation BASELINES (SNeRL, 3D-MVP, SinCro, ReViWo,
  DynaMo) producing exactly each baseline's Meta-World samples; ``configs`` derives their real-world configs.
* ``gate``: the user's full-data gate (all 100 episodes of a task validated + preprocessed before any pretraining),
  plus the calibration hold of geometry-dependent consumers (``CONSUMERS``; ``calibration_validated``).
* ``outcomes``: episode outcomes (all successful demonstrations, user decision) and the ending reset / reversal check.
* ``actions`` / ``transitions``: the robot action contract and the offline-RL transition definitions, from the raw
  ``panda_teleop_episode`` v2 recordings (read-only).
* ``fql``: a faithful PyTorch port of the official Flow Q-Learning agent (Park et al. 2025, seohongpark/fql @ e8cd16e);
  ``pixel``: the official ``impala_small`` pixel path for the end-to-end CNN + FQL condition (``e2e_cnn``).
* ``fql_data`` / ``inference``: frozen-encoder feature datasets, camera-randomised offline sampling and offline (no
  robot I/O) inference with conversion to the robot command convention; ``snerl_eval``: real-world SNeRL validation
  (input-view vs leave-one-camera-out novel-view); ``launch``: the (unregistered) training plan.

Unlike ``s4d.realworld`` (preprocessing; never imports model code), this package may import ``s4d.baselines`` and
``s4d.rl`` because it trains and evaluates the representation baselines and policies.
"""
