# Early findings (2026-11-13)

Worktree: /home/ws/ws/splatter4d/.worktrees/dsreset, branch dsreset-20261113, base 4a78216799ff671bd664c2c2469a95f5708ea8e2. Main checkout is dirty and is not edited.

Established chain: verified raw recordings -> per-task LoMa-G rig calibration and validated per-episode calibrate output -> FoundationStereo B1 depth + per-camera TrackEverything -> compact train_data (128px, timestamps/segments/calibration retained) -> QA/full-data gate -> task store/split/outcomes -> method pretraining -> frozen export -> fqldata -> FQL. 3D-MVP instead overlays verified RealSense native depth, with measured native-depth bounds. Ours uses Original architecture and workspace stats; SinCro uses calibrated rays; SNeRL uses measured lookat/pose_scale; 3D-MVP uses measured bounds. Image methods still need episode QA but may precede geometry readiness.

Read PNP geometry registration plan: previous 35-row geometry chain required validated calibration, 97-episode gate amendment, native-depth stage, scene stats, and resolver coverage. No assumption that its old gate or calibration applies to the new drawer session. Drawer-open-v2 requires its OWN per-task rig calibration. Failure episodes reuse the accepted calibration of their task/session after serial/fixedness/coverage checks; they do not inherit success outcome labels.

FQL currently already supports explicit {episode: bool} outcomes. s4d/rwlearn/transitions.py uses ogbench_final_success: reward = success(next)-1, mask=1-success(next). Failed episodes have no success observation and bootstrap at their final valid transition. The existing terminal array means contiguous-chain boundary, not absorbing MDP termination; preserve it for S bitwise compatibility and expose separate terminated/truncated metadata if needed. BC/flow uses all current valid transitions unchanged. Frozen FQL requires alpha=10 + normalize_q_loss=true; E2E keeps its current settings (augmentation/noaug variants).

At first inspection drawer-open-v2 success directory exists but no complete file was listed; drawer failure and covering-failure directories do not yet exist. Never inspect partial files. H5 schema comparison pending complete arrivals.

Retained requested methods: ours, sincro, reviwo, dynamo, snerl, mvp3d_single, mvp3d_multiview, e2e_cnn, e2e_cnn_noaug. Retirement notes exclude S2M2 and obsolete downstream RL algorithms, not these methods. 3D-MVP robot actuation remains outside this task.
