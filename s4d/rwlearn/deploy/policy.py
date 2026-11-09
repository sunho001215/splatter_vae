"""Persistent science-only deployment facade over ``OfflinePolicy`` (one interface for every FQL condition). No sockets,
threads, wire serialization or robot I/O: the numerical path is ``OfflinePolicy.act`` itself, so ``infer`` equals it
bitwise for the same observation, checkpoint and FQL noise."""
from __future__ import annotations

import copy
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from s4d.rwlearn import actions as A
from s4d.rwlearn.deploy.contracts import InputSpec, read_training_contract
from s4d.rwlearn.deploy.errors import BindingRefused, ContractViolation, SessionMismatch
from s4d.rwlearn.deploy.history import Frame, HistoryWindow, ObservationWindow
from s4d.rwlearn.inference import OfflinePolicy

TRAINING_FEATURE_DTYPE = torch.float16  # fql_data.encode_timeline stores frozen features as float16


class DeployPolicy:
    def __init__(self, checkpoint, *, device="cpu", export_path=None, export_root=None, dataset_path=None,
                 training=None):
        """Load once. ``export_path``: the frozen encoder export (sha256-checked against the checkpoint); else
        ``export_root`` / meta["export_path"] (the repo-relative ``runs/<pretrain>/encoder.pt``, e.g. a staged
        checkpoint root mirroring ``runs/``); else meta["export_path"] as recorded. ``training``: training-contract
        dict or deploy-contract sidecar (.json); else ``dataset_path`` (or the checkpoint's recorded dataset) is read
        once for the trained camera ids."""
        if export_path is None and export_root is not None:
            recorded = torch.load(checkpoint, map_location="cpu", weights_only=False)["meta"].get("export_path")
            if recorded:
                if Path(recorded).is_absolute():
                    raise ValueError(f"export_root needs a relative recorded export_path, got {recorded}")
                export_path = Path(export_root) / recorded
        self.offline = OfflinePolicy(checkpoint, device=device, export_path=export_path)
        source = training if training is not None else dataset_path
        self.input_spec = InputSpec.from_meta(self.offline.meta, read_training_contract(self.offline.meta, source))
        self.spec = None  # the most recently bound session
        self._live: dict[str, InputSpec] = {}
        self._session_ids = set()

    @classmethod
    def from_fixture(cls, offline: OfflinePolicy, spec: InputSpec) -> DeployPolicy:
        """Inject a clearly labelled synthetic policy for conditions without a trained checkpoint (including ours).

        Numerical fixture only; it makes no claim about ours' deferred training protocol or learned behaviour.
        """
        if not spec.fixture or offline.meta.get("fixture") is not True or offline.condition != spec.condition:
            raise ValueError("from_fixture requires matching, explicitly labelled FIXTURE policy and InputSpec")
        obj = cls.__new__(cls)
        obj.offline, obj.input_spec, obj.spec, obj._live, obj._session_ids = offline, spec, None, {}, set()
        return obj

    @property
    def condition(self) -> str:
        return self.offline.condition

    def reset_session(self, *, concurrent: bool = False, **choices) -> InputSpec:
        """Bind camera_id (single view), native_hw, static calibration and native correction once per session
        (``InputSpec.bind``). A new unique session_id is required on each reset so previously assembled windows cannot
        be reused; construct a fresh ``HistoryWindow`` from the returned spec. By default the new session replaces
        every live one; ``concurrent=True`` keeps the others live (several sessions share the loaded model; retire
        them with ``end_session``)."""
        if choices.get("session_id") in self._session_ids:
            raise SessionMismatch("a reset requires a new session_id")
        spec = self.input_spec.bind(**choices)
        if not concurrent:
            self._live.clear()
        self._live[spec.session_id] = spec
        self.spec = spec
        self._session_ids.add(spec.session_id)
        return spec

    def end_session(self, session_id: str) -> None:
        """Retire a live session: its windows are refused from now on (ids are never reused)."""
        self._live.pop(session_id, None)
        if self.spec is not None and self.spec.session_id == session_id:
            self.spec = None

    @property
    def live_sessions(self) -> tuple[str, ...]:
        return tuple(self._live)

    def _sync(self):
        if self.offline.device.type == "cuda":
            torch.cuda.synchronize(self.offline.device)

    def _check_binding(self, spec: InputSpec, window: ObservationWindow, obs_ids) -> None:
        if not isinstance(obs_ids, dict) or type(obs_ids.get("reference_id")) not in (str, int):
            raise BindingRefused("obs_ids must contain a measured-pose reference_id")
        for k in ("pose_time_ns", "gripper_time_ns"):
            if type(obs_ids.get(k)) is not int:
                raise BindingRefused(f"obs_ids must carry the measured state's {k} (int, host monotonic ns)")
        binding = window.binding(obs_ids["reference_id"], pose_time_ns=obs_ids["pose_time_ns"],
                                 gripper_time_ns=obs_ids["gripper_time_ns"])
        if any(obs_ids.get(k) != v for k, v in binding.items()):
            raise BindingRefused("observation/reference IDs do not match the actual history")
        t = binding["observation_time_ns"]
        # actions.step_table: the latest robot (gripper) sample at or before the observation time, at most 15 (100) ms
        if not 0 <= t - obs_ids["pose_time_ns"] <= spec.pose_max_age_ns:
            raise BindingRefused("measured pose is not the robot sample of this observation (training: <= 15 ms "
                                 "before)")
        if not 0 <= t - obs_ids["gripper_time_ns"] <= spec.gripper_max_age_ns:
            raise BindingRefused("gripper register is not the sample of this observation (training: <= 100 ms "
                                 "before)")

    @torch.inference_mode()
    def infer(self, window: ObservationWindow, measured_pose7, gripper_register, obs_ids: dict,
              noise=None) -> dict:
        """``self.act`` (``OfflinePolicy``'s path with training-precision float16 features: normalised action,
        physical base-frame flange delta, goal pose bound to ``measured_pose7``, continuous gripper closure +
        register) plus the pose/observation binding, depth provenance and per-stage timings (CUDA-synchronised on
        GPU). ``obs_ids = window.binding(...)``; the measured
        pose and register are the robot samples bound to ``window.observation_time_ns``. A response's goal must
        never be re-applied relative to a newer measurement. ``noise``: (1, 7) FQL noise (float32) or None (the
        agent's own generator, as ``OfflinePolicy``)."""
        start = perf_counter()
        spec = self._live.get(getattr(getattr(window, "spec", None), "session_id", None))
        if spec is None or window.spec != spec:
            raise SessionMismatch("window belongs to a missing or different session; reset and rebuild history")
        self._check_binding(spec, window, obs_ids)
        pose = np.array(measured_pose7, np.float64, copy=True)
        if pose.shape != (7,) or not np.isfinite(pose).all() or np.linalg.norm(pose[3:]) < 1e-12:
            raise ContractViolation("measured_pose7 must be finite xyz + nonzero quaternion xyzw")
        grip = float(gripper_register)
        if not np.isfinite(grip):
            raise ContractViolation("finite measured gripper register required")
        observation = window.observation()  # also validates complete, regularly spaced, synchronized history
        if noise is not None:
            noise = torch.as_tensor(noise, dtype=torch.float32, device=self.offline.device)
            if noise.shape != (1, A.ACTION_DIM) or not torch.isfinite(noise).all():
                raise ContractViolation("FQL noise must be finite (1, 7)")
        validated = perf_counter()
        elapsed, active = {}, {}

        def pre(name):
            def hook(_module, _args):
                self._sync()
                active[name] = perf_counter()
            return hook

        def post(name):
            def hook(_module, _args, _out):
                self._sync()
                elapsed[name] = (perf_counter() - active[name]) * 1000
            return hook

        pixel = self.offline.condition == "e2e_cnn"
        actor = self.offline.agent.actor_onestep_flow
        encoder = actor.encoder if pixel else self.offline.adapter.backbone
        handles = []
        for name, module in (("encoder_ms", encoder), ("actor_ms", actor)):
            handles.extend([module.register_forward_pre_hook(pre(name)), module.register_forward_hook(post(name))])
        try:
            out = self.act(observation, pose, grip, noise=noise)
            self._sync()
        finally:
            for handle in handles:
                handle.remove()
        finish = perf_counter()
        # FQL includes the learned projection/proprio head for frozen encoders; the CNN encoder is reported separately.
        timings = {"validation_ms": (validated - start) * 1000, "encoder_ms": elapsed["encoder_ms"],
                   "fql_ms": max(0.0, elapsed["actor_ms"] - (elapsed["encoder_ms"] if pixel else 0.0)),
                   "policy_ms": (finish - validated) * 1000, "total_ms": (finish - start) * 1000}
        out.update(measured_pose7=pose, measured_gripper_register=grip, obs_ids=copy.deepcopy(obs_ids),
                   frame_timestamps_ns=window.timestamps_ns.copy(), timings=timings,
                   metadata={"condition": spec.condition, "fixture": spec.fixture,
                             "session_id": spec.session_id, "padded": window.padded,
                             "feature_precision": spec.feature_precision,
                             "training_depth": spec.training_depth,
                             "deployment_depth": spec.deployment_depth,
                             "native_depth_correction": copy.deepcopy(spec.native_depth_correction),
                             "action_reference": "bound measured_pose7; base-frame flange delta; continuous gripper"})
        return out

    @torch.inference_mode()
    def act(self, observation: dict, measured_pose7, gripper_register, *, noise=None) -> dict:
        """The training-consistent numerical path (no validation, timings or binding): ``OfflinePolicy``'s own
        methods, with frozen-encoder features rounded to float16 exactly as the FQL dataset stored them
        (``fql_data.encode_timeline`` -> ``.to(torch.float16)``; the agent then reads ``.float()``).
        ``OfflinePolicy.act`` feeds unrounded float32 features; the e2e CNN has no frozen features (identical)."""
        proprio = A.measured_proprio(measured_pose7, gripper_register)
        if self.offline.condition == "e2e_cnn":
            return self.offline.act_from_pixels(observation["rgb"], proprio, measured_pose7, noise=noise)
        feat = self.offline.features(observation).to(TRAINING_FEATURE_DTYPE).float()
        return self.offline.act_from_features(feat, proprio, measured_pose7, noise=noise)

    @torch.inference_mode()
    def warm_up(self, window: ObservationWindow | None = None, *, iterations: int = 2) -> dict:
        """Warm the persistent modules after ``reset_session``; an optional real window, otherwise a labelled synthetic
        input for the most recent session. Fixed zero noise avoids advancing the policy noise stream. Returns the last
        timings."""
        spec = window.spec if window is not None else self.spec
        if spec is None or iterations < 1:
            raise SessionMismatch("reset_session first; iterations must be positive")
        if window is None:
            history = HistoryWindow(spec)
            for c in spec.camera_ids:
                for seq in range(spec.history_length):
                    rgb = np.zeros((*spec.image_hw, 3), np.uint8)
                    depth = np.ones(spec.image_hw, np.float32) if "depth" in spec.modalities else None
                    valid = None if depth is None else depth > 0
                    metadata = {} if depth is None else {"depth_source": spec.deployment_depth,
                                                          "native_correction": spec.native_depth_correction[c]}
                    history.push(c, Frame(seq, int(round(seq * spec.frame_period_s * 1e9)), rgb, depth, valid,
                                          metadata))
            window = history.assemble(dict.fromkeys(spec.camera_ids, spec.history_length - 1),
                                      surface_disp_m=[0.0] * spec.history_length if spec.views > 1 else None)
        t = window.observation_time_ns
        result = None
        for _ in range(iterations):
            result = self.infer(window, [0, 0, 0, 0, 0, 0, 1], 3,
                                window.binding("warm_up", pose_time_ns=t, gripper_time_ns=t),
                                noise=torch.zeros(1, A.ACTION_DIM, device=self.offline.device))
        return result["timings"]

    warmup = warm_up
