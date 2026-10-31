"""TEMPLATE adapter for the planned Franka Panda + Robotiq + three RealSense rigs recordings.

No sample data exists yet, so this adapter deliberately implements NO file layout. It documents, method by
method, what must be read and which decisions must be confirmed against the first real episode. Copy this file
(or fill in the methods) once the recording format is fixed; configure it with
configs/realworld/template_adapter.yaml. Every method raises ``NotImplementedError`` with the open question.

Checklist for the real adapter (each item becomes either code or an ``assumptions`` entry):

Cameras (per RealSense rig; serial number = rig_id):
  * which streams were recorded: left/right INFRARED (Y8/Y16; rectified when recorded at the depth resolution
    in Y8 - confirm for the chosen mode) and/or the RGB colour sensor (a different camera with its own pose);
  * whether the IR projector (emitter) was on, off or alternating (affects IR images and stereo matching);
  * intrinsics per stream exactly as the SDK reported them (``rs2_intrinsics``: width, height, ppx, ppy, fx, fy,
    model, coeffs) and their pixel convention (librealsense uses integer pixel centres -> ``integer_centers``);
  * stereo extrinsic left->right (``rs2_extrinsics`` between the two IR sensors; baseline ~50 mm for D435/D455
    at 95 mm) -> ``StereoGeometry(status="metric")``; if absent -> ``status="missing"``;
  * the depth-to-colour extrinsic if RGB is used for appearance with IR-derived depth;
  * timestamps: hardware/global time domain (``RS2_TIMESTAMP_DOMAIN_GLOBAL_TIME`` vs. hardware clock vs.
    system time) and how they relate to the robot clock; frame drops.
Robot:
  * joint positions (7, rad) and their clock; gripper state semantics (Robotiq register 0..255, normalised,
    width in m?) and the exact Robotiq model (2F-85 / 2F-140); EE pose and WHICH frame it refers to
    (panda_link8 flange, panda_hand_tcp, a Robotiq TCP?) - used only for cross-checks;
  * URDF assets: Panda description + the exact Robotiq model + measured flange-to-gripper mount transform.
"""

from __future__ import annotations

from s4d.realworld.adapters.base import EpisodeAdapter, register_adapter


@register_adapter
class TemplateRealSenseAdapter(EpisodeAdapter):
    name = "template_realsense"
    version = "0-template"

    def list_episodes(self) -> list[str]:
        raise NotImplementedError("awaiting real data: how are episodes delimited on disk (one directory, one "
                                  "bag/mcap file, one HDF5 group)? Map each to a path-safe episode id.")

    def read_meta(self, episode_id: str):
        raise NotImplementedError(
            "awaiting real data: build EpisodeMeta from the recorded camera metadata (serials, stream types, "
            "rs2 intrinsics + pixel convention, IR left->right extrinsic), robot state arrays (units, clock) and "
            "the robot description (URDF + exact Robotiq model + mount transform). See the module docstring.")

    def iter_frames(self, episode_id: str, stream_id: str, start: int = 0, stop: int | None = None):
        raise NotImplementedError("awaiting real data: decode frames exactly as recorded (Y8/Y16 infrared as 1 "
                                  "channel, colour as RGB/BGR), yielding (index, timestamp_ns, array).")
