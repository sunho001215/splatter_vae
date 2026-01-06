from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass(frozen=True)
class CameraDef:
    """
    Minimal MJCF camera definition.
    - name: camera name used everywhere in robosuite (camera_names, render_camera, etc.)
    - pos: world position [x,y,z]
    - quat: MuJoCo quaternion [w,x,y,z]
    - fovy: vertical field-of-view in degrees
    """
    name: str
    pos: List[float]
    quat: List[float]
    fovy: float


def load_camera_json(path: str) -> List[CameraDef]:
    with open(path, "r") as f:
        cfg = json.load(f)
    cams = []
    for c in cfg["cameras"]:
        cams.append(CameraDef(
            name=str(c["name"]),
            pos=list(map(float, c["pos"])),
            quat=list(map(float, c["quat"])),
            fovy=float(c.get("fovy", 45.0)),
        ))
    if len(cams) != 6:
        # You requested exactly 6 cameras; enforce that here.
        raise ValueError(f"Expected exactly 6 cameras, got {len(cams)}")
    return cams


def _fmt_vec(v: List[float]) -> str:
    return " ".join(f"{x:.6f}" for x in v)


def inject_cameras_into_mjcf_xml(xml_string: str, cameras: List[CameraDef]) -> str:
    """
    Adds <camera .../> elements under <worldbody>.
    If a camera with the same name already exists, we remove it and replace it.

    This keeps the rest of the environment unchanged.
    """
    root = ET.fromstring(xml_string)

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("MJCF has no <worldbody> element; cannot inject cameras.")

    # Remove existing cameras with same names (avoid duplicates)
    existing = list(worldbody.findall("camera"))
    names = {c.name for c in cameras}
    for cam_elem in existing:
        if cam_elem.get("name") in names:
            worldbody.remove(cam_elem)

    # Add new cameras
    for c in cameras:
        cam_elem = ET.Element("camera")
        cam_elem.set("name", c.name)
        cam_elem.set("pos", _fmt_vec(c.pos))
        cam_elem.set("quat", _fmt_vec(c.quat))
        cam_elem.set("fovy", f"{c.fovy:.6f}")
        # Optional: cam_elem.set("mode", "fixed")  # usually unnecessary
        worldbody.append(cam_elem)

    # Return updated XML string
    return ET.tostring(root, encoding="unicode")


def apply_custom_cameras_to_env(env, cameras: List[CameraDef]) -> None:
    """
    Grabs the current MJCF XML from the env, injects cameras, and reloads the env.

    This uses robosuite's supported API:
      MujocoEnv.reset_from_xml_string(xml_string)
    """
    # robosuite models typically expose get_xml()
    if hasattr(env.model, "get_xml"):
        xml_string = env.model.get_xml()
    else:
        # Fallback if API changes
        xml_string = str(env.model)

    new_xml = inject_cameras_into_mjcf_xml(xml_string, cameras)

    # Reload sim/model from updated XML
    env.reset_from_xml_string(new_xml)

    # Ensure env.camera_names is consistent (robosuite uses this for obs collection)
    try:
        env.camera_names = [c.name for c in cameras]
    except Exception:
        # Not fatal; most robosuite builds use env.camera_names
        pass
