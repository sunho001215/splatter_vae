from __future__ import annotations

import argparse
import json
import os
import re
import sys
import warnings
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SEG_TYPE_ALIASES = {
    "background": -1,
    "body": 1,
    "joint": 3,
    "geom": 5,
    "site": 6,
    "camera": 7,
    "light": 8,
}
SEG_TYPE_NAMES = {value: key for key, value in SEG_TYPE_ALIASES.items()}


def _decode(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _demo_sort_key(key: str) -> int:
    try:
        return int(key.replace("demo", ""))
    except Exception:
        return 10**12


def _parse_views(raw: str | None) -> list[str] | None:
    if raw is None or not raw.strip():
        return None
    return [part.strip() for part in raw.split(",") if part.strip()]


def _type_name(obj_type: int | str | None) -> str:
    if obj_type is None or obj_type == "":
        return ""
    try:
        return SEG_TYPE_NAMES.get(int(obj_type), str(int(obj_type)))
    except Exception:
        return str(obj_type)


def _selector(obj_id: int, obj_type: int | str | None) -> str:
    name = _type_name(obj_type)
    if not name:
        return str(int(obj_id))
    return f"{name}:{int(obj_id)}"


def _yaml_list(items: Iterable[str]) -> str:
    values = [str(item) for item in items]
    if not values:
        return "[]"
    return "[" + ", ".join(json.dumps(v) for v in values) + "]"


def _parse_selectors(raw: str | None) -> list[str] | None:
    if raw is None or not raw.strip():
        return None
    out: list[str] = []
    for part in raw.split(","):
        text = part.strip()
        if not text:
            continue
        if ":" not in text:
            out.append(str(int(text)))
            continue
        typ, obj_id = [piece.strip() for piece in text.split(":", 1)]
        if not typ or not obj_id:
            raise ValueError(f"Invalid selector {text!r}; expected geom:43, site:5, or a plain integer ID.")
        if typ.lower() in SEG_TYPE_ALIASES:
            typ = typ.lower()
        else:
            typ = _type_name(int(typ))
        out.append(f"{typ}:{int(obj_id)}")
    return out


def _dedupe(items: Iterable[str]) -> list[str]:
    seen = set()
    out = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _norm_tokens(*parts: Any) -> set[str]:
    text = " ".join(str(part or "") for part in parts).lower()
    return {token for token in re.split(r"[^a-z0-9]+", text) if token}


def _has_any_token(tokens: set[str], *needles: str) -> bool:
    return any(needle in tokens for needle in needles)


def _row(
    *,
    obj_id: int,
    obj_type: int,
    name: str = "",
    type_name: str = "",
    body_id: int | None = None,
    body_name: str = "",
    body_path: str = "",
) -> dict[str, Any]:
    return {
        "id": int(obj_id),
        "type": int(obj_type),
        "type_name": type_name or _type_name(obj_type),
        "name": str(name),
        "body_id": None if body_id is None else int(body_id),
        "body_name": str(body_name),
        "body_path": str(body_path),
    }


def _load_name_table(demo: h5py.Group) -> dict[tuple[int, int], dict[str, Any]]:
    table: dict[tuple[int, int], dict[str, Any]] = {}
    if "segmentation" not in demo:
        return table
    seg = demo["segmentation"]
    if not {"ids", "names"}.issubset(seg.keys()):
        return table
    ids = np.asarray(seg["ids"], dtype=np.int32)
    names = [_decode(v) for v in seg["names"][:]]
    types = np.asarray(seg["types"], dtype=np.int32) if "types" in seg else np.full_like(ids, -999)
    type_names = [_decode(v) for v in seg["type_names"][:]] if "type_names" in seg else [""] * len(ids)
    body_ids = np.asarray(seg["body_ids"], dtype=np.int32) if "body_ids" in seg else np.full_like(ids, -1)
    body_names = [_decode(v) for v in seg["body_names"][:]] if "body_names" in seg else [""] * len(ids)
    body_paths = [_decode(v) for v in seg["body_paths"][:]] if "body_paths" in seg else [""] * len(ids)
    for obj_id, obj_type, name, type_name, body_id, body_name, body_path in zip(ids, types, names, type_names, body_ids, body_names, body_paths):
        table[(int(obj_id), int(obj_type))] = _row(
            obj_id=int(obj_id),
            obj_type=int(obj_type),
            name=name,
            type_name=type_name,
            body_id=None if int(body_id) < 0 else int(body_id),
            body_name=body_name,
            body_path=body_path,
        )
    return table


@lru_cache(maxsize=None)
def _load_model_table(env_id: str, env_name: str) -> dict[tuple[int, int], dict[str, Any]]:
    os.environ.setdefault("MUJOCO_GL", "egl")
    warnings.filterwarnings("ignore", message=".*Box observation space maximum and minimum values are equal.*")
    try:
        import gymnasium as gym
        import metaworld  # noqa: F401
        import mujoco
    except Exception:
        return {}

    gym_env_name = env_name if env_name.endswith("-v3") else f"{env_name}-v3"
    env = None
    try:
        env = gym.make(env_id or "Meta-World/MT1", env_name=gym_env_name, seed=0)
        model = env.unwrapped.model if hasattr(env.unwrapped, "model") else env.unwrapped.sim.model
        table: dict[tuple[int, int], dict[str, Any]] = {}
        background_type = SEG_TYPE_ALIASES["background"]
        table[(-1, background_type)] = _row(obj_id=-1, obj_type=background_type, name="background", type_name="background")

        geom_type = int(mujoco.mjtObj.mjOBJ_GEOM)
        body_type = int(mujoco.mjtObj.mjOBJ_BODY)
        site_type = int(mujoco.mjtObj.mjOBJ_SITE)
        camera_type = int(mujoco.mjtObj.mjOBJ_CAMERA)
        light_type = int(mujoco.mjtObj.mjOBJ_LIGHT)

        def body_path(body_id: int) -> str:
            names = []
            current = int(body_id)
            seen = set()
            while current >= 0 and current not in seen:
                seen.add(current)
                names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, current) or f"body_{current}")
                parent = int(model.body_parentid[current]) if current > 0 else -1
                current = parent
            return "/".join(reversed(names))

        for geom_id in range(int(model.ngeom)):
            body_id = int(model.geom_bodyid[geom_id])
            table[(geom_id, geom_type)] = _row(
                obj_id=geom_id,
                obj_type=geom_type,
                name=mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or f"geom_{geom_id}",
                type_name="geom",
                body_id=body_id,
                body_name=mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"body_{body_id}",
                body_path=body_path(body_id),
            )
        for body_id in range(int(model.nbody)):
            table[(body_id, body_type)] = _row(
                obj_id=body_id,
                obj_type=body_type,
                name=mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"body_{body_id}",
                type_name="body",
                body_id=body_id,
                body_name=mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"body_{body_id}",
                body_path=body_path(body_id),
            )
        for site_id in range(int(model.nsite)):
            body_id = int(model.site_bodyid[site_id])
            table[(site_id, site_type)] = _row(
                obj_id=site_id,
                obj_type=site_type,
                name=mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, site_id) or f"site_{site_id}",
                type_name="site",
                body_id=body_id,
                body_name=mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"body_{body_id}",
                body_path=body_path(body_id),
            )
        for cam_id in range(int(model.ncam)):
            table[(cam_id, camera_type)] = _row(
                obj_id=cam_id,
                obj_type=camera_type,
                name=mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_id) or f"camera_{cam_id}",
                type_name="camera",
            )
        for light_id in range(int(model.nlight)):
            table[(light_id, light_type)] = _row(
                obj_id=light_id,
                obj_type=light_type,
                name=mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_LIGHT, light_id) or f"light_{light_id}",
                type_name="light",
            )
        return table
    except Exception:
        return {}
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


def _iter_demos(f: h5py.File, requested: str | None) -> Iterable[str]:
    demos = sorted(f["data"].keys(), key=_demo_sort_key)
    if requested is None:
        yield demos[0]
        return
    for demo in requested.split(","):
        demo = demo.strip()
        if not demo:
            continue
        if demo not in f["data"]:
            raise ValueError(f"Demo {demo!r} not found. Available examples: {demos[:10]}")
        yield demo


def _role_for(row: dict[str, Any]) -> str:
    typ = str(row.get("type_name", "")).lower()
    name = str(row.get("name", "")).lower()
    body = str(row.get("body_name", "")).lower()
    body_path = str(row.get("body_path", "")).lower()
    label = f"{name} {body} {body_path}"
    tokens = _norm_tokens(name, body, body_path)

    if typ == "background" or int(row.get("id", 0)) < 0:
        return "exclude:background"
    if typ == "geom":
        if name == "floor":
            return "exclude:floor"
        if "tablelink" in body:
            return "exclude:table"
        if "retainingwall" in body:
            return "optional:table_retaining_wall"
        if any(token in body for token in ("controller_box", "pedestal", "pedestal_feet", "torso")):
            return "exclude:robot_base"
        if body == "base":
            return "exclude:robot_base"
        if "buttonbox" in body or body == "box":
            return "optional:button_box"

        # Keep robot gripper matching exact/token based. A substring check for
        # "hand" incorrectly classifies task parts such as "handle" and
        # "HammerHandle" as gripper geometry.
        if (
            body in {"right_hand", "leftpad", "rightpad", "leftclaw", "rightclaw", "hand"}
            or name in {"rightpad_geom", "leftpad_geom", "rail"}
            or _has_any_token(tokens, "rightpad", "leftpad", "rightclaw", "leftclaw")
        ):
            return "robot_gripper"
        if (
            body == "right_arm_base_link"
            or body == "head"
            or re.fullmatch(r"right_l\d+", body or "") is not None
            or any(re.fullmatch(r"right_l\d+", token or "") for token in tokens)
        ):
            return "robot_arm"

        if "wall" in body_path or body == "wall" or "plug_wall" in body:
            return "task_wall"
        if body == "button" or "button" in name or "btn" in name:
            return "task_button"
        if "hammerblock" in label or "nail" in label:
            return "task_fixture"
        if (
            "mug" in body
            or "mug" in name
            or body == "obj"
            or name == "objgeom"
            or "plug" in body
            or "hammer" in body
            or "hammer" in name
        ):
            return "task_object"
        if "handle" in body or "handle" in name or "hdlprs" in body or "hdlprs" in name:
            return "task_handle"
        if any(
            token in label
            for token in (
                "cm_link",
                "cmbutton",
                "door",
                "drawer",
                "faucet",
                "lever",
                "window",
            )
        ):
            return "task_fixture"
    if typ == "site":
        if "endeffector" in name:
            return "optional:gripper_site"
        if any(token in name for token in ("goal", "target", "start", "button", "handle", "hole", "coffee")):
            return "optional:task_site"
        return "exclude:site_marker"
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="List and recommend segmentation selectors stored in a Meta-World HDF5 dataset.")
    parser.add_argument("hdf5_paths", nargs="+", help="Dataset(s) created by collect_metaworld_demos.py")
    parser.add_argument("--demo", default=None, help="Comma-separated demo keys to inspect. Defaults to the first demo.")
    parser.add_argument("--views", default=None, help="Comma-separated camera names. Defaults to every camera in the demo.")
    parser.add_argument("--max-frames", type=int, default=25, help="Maximum frames per demo to scan; use 0 for all frames.")
    parser.add_argument("--top", type=int, default=80, help="Maximum rows to print per environment.")
    parser.add_argument("--selectors", default=None, help="Comma-separated typed selectors, e.g. geom:43,geom:31. Prints a config snippet.")
    parser.add_argument("--ids", default=None, help="Deprecated alias for --selectors. Plain IDs are ambiguous when obj types collide.")
    parser.add_argument("--no-model-lookup", action="store_true", help="Do not instantiate Meta-World to enrich generic geom IDs with body names.")
    args = parser.parse_args()

    selected = _parse_selectors(args.selectors or args.ids)

    env_counts: dict[str, Counter[int]] = defaultdict(Counter)
    env_type_counts: dict[str, Counter[tuple[int, int]]] = defaultdict(Counter)
    env_pixels: Counter[str] = Counter()
    env_infos: dict[str, dict[tuple[int, int], dict[str, Any]]] = defaultdict(dict)
    env_ids: dict[str, str] = {}

    for raw_path in args.hdf5_paths:
        hdf5_path = Path(raw_path)
        if not hdf5_path.exists():
            raise FileNotFoundError(hdf5_path)

        with h5py.File(hdf5_path, "r") as f:
            if "data" not in f:
                raise ValueError(f"{hdf5_path} has no /data group.")
            for demo_key in _iter_demos(f, args.demo):
                demo = f["data"][demo_key]
                env_name = str(demo.attrs.get("env_name", "default"))
                env_id = str(demo.attrs.get("env_id", "Meta-World/MT1"))
                env_ids.setdefault(env_name, env_id)
                env_infos[env_name].update(_load_name_table(demo))
                if not args.no_model_lookup:
                    env_infos[env_name].update(_load_model_table(env_id, env_name))

                obs = demo["obs"]
                camera_names = json.loads(demo.attrs["camera_names"])
                views = _parse_views(args.views) or list(camera_names)
                for view in views:
                    seg_name = f"{view}_seg"
                    if seg_name not in obs:
                        raise ValueError(f"Missing {seg_name} in /data/{demo_key}/obs. Was segmentation enabled during collection?")
                    seg = obs[seg_name]
                    frame_count = seg.shape[0] if args.max_frames <= 0 else min(seg.shape[0], int(args.max_frames))
                    seg_type = obs.get(f"{view}_seg_type", None)
                    for frame_idx in range(frame_count):
                        ids = np.asarray(seg[frame_idx], dtype=np.int32)
                        unique, counts = np.unique(ids, return_counts=True)
                        env_counts[env_name].update({int(k): int(v) for k, v in zip(unique, counts)})
                        env_pixels[env_name] += int(ids.size)
                        if seg_type is not None:
                            types = np.asarray(seg_type[frame_idx], dtype=np.int32)
                            pairs, pair_counts = np.unique(
                                np.stack([ids.reshape(-1), types.reshape(-1)], axis=1),
                                axis=0,
                                return_counts=True,
                            )
                            env_type_counts[env_name].update({(int(row[0]), int(row[1])): int(c) for row, c in zip(pairs, pair_counts)})

    for env_name in sorted(env_counts):
        total = max(1, env_pixels[env_name])
        print(f"\nEnvironment: {env_name}")
        print("selector\tid\ttype\tpixels\tpercent\trole\tname\tbody")
        rows = []
        if env_type_counts[env_name]:
            for (obj_id, obj_type), count in env_type_counts[env_name].items():
                info = dict(env_infos.get(env_name, {}).get((obj_id, obj_type), {}))
                if not info:
                    info = _row(obj_id=obj_id, obj_type=obj_type, name=f"{_type_name(obj_type)}_{obj_id}")
                role = _role_for(info)
                rows.append((count, obj_id, obj_type, info, role))
        else:
            for obj_id, count in env_counts[env_name].items():
                info = _row(obj_id=obj_id, obj_type=-999, name=str(obj_id))
                role = _role_for(info)
                rows.append((count, obj_id, -999, info, role))

        sorted_rows = sorted(rows, key=lambda item: (item[0], item[1], item[2]), reverse=True)

        role_selectors: dict[str, list[str]] = defaultdict(list)
        for count, obj_id, obj_type, info, role in sorted_rows:
            if role and not role.startswith("exclude:"):
                role_selectors[role].append(_selector(obj_id, obj_type))

        for count, obj_id, obj_type, info, role in sorted_rows[: max(1, int(args.top))]:
            name = str(info.get("name", ""))
            body = str(info.get("body_name", ""))
            print(
                f"{_selector(obj_id, obj_type)}\t{obj_id}\t{_type_name(obj_type)}\t"
                f"{count}\t{100.0 * count / total:.3f}\t{role}\t{name}\t{body}"
            )

        core_roles = ["robot_arm", "robot_gripper", "task_object", "task_handle", "task_button", "task_fixture", "task_wall"]
        optional_roles = ["optional:button_box", "optional:table_retaining_wall", "optional:gripper_site", "optional:task_site"]
        task_selectors: list[str] = []
        for role in core_roles:
            task_selectors.extend(role_selectors.get(role, []))
        task_selectors = _dedupe(task_selectors)

        print("\nSuggested role selectors from scanned pixels:")
        for role in core_roles + optional_roles:
            values = role_selectors.get(role, [])
            if values:
                print(f"  {role}: {_yaml_list(_dedupe(values))}")

        if task_selectors:
            print("\nRecommended config snippet for robot + task-relevant geometry:")
            print("dataset:")
            print("  use_segmentation_mask: true")
            print("  selected_seg_ids:")
            print(f"    {env_name}: {_yaml_list(task_selectors)}")
            if role_selectors.get("optional:button_box"):
                with_box = _dedupe(task_selectors + role_selectors["optional:button_box"])
                print("\nOptional if you also want the button housing/box:")
                print(f"    {env_name}: {_yaml_list(with_box)}")

        if selected is not None:
            print("\nConfig snippet for requested selectors:")
            print("dataset:")
            print("  use_segmentation_mask: true")
            print("  selected_seg_ids:")
            print(f"    {env_name}: {_yaml_list(selected)}")

        print("\nTip: prefer typed selectors like \"geom:43\" over plain \"43\" because MuJoCo object IDs collide across types.")


if __name__ == "__main__":
    main()
