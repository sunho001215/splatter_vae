"""Read-only, standard-library-only inspection of a finished run's artifacts on the remote host (2026-10-27).

The remote host has Python 3 but no PyTorch outside the job containers. The controller sends this file's source over SSH
(``RemoteBackend.verify_remote_artifacts``) and receives one small JSON report, so a completed remote run can release its
same-host dependents without first downloading its results:

- ``files``: sha256 and size of every named file (required results plus exit / provenance records); a name that is
  missing, empty, a symlink or outside the run directory is reported as such;
- ``payloads``: the text of the small JSON / YAML records the controller checks (exit.json, provenance.json, and
  completion.json / final.json / config.yaml when present);
- ``archives``: for every ``.pt`` file among the names, the zip archive integrity (``testzip`` CRC of every member)
  and the pickle structure read with inert stand-ins (no global is ever resolved, so no code runs): the top-level
  keys, ``step``, ``format``, ``run_name``, ``source_checkpoint``, ``policy_frames``, ``encoder_config``, and the
  name / dtype / shape of every tensor of ``state_dict`` (exports), ``encoder`` (training checkpoints) or of each module
  of ``policy`` (RL policy snapshots), each checked against its storage record's size, and the pickled globals
  referenced (a weights-only load accepts only tensor and container globals).

    python3 remote_artifact_check.py <run directory> <name> [<name> ...]     (prints one JSON object)
"""
import hashlib
import json
import os
import pickle
import sys
import zipfile

SMALL = ("exit.json", "provenance.json", "completion.json", "final.json", "config.yaml")
SMALL_LIMIT = 1 << 20
ELEMENT = {"FloatStorage": 4, "DoubleStorage": 8, "HalfStorage": 2, "BFloat16Storage": 2, "LongStorage": 8,
           "IntStorage": 4, "ShortStorage": 2, "CharStorage": 1, "ByteStorage": 1, "BoolStorage": 1}


class Inert:
    """Stand-in for any pickled global or object: records its name and arguments, accepts state and items."""

    def __init__(self, name, args=()):
        self.name, self.args, self.items, self.state = name, args, {}, None

    def __setstate__(self, state):
        self.state = state

    def __setitem__(self, key, value):
        self.items[key] = value

    def append(self, value):
        self.items[len(self.items)] = value

    def extend(self, values):
        for value in values:
            self.append(value)


class Global:
    def __init__(self, module, name):
        self.qualname = module + "." + name

    def __call__(self, *args, **kwargs):
        return Inert(self.qualname, args)


class Storage:
    def __init__(self, kind, key, numel):
        self.kind, self.key, self.numel = kind, key, numel


class Tensor:
    def __init__(self, storage, offset, size, stride):
        self.storage, self.offset, self.size, self.stride = storage, offset, tuple(size), tuple(stride)


class Mapping(dict):
    """dict stand-in for OrderedDict (a state dict carries its _metadata as pickled instance state)."""

    def __setstate__(self, state):
        self.state = state


class InertUnpickler(pickle.Unpickler):
    SAFE = {("collections", "OrderedDict"): Mapping, ("builtins", "set"): set, ("builtins", "frozenset"): frozenset,
            ("builtins", "dict"): Mapping, ("builtins", "list"): list, ("builtins", "tuple"): tuple}

    def __init__(self, stream):
        super().__init__(stream)
        self.seen = set()

    def find_class(self, module, name):
        self.seen.add(module + "." + name)
        if (module, name) in self.SAFE:
            return self.SAFE[(module, name)]
        if (module, name) in {("torch._utils", "_rebuild_tensor_v2"), ("torch._utils", "_rebuild_tensor")}:
            return lambda storage, offset, size, stride, *rest: Tensor(storage, offset, size, stride)
        return Global(module, name)

    def persistent_load(self, pid):
        if isinstance(pid, tuple) and pid and pid[0] == "storage" and len(pid) >= 5:
            kind = pid[1].qualname.rsplit(".", 1)[-1] if isinstance(pid[1], Global) else str(pid[1])
            return Storage(kind, str(pid[2]), int(pid[4]))
        raise pickle.UnpicklingError("unsupported persistent id")


def tensors(mapping, entries, prefix):
    out, problems = {}, []
    for name, value in (mapping.items() if isinstance(mapping, dict) else []):
        if not isinstance(value, Tensor):
            continue
        storage = value.storage
        size = 1
        for dim in value.size:
            size *= int(dim)
        element = ELEMENT.get(storage.kind)
        record = entries.get(prefix + "data/" + storage.key)
        contiguous, step = [], 1
        for dim in reversed(value.size):
            contiguous.insert(0, step)
            step *= int(dim)
        extent = int(value.offset) + sum((int(d) - 1) * int(st) for d, st in zip(value.size, value.stride)) + 1
        if (element is None or record is None or record != storage.numel * element or int(value.offset) < 0
                or (size and (list(value.stride) != contiguous or extent > storage.numel))):
            problems.append(name)  # storage size, offset, contiguity and bounds as a weights-only load expects
        out[name] = {"dtype": storage.kind, "shape": list(value.size)}
    return out, problems


def plain(value, depth=0):
    """JSON-safe view of small configuration values (inert objects become their type name)."""
    if depth > 6:
        return "..."
    if isinstance(value, dict):
        return {str(k): plain(v, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v, depth + 1) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return "<" + getattr(value, "name", type(value).__name__) + ">"


def inspect_archive(path):
    report = {"zip_ok": False}
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            return report
        names = archive.namelist()
        pkl = [n for n in names if n.endswith("/data.pkl")]
        if len(pkl) != 1:
            return report
        prefix = pkl[0][: -len("data.pkl")]
        entries = {info.filename: info.file_size for info in archive.infolist()}
        with archive.open(pkl[0]) as stream:
            unpickler = InertUnpickler(stream)
            obj = unpickler.load()
    report["zip_ok"] = True
    report["globals"] = sorted(unpickler.seen)  # a weights-only load accepts only tensor / container globals
    if not isinstance(obj, dict):
        report["top"] = type(obj).__name__
        return report
    report["keys"] = sorted(str(k) for k in obj)
    for key in ("step", "format", "run_name", "source_checkpoint", "policy_frames", "frame_strides",
                "pretraining_objective"):
        if key in obj:
            report[key] = plain(obj[key])
    if isinstance(obj.get("encoder_config"), dict):
        report["encoder_config"] = plain(obj["encoder_config"])
    for key in ("state_dict", "encoder"):
        if isinstance(obj.get(key), dict):
            found, problems = tensors(obj[key], entries, prefix)
            report[key] = found
            report[key + "_storage_problems"] = problems
    if isinstance(obj.get("policy"), dict):  # RL policy snapshots {"step", "policy": {module: state dict}} (2026-10-28)
        report["policy"], report["policy_storage_problems"] = {}, []
        for module, mapping in obj["policy"].items():
            found, problems = tensors(mapping, entries, prefix)
            report["policy"][str(module)] = found if isinstance(mapping, dict) else None
            report["policy_storage_problems"] += [str(module) + "." + name for name in problems]
    return report


def inspect_run(root, names):
    root = os.path.realpath(root)
    out = {"root": root, "files": {}, "payloads": {}, "archives": {}, "problems": []}
    for name in names:
        path = os.path.join(root, name)
        real = os.path.realpath(path)
        if not real.startswith(root + os.sep) or os.path.islink(path) or not os.path.isfile(real):
            out["problems"].append({"name": name, "problem": "missing, symlink or outside the run"})
            continue
        size = os.path.getsize(real)
        digest = hashlib.sha256()
        with open(real, "rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
        out["files"][name] = {"sha256": digest.hexdigest(), "bytes": size}
        if name in SMALL and size <= SMALL_LIMIT:
            with open(real, "rb") as stream:
                out["payloads"][name] = stream.read().decode("utf-8", errors="replace")
        if name.endswith(".pt"):
            try:
                out["archives"][name] = inspect_archive(real)
            except (zipfile.BadZipFile, pickle.UnpicklingError, EOFError, ValueError, KeyError, TypeError,
                    AttributeError, IndexError, OSError) as exc:
                out["archives"][name] = {"zip_ok": False, "error": type(exc).__name__}
    return out


def main(argv):
    sys.stdout.write(json.dumps(inspect_run(argv[1], argv[2:]), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
