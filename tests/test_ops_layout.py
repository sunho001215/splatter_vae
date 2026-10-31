"""Operational code lives in ops/; the old s4d/, scripts/ and runs/setup/tools/ paths stay importable as the same module
objects and runnable with the same command lines."""

from __future__ import annotations

import ast
import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ops.resources.gpu_guard import HOST_NAME

REPO = Path(__file__).resolve().parents[1]
ALIASES = {
    "s4d.remote_access": "ops.remote.access",
    "s4d.remote_artifact_check": "ops.remote.artifact_check",
    "s4d.remote_jobs": "ops.jobs.remote_backend",
    "s4d.lifecycle_cleanup": "ops.jobs.lifecycle",
    "s4d.replay_cleanup_host": "ops.jobs.replay_host",
    "s4d.run_identity": "ops.jobs.run_identity",
    "s4d.gpu_guard": "ops.resources.gpu_guard",
    "s4d.cross_host_equivalence": "ops.resources.host_equivalence",
    "s4d.remote_launch": "ops.containers.launch",
    "s4d.remote_verification": "ops.containers.verification",
    "scripts.jobs": "ops.jobs.scheduler",
    "scripts._nvidia_query": "ops.resources.nvidia_query",
    "scripts.heartbeat": "ops.resources.heartbeat",
    "scripts.watch_events": "ops.resources.watch_events",
    "scripts.sync_media": "ops.remote.sync_media",
    "scripts.cleanup_replay": "ops.jobs.cleanup_replay",
}
# The only research-side imports of ops (lazy, inside functions): output-path authorization and the W&B host tag.
S4D_IMPORTS_OF_OPS = {
    ("s4d/data/__init__.py", "ops.resources.gpu_guard"),
    ("s4d/diag/wandb_log.py", "ops.resources.gpu_guard"),
}


@pytest.mark.parametrize("old,new", sorted(ALIASES.items()))
def test_old_module_path_is_the_same_module_object(old, new):
    module = importlib.import_module(new)
    assert importlib.import_module(old) is module and sys.modules[old] is module
    parent, leaf = old.rsplit(".", 1)
    assert getattr(importlib.import_module(parent), leaf) is module
    shim = REPO / (old.replace(".", "/") + ".py")
    assert len(shim.read_text().splitlines()) < 25  # an alias only: the implementation is not duplicated


def test_moved_modules_keep_their_repository_root():
    from ops.jobs import scheduler
    from ops.remote.access import RemoteClient
    from ops.resources import gpu_guard

    assert scheduler.REPO == gpu_guard.REPO == REPO
    assert scheduler.QUEUE == REPO / "experiments/queue.yaml" and scheduler.RUNS == REPO / "runs"
    assert RemoteClient.__init__.__code__.co_filename == str(REPO / "ops/remote/access.py")


def test_research_package_imports_ops_only_at_documented_points():
    found = set()
    for path in sorted((REPO / "s4d").rglob("*.py")):
        relative = str(path.relative_to(REPO))
        if relative.replace("/", ".")[:-3] in ALIASES:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            names = [node.module or ""] if isinstance(node, ast.ImportFrom) else (
                [alias.name for alias in node.names] if isinstance(node, ast.Import) else [])
            found.update((relative, name) for name in names if name == "ops" or name.startswith("ops."))
    assert found == S4D_IMPORTS_OF_OPS


@pytest.mark.parametrize("script", ["scripts/jobs.py", "scripts/heartbeat.py", "scripts/watch_events.py",
                                    "scripts/sync_media.py", "s4d/remote_access.py", "ops/remote/access.py"])
def test_command_line_entry_points_run_isolated(script):
    result = subprocess.run([sys.executable, "-I", str(REPO / script), "--help"], capture_output=True, text=True,
                            timeout=120)
    assert result.returncode == 0 and result.stdout.startswith("usage: " + Path(script).name)


# ---- refactor pass 2: reusable drivers of runs/setup/tools moved into ops (docs/DEVELOPER.md) ----
TOOLS = REPO / "runs/setup/tools"
TOOL_MODULES = {
    "memory_budget": "ops.resources.memory_budget",
    "memory_watch": "ops.resources.memory_watch",
    "admission_pause": "ops.jobs.admission_pause",
    "pool_placement": "ops.jobs.pool_placement",
    "remote_ledger": "ops.remote.ledger",
    "remote_outage_guard": "ops.remote.outage_guard",
    "sync_wandb_remote": "ops.remote.sync_wandb",
    **{name: f"ops.campaign.{name}" for name in (
        "stage2_local_driver", "baseline_local_driver", "remote_campaign_driver", "pause_pretrain_at_checkpoint",
        "pause_rl_at_checkpoint", "pause_remote_pretrain_at_checkpoint", "reconcile_unlaunched_remote", "release_chain",
        "prepare_remote_verification", "assemble_remote_acceptance", "adopt_remote_suite_evidence")},
}
# Importing these has no side effect beyond reading the pause list. Every other ops.campaign module runs its command
# (a driver pass, a pause, a release step) at import and is only parsed here, never executed.
IMPORTABLE_TOOLS = ("memory_budget", "memory_watch", "admission_pause", "pool_placement", "remote_ledger",
                    "remote_outage_guard", "sync_wandb_remote", "pause_remote_pretrain_at_checkpoint")
# --help of the argparse entry points, byte for byte as before the move (Python 3.10, COLUMNS=80)
TOOL_HELP = {
    "pool_placement": "usage: pool_placement.py [-h] {status,move} ...\n\npositional arguments:\n  {status,move}\n\n"
                      "options:\n  -h, --help     show this help message and exit\n",
    "sync_wandb_remote": "usage: sync_wandb_remote.py [-h] [--run RUN] [--no-upload]\n\noptions:\n"
                         "  -h, --help   show this help message and exit\n  --run RUN\n  --no-upload\n",
}


def tool_path(name: str) -> Path | None:
    """runs/setup/tools/<name>.py, or None inside the release containers, where the runs/ bind mount hides the tracked
    campaign records (tests/test_run_identity.py); everywhere else the entry point must exist."""
    path = TOOLS / f"{name}.py"
    if path.is_file():
        return path
    assert Path("/.dockerenv").exists() or HOST_NAME == "remote", f"{path} is missing"
    return None


ENV = {**os.environ, "COLUMNS": "80"}  # the host config (S4D_HOST_CONFIG) stays; argparse wraps at 80 columns


def run_python(code: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-I", "-c", code, *args], capture_output=True, text=True, timeout=300,
                          cwd=REPO, env=ENV)


@pytest.mark.parametrize("name,module", sorted(TOOL_MODULES.items()))
def test_moved_tool_paths_are_thin_entry_points_of_the_ops_modules(name, module):
    implementation = REPO / (module.replace(".", "/") + ".py")
    compile(implementation.read_text(), str(implementation), "exec")
    shim = tool_path(name)
    if shim is None:
        return
    text = shim.read_text()
    assert len(text.splitlines()) < 25  # an entry point only: the implementation is not duplicated
    parent, leaf = module.rsplit(".", 1)
    nodes = list(ast.walk(ast.parse(text)))
    assert any(isinstance(n, ast.Call) and ast.unparse(n.func) == "runpy.run_module"
               and [ast.unparse(a) for a in n.args] == [repr(module)] and ast.unparse(n.keywords[0]) == "run_name='__main__'"
               for n in nodes)
    assert any(isinstance(n, ast.ImportFrom) and n.module == parent
               and [(a.name, a.asname) for a in n.names] == [(leaf, "_impl")] for n in nodes)
    assert any(isinstance(n, ast.Assign) and [ast.unparse(t) for t in n.targets] == ["sys.modules[__name__]"]
               and ast.unparse(n.value) == "_impl" for n in nodes)
    assert "Path(__file__).resolve().parents[3]" in text  # the repository root seen from runs/setup/tools


@pytest.mark.parametrize("name", IMPORTABLE_TOOLS)
def test_importable_tool_paths_are_the_same_module_objects(name):
    shim = tool_path(name)
    code = ("import importlib, sys\n"
            f"sys.path[:0] = [{str(TOOLS)!r}, {str(REPO)!r}]\n"
            f"new = importlib.import_module({TOOL_MODULES[name]!r})\n"
            f"if {shim is not None}:\n"
            f"    old = importlib.import_module({name!r})\n"
            f"    assert old is new and sys.modules[{name!r}] is new, old\n"
            "print(new.__file__)\n")
    result = run_python(code)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(REPO / (TOOL_MODULES[name].replace(".", "/") + ".py"))


def test_memory_watch_puts_the_same_tools_directory_on_sys_path():
    """Its sys.path entry for its former directory is computed from the new location; the value is unchanged."""
    result = run_python(f"import sys\nsys.path.insert(0, {str(REPO)!r})\nimport ops.resources.memory_watch\n"
                        "print(sys.path[0])\n")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(TOOLS)


@pytest.mark.parametrize("name", sorted(TOOL_HELP))
def test_argparse_tool_entry_points_keep_their_help(name):
    shim = tool_path(name)
    if shim is not None:
        result = subprocess.run([sys.executable, "-I", str(shim), "--help"], capture_output=True, text=True,
                                timeout=300, cwd=REPO, env=ENV)
    else:  # the implementation, run as the entry point runs it (sys.argv[0] names the entry point)
        result = run_python(f"import runpy, sys\nsys.path.insert(0, {str(REPO)!r})\nsys.argv[0] = {name + '.py'!r}\n"
                            f"runpy.run_module({TOOL_MODULES[name]!r}, run_name='__main__')\n", "--help")
    assert (result.returncode, result.stdout, result.stderr) == (0, TOOL_HELP[name], "")


def test_campaign_command_modules_are_not_imported_by_other_code():
    """Importing an ops.campaign command module runs it; only the runs/setup/tools entry points refer to them."""
    found = []
    for folder in ("s4d", "ops", "scripts", "tests"):
        for path in sorted((REPO / folder).rglob("*.py")):
            for node in ast.walk(ast.parse(path.read_text())):
                names = ([node.module or ""] + [f"{node.module}.{a.name}" for a in node.names]
                         if isinstance(node, ast.ImportFrom) else
                         [a.name for a in node.names] if isinstance(node, ast.Import) else [])
                if any(n == "ops.campaign" or n.startswith("ops.campaign.") for n in names):
                    found.append(str(path.relative_to(REPO)))
    assert found == []


def test_tool_records_still_compile_and_find_the_moved_names():
    """Every tool, dated one-off and .before-* backup in runs/setup/tools still compiles; the current tools and the dated
    one-offs that import a moved module by its old name (``import pool_placement as pool``, ``from admission_pause
    import paused``) only use names the ops implementation defines."""
    if tool_path("pool_placement") is None:
        return
    files = sorted(p for p in TOOLS.iterdir() if p.suffix == ".py" or ".py.before-" in p.name)
    assert len(files) > 150
    defined = {}
    for name, module in TOOL_MODULES.items():
        tree = ast.parse((REPO / (module.replace(".", "/") + ".py")).read_text())
        defined[name] = {t.id for n in ast.walk(tree) if isinstance(n, (ast.Assign, ast.AnnAssign)) and n.col_offset == 0
                         for target in (n.targets if isinstance(n, ast.Assign) else [n.target])
                         for t in ast.walk(target) if isinstance(t, ast.Name)}
        defined[name] |= {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        defined[name] |= {(a.asname or a.name).split(".")[0] for n in ast.walk(tree)
                          if isinstance(n, (ast.Import, ast.ImportFrom)) and n.col_offset == 0 for a in n.names}
    missing = []
    for path in files:
        tree = ast.parse(path.read_text(), str(path))
        compile(tree, str(path), "exec")
        if ".before-" in path.name or path.stem in TOOL_MODULES:
            continue  # backups record older interfaces; entry points are checked above
        aliases = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                aliases.update({a.asname or a.name: a.name for a in node.names if a.name in TOOL_MODULES})
            elif isinstance(node, ast.ImportFrom) and node.module in TOOL_MODULES:
                missing += [(path.name, f"{node.module}.{a.name}") for a in node.names if a.name not in defined[node.module]]
        missing += [(path.name, f"{aliases[n.value.id]}.{n.attr}") for n in ast.walk(tree)
                    if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id in aliases
                    and n.attr not in defined[aliases[n.value.id]]]
    assert missing == []
