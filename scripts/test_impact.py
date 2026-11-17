"""Change-impact test selection and reusable per-file results for scripts/run_tests.py (2026-11-17).

Dependency graph (static: nothing is imported or executed). A test file depends on:
  * every repository Python file it imports, transitively: ``import``/``from`` statements anywhere (module level, inside
    functions, relative), resolved against the roots the suite puts on ``sys.path`` (repository root, scripts/, tests/,
    runs/setup/tools/ and the importing file's own directory) plus parent ``__init__.py`` files; repository module
    names given as strings (importlib or monkeypatch targets) resolve too, except test modules (a ``sys.modules`` lookup
    of another test module does not execute it);
  * path literals of test-side files (tests/): a string constant, a token of a command line, an f-string's constant
    prefix or a ``REPO / "a" / "b"`` chain naming an existing repository file or directory, relative to the repository
    or to the referencing file. A directory adds every file below it, a Python file adds its own closure (a script or
    worker the test runs), an absolute path outside the repository is an external input. Naming a test file (shard
    tables, regime lists) is not a dependency;
  * path literals of library files (everything else): only data files they name (not Python, tests, docs, run records
    or experiment state) and whole directories under LIBRARY_DATA_DIRS (configuration trees);
  * tests/conftest.py: its own and its test-side helpers' imports, and the import-time (module-level) closure of the
    library modules they import - that is what runs for every test;
  * scope guards: a test file naming an execution-scope API depends on every file under EXECUTION_SCOPE_ROOTS, a test
    file naming a source-fingerprint API on every file under FINGERPRINT_ROOTS (they compare those trees with commits).

Selection (``--changed BASE``): changed = ``git diff BASE...HEAD`` + uncommitted (staged/unstaged) + untracked files.
  1. global   - a GLOBAL_FILES/GLOBAL_PATTERNS path changed: every test file;
     protected - a robot-facing, GPU-guard or container-launch path (PROTECTED) changed: every test file;
  2. test     - a changed test file selects itself (a deleted one is reported, not run); any added, changed or deleted
                test file also selects INVENTORY_TESTS (they check the test-file inventory);
  3. closure  - every test file whose dependencies contain a changed path (or a deleted module it imports), including
                test files that import a changed test module as a helper;
  4. widen    - a changed path under tests/, or a changed non-Python path under configs/, that no test file depends on
                specifically (scope guards depend on whole trees and do not count): every test file (a test may read it
                through a path the static scan cannot see);
  anything else no test depends on (documentation, run records, unimported code) selects nothing and is reported as
  ``unattributed``.

Reuse: a test file's key is the sha256 of its path, the sha256 of every dependency (files, external files, the third-
party identity), the interpreter, torch and CUDA runtime versions, the host and the visible device names, and the git
HEAD for files that name it. Files whose node ids passed are recorded under their key; a later run with the same key
whose selected node ids are a subset of the recorded ones reuses the pass instead of executing the file. Files with
live-data markers (they read host data the key cannot hash) or unhashable external inputs are never reused. Failures,
skips and errors are never recorded.
"""
from __future__ import annotations

import argparse
import ast
import bisect
import fnmatch
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

REPO = Path(__file__).resolve().parents[1]
CONFTEST = "tests/conftest.py"
IMPORT_ROOTS = ("", "scripts", "tests", "runs/setup/tools")
# Any change here can alter every test's outcome: the runner and its worker preamble, the session conftest, the
# environment lock and pytest configuration, the pinned third-party trees. (scripts/test_shards.py only schedules files
# and scripts/test_impact.py only selects them: their own tests import them, so they are ordinary closure members.)
GLOBAL_FILES = frozenset({CONFTEST, "scripts/run_tests.py", "scripts/_bootstrap.py", "pyproject.toml", "uv.lock",
                          ".gitmodules", "pytest.ini", "setup.cfg", "tox.ini", "conftest.py", ".python-version"})
GLOBAL_PATTERNS = ("requirements*.txt", "third_party", "third_party/*")
# Robot-facing, GPU-guard and container-launch code: a change here always runs every test file (protected paths,
# infra roadmap 2026-11-17, C5), whatever its static closure says.
PROTECTED = ("ops/robot_pc/", "scripts/robot_pc/", "configs/robot_pc/", "ops/policy_serving/", "ops/inference_server/",
             "s4d/rwlearn/deploy/", "configs/realworld/policy_serving.yaml", "configs/realworld/inference_server.yaml",
             "ops/resources/gpu_guard.py", "s4d/gpu_guard.py", "ops/containers/launch.py", "mujoco_mig_setup.py")
INVENTORY_TESTS = ("tests/test_run_tests_sharding.py",)
WIDEN_ANY = ("tests/",)          # unattributed changes here (Python or not) widen to every test file
WIDEN_DATA = ("configs/",)       # unattributed non-Python changes here widen to every test file
EXECUTION_SCOPE_GUARD = re.compile(r"\b(?:execution_scope|placement_scope|dsreset_scope|closure_impact|import_closure)\b")
EXECUTION_SCOPE_ROOTS = ("s4d/", "ops/", "scripts/", "configs/", "runs/setup/tools/", "mujoco_mig_setup.py")
FINGERPRINT_GUARD = re.compile(r"\b(?:source_fingerprints|source_identity|commit_source_sha256)\b")
# _bootstrap.source_fingerprints also hashes tests/, but the guards compare a tree with itself or with a commit, so a
# test file's content does not change their outcome
FINGERPRINT_ROOTS = ("s4d/", "ops/", "scripts/", "configs/", "mujoco_mig_setup.py")
LIVE_DATA_MARKERS = re.compile(r"\b(?:metaworld_data|local_host_data)\b")
HEAD_READER = re.compile(r"""["']HEAD["']""")
LIBRARY_DATA_DIRS = ("configs/",)
LIBRARY_IGNORED = ("tests/", "docs/", "runs/", "experiments/")
DOTTED_MODULE = re.compile(r"(?:s4d|ops|scripts|tests|mujoco_mig_setup)(?:\.[A-Za-z_]\w*)+")
REPO_EXPRESSION = re.compile(r"repo|root|__file__|parent|cwd", re.I)
TOKEN_SPLIT = re.compile(r"""[\s:=,;'"()\[\]{}<>|*?!&^%$#@`]+""")
IGNORED_EXTERNAL = (Path("/tmp"), Path("/proc"), Path("/dev"), Path("/sys"))
MAX_EXTERNAL_BYTES = 256 << 20
CACHE_VERSION = 1


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=True).stdout


def _is_test_module(relative: str) -> bool:
    return relative.startswith("tests/") and PurePosixPath(relative).name.startswith("test_")


def _test_side(relative: str) -> bool:
    return relative.startswith("tests/")


# ---------------------------------------------------------------------------------------------------- file sources
class Tree:
    """The working tree: tracked plus untracked-but-not-ignored files (git), read from disk."""

    def __init__(self, root: Path = REPO):
        self.root = Path(root)
        listing = _git(self.root, "ls-files", "-z", "--cached", "--others", "--exclude-standard")
        self._index(p for p in listing.split("\0") if p and (self.root / p).is_file())

    def _index(self, files) -> None:
        self.files = sorted(set(files))
        self._set = set(self.files)

    def exists(self, relative: str) -> bool:
        return relative in self._set

    def under(self, prefix: str) -> list[str]:
        prefix = prefix.rstrip("/") + "/"
        out = []
        for path in self.files[bisect.bisect_left(self.files, prefix):]:
            if not path.startswith(prefix):
                break
            out.append(path)
        return out

    def read(self, relative: str) -> bytes | None:
        try:
            return (self.root / relative).read_bytes()
        except OSError:
            return None


class CommitTree(Tree):
    """A committed tree read from git objects (dry runs over historical ranges) through one ``cat-file --batch``."""

    def __init__(self, root: Path, commit: str):
        self.root, self.commit, self._batch = Path(root), commit, None
        self._index(p for p in _git(self.root, "ls-tree", "-r", "-z", "--name-only", commit).split("\0") if p)

    def read(self, relative: str) -> bytes | None:
        if relative not in self._set or "\n" in relative:
            return None
        if self._batch is None:
            self._batch = subprocess.Popen(["git", "-C", str(self.root), "cat-file", "--batch"],
                                           stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        self._batch.stdin.write(f"{self.commit}:{relative}\n".encode())
        self._batch.stdin.flush()
        header = self._batch.stdout.readline().split()
        if len(header) != 3 or header[1] != b"blob":
            return None
        data = self._batch.stdout.read(int(header[2]))
        self._batch.stdout.read(1)  # the newline after each object
        return data

    def __del__(self):
        if getattr(self, "_batch", None) is not None:
            self._batch.stdin.close()
            self._batch.wait()


# ---------------------------------------------------------------------------------------------------- the graph
@dataclass
class Node:
    imports: set[str] = field(default_factory=set)   # repository .py files imported anywhere (or run by path)
    eager: set[str] = field(default_factory=set)     # the subset imported at import time (outside function bodies)
    files: set[str] = field(default_factory=set)     # repository non-.py files named by path
    dirs: set[str] = field(default_factory=set)      # repository directories named by path ("a/b/")
    external: set[str] = field(default_factory=set)  # absolute paths outside the repository
    missing: set[str] = field(default_factory=set)   # module paths an unresolved import would have used


@dataclass
class Closure:
    files: set[str] = field(default_factory=set)     # every repository file it depends on (directories expanded)
    dirs: set[str] = field(default_factory=set)
    guard_dirs: set[str] = field(default_factory=set)  # whole trees a scope guard compares (also expanded into files)
    external: set[str] = field(default_factory=set)
    missing: set[str] = field(default_factory=set)
    guard_files: set[str] = field(default_factory=set)
    scope_guard: bool = False
    live_data: bool = False
    reads_head: bool = False


class _Scanner(ast.NodeVisitor):
    """Collects imports (and whether they run at import time), strings and string path chains of one module."""

    def __init__(self):
        self.depth = 0
        self.imports: list[tuple[str, int, list[str], bool]] = []  # (module, level, names, eager)
        self.strings: list[str] = []
        self.paths: list[tuple[str, bool]] = []                    # ("a/b", repo-relative base)
        self.prefixes: list[str] = []                              # f-string constant prefixes
        self._inner: set[int] = set()

    def _function(self, node):
        for decorator in getattr(node, "decorator_list", []):
            self.visit(decorator)
        self.depth += 1
        for child in ast.iter_child_nodes(node):
            if child not in getattr(node, "decorator_list", []):
                self.visit(child)
        self.depth -= 1

    visit_FunctionDef = visit_AsyncFunctionDef = visit_Lambda = _function

    def visit_Call(self, node):
        # sys.path.insert(0, str(REPO / "scripts")) adds an import root (resolved through IMPORT_ROOTS); it does not
        # read the directory, so its arguments name no dependency
        if ast.unparse(node.func) in ("sys.path.insert", "sys.path.append"):
            return
        self.generic_visit(node)

    def visit_Import(self, node):
        for alias in node.names:
            self.imports.append((alias.name, 0, [], self.depth == 0))

    def visit_ImportFrom(self, node):
        self.imports.append((node.module or "", node.level, [a.name for a in node.names], self.depth == 0))

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            self.strings.append(node.value)

    def visit_JoinedStr(self, node):
        prefix = ""
        for part in node.values:
            if not (isinstance(part, ast.Constant) and isinstance(part.value, str)):
                break
            prefix += part.value
        self.prefixes.append(prefix)
        self.generic_visit(node)

    def visit_BinOp(self, node):
        if _string_division(node) and id(node) not in self._inner:
            segments, base = [], node
            while _string_division(base):
                segments.insert(0, base.right.value)
                if isinstance(base.left, ast.BinOp):
                    self._inner.add(id(base.left))
                base = base.left
            self.paths.append(("/".join(s.strip("/") for s in segments),
                               bool(REPO_EXPRESSION.search(ast.unparse(base)))))
        self.generic_visit(node)


def _string_division(node) -> bool:
    return (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div) and isinstance(node.right, ast.Constant)
            and isinstance(node.right.value, str))


class Graph:
    def __init__(self, tree: Tree):
        self.tree = tree
        self.root = tree.root
        self._nodes: dict[str, Node] = {}
        self._closures: dict[str, Closure] = {}
        self._conftest: Closure | None = None
        tops = {f.split("/", 1)[0] for f in tree.files if "/" in f}
        tops |= {f[:-3] for f in tree.files if "/" not in f and f.endswith(".py")}
        for root in IMPORT_ROOTS[1:]:
            tops |= {PurePosixPath(f).relative_to(root).parts[0].removesuffix(".py") for f in tree.under(root)}
        self.tops = tops - {"third_party"}

    # -- module resolution
    def module_files(self, name: str, importer: str) -> tuple[list[str], list[str]]:
        """(existing files, candidate paths) of module ``name`` imported from ``importer``."""
        parts = [p for p in name.split(".") if p]
        if not parts:
            return [], []
        importer_dir = str(PurePosixPath(importer).parent)
        found, candidates = [], []
        for root in dict.fromkeys([*IMPORT_ROOTS, "" if importer_dir == "." else importer_dir]):
            prefix = [root] if root else []
            base = "/".join(prefix + parts)
            for candidate in (base + ".py", base + "/__init__.py"):
                candidates.append(candidate)
                if self.tree.exists(candidate):
                    found.append(candidate)
                    for depth in range(1, len(parts)):  # importing a.b.c executes a/__init__.py and a/b/__init__.py
                        init = "/".join(prefix + parts[:depth] + ["__init__.py"])
                        if self.tree.exists(init):
                            found.append(init)
        return found, candidates

    def _dotted(self, name: str, importer: str, string: bool) -> tuple[list[str], list[str]]:
        """The longest prefix of ``a.b.c`` that is a module (``c`` may be an attribute)."""
        parts = name.split(".")
        missing = []
        for end in range(len(parts), 0, -1):
            found, candidates = self.module_files(".".join(parts[:end]), importer)
            if string:
                found = [f for f in found if not _is_test_module(f)]
            if found:
                return found, missing
            if not string and end == len(parts) and parts[0] in self.tops:
                missing = candidates
        return [], missing

    # -- path literals
    def _path(self, text: str, importer: str, node: Node, repo_relative: bool = False) -> None:
        importer_dir = PurePosixPath(importer).parent
        test_side = _test_side(importer)
        for token in TOKEN_SPLIT.split(text):
            token = token.strip().rstrip(".")
            if len(token) < 2 or len(token) > 300 or token in ("..", "./"):
                continue
            if token.startswith("/"):
                if test_side:
                    self._absolute(token, node)
                continue
            bare = "/" not in token.rstrip("/") and "." not in token
            bases = [importer_dir] if bare and not repo_relative else [PurePosixPath(""), importer_dir]
            for base in bases:
                relative = os.path.normpath(str(base / token.removeprefix("./")))
                if relative.startswith("..") or relative in (".", ""):
                    continue
                if self.tree.exists(relative):
                    if _is_test_module(relative):
                        continue  # naming a test file (shard tables, regime lists) is not reading it
                    if not test_side:
                        if not relative.endswith((".py", ".md")) and not relative.startswith(LIBRARY_IGNORED):
                            node.files.add(relative)
                    elif relative.endswith(".py"):
                        node.imports.add(relative)  # a script or worker the test runs or loads by path
                    else:
                        node.files.add(relative)
                elif self.tree.under(relative):
                    directory = relative.rstrip("/") + "/"
                    if test_side or directory.startswith(LIBRARY_DATA_DIRS):
                        node.dirs.add(directory)

    def _absolute(self, token: str, node: Node) -> None:
        path = Path(os.path.normpath(token))
        if path == self.root or self.root in path.parents:
            self._path(str(path.relative_to(self.root)) if path != self.root else "", "tests/", node,
                       repo_relative=True)
        elif not any(path == p or p in path.parents for p in IGNORED_EXTERNAL) and path.exists():
            node.external.add(str(path))

    def node(self, relative: str) -> Node:
        if relative in self._nodes:
            return self._nodes[relative]
        node = self._nodes[relative] = Node()
        data = self.tree.read(relative)
        if data is None or not relative.endswith(".py"):
            return node
        try:
            scanner = _Scanner()
            scanner.visit(ast.parse(data))
        except (SyntaxError, ValueError, RecursionError):
            return node
        package = PurePosixPath(relative).parent
        for module, level, names, eager in scanner.imports:
            if level:
                base_dir = package
                for _ in range(level - 1):
                    base_dir = base_dir.parent
                module = ".".join([*base_dir.parts, *(module.split(".") if module else [])])
            found, missing = self._dotted(module, relative, string=False) if module else ([], [])
            for name in names:
                if name != "*" and module:
                    found += self.module_files(f"{module}.{name}", relative)[0]
            node.imports.update(found)
            node.missing.update(missing)
            if eager:
                node.eager.update(found)
        for value in scanner.strings:
            if DOTTED_MODULE.fullmatch(value):
                node.imports.update(self._dotted(value, relative, string=True)[0])
            if "/" in value or "." in value or value.isidentifier():
                self._path(value, relative, node)
        for path, repo_relative in scanner.paths:
            self._path(path, relative, node, repo_relative=repo_relative)
        for prefix in scanner.prefixes:
            if "/" in prefix:
                head = prefix.rsplit("/", 1)[0]
                if head:
                    self._path(head + "/", relative, node)
            elif prefix.endswith(".") and DOTTED_MODULE.fullmatch(prefix + "x"):
                for found in self._dotted(prefix[:-1], relative, string=True)[0]:
                    if found.endswith("__init__.py"):
                        node.dirs.add(str(PurePosixPath(found).parent) + "/")
        node.imports.discard(relative)
        node.eager.discard(relative)
        return node

    def _walk(self, starts: list[str], eager_library: bool) -> Closure:
        out, todo, seen = Closure(), list(starts), set()
        while todo:
            relative = todo.pop()
            if relative in seen:
                continue
            seen.add(relative)
            node = self.node(relative)
            follow = node.eager if eager_library and not _test_side(relative) else node.imports
            todo.extend(f for f in follow if f not in seen)
            out.files.update(node.files)
            out.dirs.update(node.dirs)
            out.external.update(node.external)
            out.missing.update(node.missing)
        out.files.update(seen)
        return out

    def conftest(self) -> Closure:
        if self._conftest is None:
            self._conftest = self._walk([CONFTEST], eager_library=True) if self.tree.exists(CONFTEST) else Closure()
        return self._conftest

    def closure(self, test_file: str) -> Closure:
        if test_file in self._closures:
            return self._closures[test_file]
        out = self._walk([test_file], eager_library=False)
        shared = self.conftest()
        out.files |= shared.files
        out.dirs |= shared.dirs
        out.external |= shared.external
        out.missing |= shared.missing
        source = (self.tree.read(test_file) or b"").decode("utf-8", "replace")
        roots = (EXECUTION_SCOPE_ROOTS if EXECUTION_SCOPE_GUARD.search(source) else ()) + \
                (FINGERPRINT_ROOTS if FINGERPRINT_GUARD.search(source) else ())
        out.scope_guard = bool(roots)
        out.live_data = bool(LIVE_DATA_MARKERS.search(source))
        out.reads_head = bool(HEAD_READER.search(source))
        for directory in out.dirs:
            out.files.update(self.tree.under(directory))
        out.guard_dirs.update(r for r in roots if r.endswith("/"))
        out.guard_files = {r for r in roots if not r.endswith("/") and self.tree.exists(r)}
        for directory in out.guard_dirs:
            out.guard_files.update(self.tree.under(directory))
        self._closures[test_file] = out
        return out

    def depends_on(self, test_file: str, changed: str, specific: bool = False) -> bool:
        """Whether ``changed`` is a dependency; ``specific`` ignores the whole trees a scope guard compares."""
        closure = self.closure(test_file)
        if changed in closure.files or changed in closure.missing or any(changed.startswith(d) for d in closure.dirs):
            return True
        return not specific and (changed in closure.guard_files
                                 or any(changed.startswith(d) for d in closure.guard_dirs))


def suite_files(tree: Tree) -> list[str]:
    return sorted(f for f in tree.files if PurePosixPath(f).parent == PurePosixPath("tests")
                  and fnmatch.fnmatch(PurePosixPath(f).name, "test_*.py"))


def is_global(path: str) -> bool:
    return path in GLOBAL_FILES or any(fnmatch.fnmatch(path, pattern) for pattern in GLOBAL_PATTERNS)


def is_protected(path: str) -> bool:
    return any(path == p or (p.endswith("/") and path.startswith(p)) for p in PROTECTED)


# ---------------------------------------------------------------------------------------------------- selection
def changed_files(base: str, root: Path = REPO, head: str = "HEAD", worktree: bool = True) -> list[str]:
    """Paths changed on ``head`` since its merge base with ``base``, plus (``worktree``) uncommitted and untracked ones."""
    out = set(_git(root, "diff", "--name-only", "--no-renames", "-z", f"{base}...{head}").split("\0"))
    if worktree:
        out |= set(_git(root, "diff", "--name-only", "--no-renames", "-z", "HEAD").split("\0"))
        out |= set(_git(root, "ls-files", "-z", "--others", "--exclude-standard").split("\0"))
    return sorted(p for p in out if p)


def select(graph: Graph, changed: list[str]) -> dict:
    """The test files a change set can affect: {"files", "all", "reasons", "global", "widened", "unattributed", ...}."""
    tests = suite_files(graph.tree)
    reasons: dict[str, set[str]] = {}

    def add(test: str, why: str) -> None:
        reasons.setdefault(test, set()).add(why)

    global_hits = [p for p in changed if is_global(p)]
    protected_hits = [p for p in changed if is_protected(p) and not is_global(p)]
    widened, unattributed, deleted = [], [], []
    for path in changed:
        if is_global(path):
            continue
        test_module = PurePosixPath(path).parent == PurePosixPath("tests") and fnmatch.fnmatch(
            PurePosixPath(path).name, "test_*.py")
        if test_module:
            for inventory in INVENTORY_TESTS:
                if inventory in tests:
                    add(inventory, f"inventory:{path}")
            if path in tests:
                add(path, f"test:{path}")
            else:
                deleted.append(path)
        hits = [t for t in tests if t != path and graph.depends_on(t, path)]
        for test in hits:
            add(test, f"closure:{path}")
        if test_module or any(graph.depends_on(t, path, specific=True) for t in hits):
            continue
        if path.startswith(WIDEN_ANY) or (path.startswith(WIDEN_DATA) and not path.endswith(".py")):
            widened.append(path)
        elif not hits:
            unattributed.append(path)
    for test in tests:
        for path in global_hits[:3]:
            add(test, f"global:{path}")
        for path in protected_hits[:3]:
            add(test, f"protected:{path}")
        for path in widened[:3]:
            add(test, f"widen:{path}")
    selected = sorted(reasons)
    return {"files": selected, "all": bool(tests) and len(selected) == len(tests), "total_files": len(tests),
            "changed": list(changed), "global": global_hits, "protected": protected_hits, "widened": widened,
            "unattributed": unattributed,
            "deleted_tests": deleted, "reasons": {t: sorted(r)[:8] for t, r in sorted(reasons.items())}}


# ---------------------------------------------------------------------------------------------------- reuse keys
class Hasher:
    def __init__(self, tree: Tree):
        self.tree = tree
        self._memo: dict[str, str | None] = {}

    def file(self, relative: str) -> str | None:
        if relative not in self._memo:
            data = self.tree.read(relative)
            self._memo[relative] = None if data is None else hashlib.sha256(data).hexdigest()
        return self._memo[relative]

    def external(self, path: str) -> str | None:
        """sha256 of a regular external file; "dir" for a directory (named, not hashed); None when unhashable."""
        key = "abs:" + path
        if key not in self._memo:
            target = Path(path)
            try:
                if target.is_dir():
                    self._memo[key] = "dir"
                elif target.is_file() and target.stat().st_size <= MAX_EXTERNAL_BYTES:
                    self._memo[key] = hashlib.sha256(target.read_bytes()).hexdigest()
                else:
                    self._memo[key] = None
            except OSError:
                self._memo[key] = None
        return self._memo[key]


def environment_identity(gpu_mapping: list[dict] | None = None, host: str | None = None) -> dict:
    import torch

    return {"python": sys.version, "prefix": sys.prefix, "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
            "host": host, "devices": sorted(str(d.get("name")) for d in (gpu_mapping or []))}


def reuse_key(graph: Graph, hasher: Hasher, test_file: str, environment: dict,
              third_party: dict) -> tuple[str | None, str]:
    """(key, reason); a None key means the file is never reused."""
    closure = graph.closure(test_file)
    if closure.live_data:
        return None, "live-data markers"
    deps = {}
    for relative in sorted(closure.files | closure.guard_files):
        digest = hasher.file(relative)
        if digest is None:
            return None, f"unreadable dependency {relative}"
        deps[relative] = digest
    external = {}
    for path in sorted(closure.external):
        digest = hasher.external(path)
        if digest is None:
            return None, f"unhashable external input {path}"
        external[path] = digest
    material = {"version": CACHE_VERSION, "file": test_file, "deps": deps, "external": external,
                "third_party": third_party, "environment": environment}
    if closure.reads_head:
        material["head"] = _git(graph.root, "rev-parse", "HEAD").strip()
    blob = json.dumps(material, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest(), "ok"


class ResultCache:
    """Content-addressed records of passed node ids: ``<root>/<key[:2]>/<key>.json``."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def lookup(self, key: str, test_file: str) -> dict | None:
        try:
            entry = json.loads(self.path(key).read_text())
        except (OSError, ValueError):
            return None
        if (not isinstance(entry, dict) or entry.get("version") != CACHE_VERSION or entry.get("key") != key
                or entry.get("file") != test_file or not isinstance(entry.get("passed"), list)
                or not all(isinstance(n, str) for n in entry["passed"])):
            return None
        return entry

    def record(self, key: str, test_file: str, passed: list[str], **meta) -> Path:
        previous = self.lookup(key, test_file)
        entry = {"version": CACHE_VERSION, "key": key, "file": test_file,
                 "passed": sorted(set(passed) | set(previous["passed"] if previous else [])),
                 "recorded_utc": datetime.now(timezone.utc).isoformat(), **meta}
        path = self.path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, suffix=".tmp") as stream:
            json.dump(entry, stream, indent=1)
            temporary = Path(stream.name)
        os.replace(temporary, path)
        return path


def main_checkout(root: Path = REPO) -> Path:
    """The primary checkout of this repository (the worktree that owns the common git directory)."""
    try:
        common = Path(_git(root, "rev-parse", "--path-format=absolute", "--git-common-dir").strip())
    except (OSError, subprocess.CalledProcessError):
        return Path(root)
    return common.parent if common.name == ".git" else Path(root)


def cache_root(root: Path = REPO) -> Path:
    """``S4D_TEST_RESULT_CACHE`` or ``<checkout>/.cache/test-results``; it must lie in the .cache of this checkout or of
    the main checkout (to share results between worktrees), as every repository cache must (gpu_guard)."""
    from ops.resources.gpu_guard import validate_runtime_path

    path = Path(os.environ.get("S4D_TEST_RESULT_CACHE") or Path(root) / ".cache" / "test-results").resolve()
    for repository in dict.fromkeys([Path(root).resolve(), main_checkout(root).resolve()]):
        if repository in path.parents:
            return validate_runtime_path(path, repository=repository, cache=True)
    return validate_runtime_path(path, repository=Path(root), cache=True)  # fails closed (outside every checkout)


def plan_reuse(graph: Graph, cache: ResultCache | None, nodeids: list[str], environment: dict, third_party: dict,
               lookup: bool = True) -> dict:
    """Split ``nodeids`` into those to run and whole files whose recorded pass is reused.

    Returns {"run": nodeids to execute (collection order), "reused": [...], "keys": {file: key or None},
    "ineligible": {file: reason}}. With ``lookup`` False only the keys are computed (for recording)."""
    hasher, tests = Hasher(graph.tree), set(suite_files(graph.tree))
    keys, ineligible, reused = {}, {}, []
    for test_file, nodes in by_file(nodeids).items():
        if test_file not in tests:
            keys[test_file], ineligible[test_file] = None, "not a tests/test_*.py module of this checkout"
            continue
        key, reason = reuse_key(graph, hasher, test_file, environment, third_party)
        keys[test_file] = key
        if key is None:
            ineligible[test_file] = reason
            continue
        entry = cache.lookup(key, test_file) if lookup and cache is not None else None
        if entry is not None and set(nodes) <= set(entry["passed"]):
            reused.append({"file": test_file, "key": key, "tests": len(nodes), "recorded_utc": entry.get("recorded_utc"),
                           "commit": entry.get("commit")})
    skipped = {item["file"] for item in reused}
    return {"run": [n for n in nodeids if n.split("::", 1)[0] not in skipped], "reused": reused, "keys": keys,
            "ineligible": ineligible}


def record_passes(cache: ResultCache, keys: dict, selected: list[str], outcomes: list[dict], **meta) -> list[str]:
    """Record every file whose selected node ids all passed; returns the recorded files."""
    passed, recorded = set(passed_nodes(outcomes)), []
    for test_file, nodes in by_file(selected).items():
        key = keys.get(test_file)
        if key and nodes and set(nodes) <= passed:
            cache.record(key, test_file, nodes, **meta)
            recorded.append(test_file)
    return recorded


def passed_nodes(outcomes: list[dict]) -> list[str]:
    """Node ids whose setup, call and teardown all passed and whose subtests neither failed nor skipped."""
    out = []
    for outcome in outcomes:
        phases = outcome.get("phases", [])
        if (outcome.get("completed") is True and [p.get("when") for p in phases] == ["setup", "call", "teardown"]
                and all(p.get("outcome") == "passed" for p in [*phases, *outcome.get("subtests", [])])):
            out.append(outcome["nodeid"])
    return out


def by_file(nodeids: list[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for node in nodeids:
        out.setdefault(node.split("::", 1)[0], []).append(node)
    return out


# ---------------------------------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Dry-run the change-impact selection (nothing is executed).")
    ap.add_argument("--base", required=True, help="select for BASE...HEAD (from the merge base)")
    ap.add_argument("--head", help="a commit: use that committed tree and ignore the working tree")
    ap.add_argument("--json", action="store_true")
    options = ap.parse_args(argv)
    tree = CommitTree(REPO, options.head) if options.head else Tree(REPO)
    changed = changed_files(options.base, REPO, options.head or "HEAD", worktree=options.head is None)
    plan = select(Graph(tree), changed)
    if options.json:
        print(json.dumps(plan, indent=1))
        return 0
    print(f"changed {len(changed)} path(s); selected {len(plan['files'])}/{plan['total_files']} test files"
          + (" (complete suite)" if plan["all"] else ""))
    for name in ("global", "protected", "widened", "unattributed", "deleted_tests"):
        if plan[name]:
            print(f"  {name}: {', '.join(plan[name][:10])}" + (" ..." if len(plan[name]) > 10 else ""))
    if not plan["all"]:
        for test, why in plan["reasons"].items():
            print(f"  {test}: {', '.join(why[:3])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
