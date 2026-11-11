"""MIG variant releases: "same science + MIG allowlist" commits of the releases that pinned runs use (ARCH Phase 0).

A run pinned to a pre-MIG release cannot run on a MIG instance: that release's in-container gpu_guard authorizes only
the physical GPUs. Its MIG variant is the base release plus ONLY the MIG change of the container-side operational files
(VARIANT_FILES: the GPU allowlist, its loader, the EGL selection, runtime provenance, the isolation check) and the
rendered allowlist (configs/hosts/remote-mig.yaml + remote.yaml). The variant must then
1. differ from its base, inside every campaign script's execution scope, only in release_repin.REPIN_OPERATIONAL
   (checked here with release_repin.scope_comparison, third-party pins included);
2. become an ancestor of pushed origin/main (run_identity requires a pushed main commit): merge it into main's history
   with a merge that keeps main's tree (`git merge --no-ff -s ours <variant>`), then push;
3. pass the release chain on MIG (native suite on one instance + one isolation check per instance) like any release.

    python -I ops/jobs/mig_variant.py create --base <commit> --mig-base 8299442 --mig-head <MIG commit> \
        --configs <dir with configs/hosts/remote.yaml + remote-mig.yaml> --script scripts/train_snerl.py [...]
    (prints the variant sha and the merge/release commands; nothing is pushed, merged or released)
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ops.jobs.migration import commit_reader  # noqa: E402
from ops.jobs.release_repin import scope_comparison  # noqa: E402

VARIANT_FILES = ("ops/resources/gpu_guard.py", "ops/resources/mig_inventory.py", "mujoco_mig_setup.py",
                 "ops/jobs/run_identity.py", "scripts/check_gpu_isolation.py")
CONFIG_FILES = ("configs/hosts/remote.yaml", "configs/hosts/remote-mig.yaml")


class VariantError(RuntimeError):
    pass


def git(root: Path, *args: str, stdin: bytes | None = None) -> bytes:
    result = subprocess.run(["git", "-C", str(root), *args], input=stdin, capture_output=True)
    if result.returncode:
        raise VariantError(f"git {' '.join(args[:3])} failed: {result.stderr.decode()[-600:]}")
    return result.stdout


def mig_patch(root: Path, mig_base: str, mig_head: str, files=VARIANT_FILES) -> bytes:
    """The MIG change of the container-side files only (binary-safe, full index for a 3-way apply)."""
    return git(root, "diff", "--binary", "--full-index", mig_base, mig_head, "--", *files)


def create(root: Path, base: str, mig_base: str, mig_head: str, configs: Path, scripts: list[str],
           author=("Claude", "noreply@anthropic.com")) -> dict:
    """Build the variant commit (detached; nothing pushed) and verify its execution scope against ``base``."""
    base = git(root, "rev-parse", f"{base}^{{commit}}").decode().strip()
    patch = mig_patch(root, mig_base, mig_head)
    if not patch.strip():
        raise VariantError("the MIG range changes none of the variant files")
    for name in CONFIG_FILES:
        if not (configs / name).is_file():
            raise VariantError(f"missing rendered {name} (ops/resources/mig_inventory.py render)")
    work = Path(tempfile.mkdtemp(prefix="mig-variant-"))
    tree = work / "tree"
    try:
        git(root, "worktree", "add", "--detach", str(tree), base)
        try:
            git(tree, "apply", "--3way", "--index", "-", stdin=patch)
        except VariantError as exc:
            raise VariantError(f"the MIG patch does not apply cleanly on {base[:7]}: {exc}") from None
        conflicted = git(tree, "diff", "--name-only", "--diff-filter=U").decode().split()
        if conflicted:
            raise VariantError(f"conflicts on {base[:7]}: {conflicted}")
        for name in CONFIG_FILES:
            (tree / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(configs / name, tree / name)
            git(tree, "add", name)
        message = (f"MIG variant of {base[:7]}: the remote MIG allowlist and its container-side consumers only\n\n"
                   f"Applied: git diff {mig_base[:7]}..{mig_head[:7]} -- {' '.join(VARIANT_FILES)}; rendered "
                   f"{' + '.join(CONFIG_FILES)}. Science, configs and data paths are those of {base[:7]}.")
        git(tree, "-c", f"user.name={author[0]}", "-c", f"user.email={author[1]}", "commit", "-q", "-m", message)
        variant = git(tree, "rev-parse", "HEAD").decode().strip()
    finally:
        subprocess.run(["git", "-C", str(root), "worktree", "remove", "--force", str(tree)], capture_output=True)
        shutil.rmtree(work, ignore_errors=True)
    before_read, before_listed = commit_reader(root, base)
    after_read, after_listed = commit_reader(root, variant)
    from ops.provenance.upstreams import commit_identity  # noqa: PLC0415

    scope = scope_comparison([*scripts, "scripts/remote/entry.py"], before_read, before_listed, after_read,
                             after_listed, commit_identity(root, base), commit_identity(root, variant))
    changed = git(root, "diff", "--name-only", base, variant).decode().split()
    return {"base": base, "variant": variant, "identical_scope": scope["identical"],
            "scope_differences": scope["differences"], "operational_differences": scope["operational_differences"],
            "changed_files": changed,
            "next": [f"git -C {root} checkout main && git -C {root} merge --no-ff -s ours {variant} "
                     f"-m 'Adopt MIG variant {variant[:7]} of {base[:7]} into main history (tree unchanged)'",
                     f"git -C {root} push origin main   (credentials: gh auth git-credential)",
                     f"python -I runs/setup/tools/release_chain.py {variant} --image <pinned digest> --tag cc "
                     "--native-gpu <one MIG UUID>   (after the verification HOLD is lifted for the MIG chains)"]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("create", "patch"))
    parser.add_argument("--base", required=False)
    parser.add_argument("--mig-base", required=True)
    parser.add_argument("--mig-head", required=True)
    parser.add_argument("--configs", type=Path)
    parser.add_argument("--script", action="append", default=[])
    args = parser.parse_args(argv)
    if args.command == "patch":
        sys.stdout.buffer.write(mig_patch(REPO, args.mig_base, args.mig_head))
        return 0
    if not args.base or not args.configs or not args.script:
        raise SystemExit("create needs --base, --configs and at least one --script")
    result = create(REPO, args.base, args.mig_base, args.mig_head, args.configs, args.script)
    print(json.dumps(result, indent=1))
    return 0 if result["identical_scope"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
