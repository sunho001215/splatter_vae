"""Remote cleanup (user directive 2026-11-11 ~17:30) under the authorized root /home/compu/kaist/sunho: dated copy of
disk_cleanup_remote_20261025.py, tightened as in the 2026-11-06 CLEAN pass and disk_cleanup_common_20261111.py.
Remote access only through ops.remote.access.RemoteClient (no scheduler import). Replay is NOT handled here
(scripts/cleanup_replay.py, background loop). /data_8T, realworld/, data/, releases/, W&B, other users' files and
containers are never touched.

Categories (exact paths, no symlinks, realpath inside the root):
  policy_snapshot  runs/R/snapshots/step_N.pt of a completed, evaluated, closed-out RL run R with N < budget, N not a
                   100k multiple, N in the (synced) eval history, the path / (R, N) not referenced; the final snapshot
                   and checkpoints/latest.pt are kept;
  checkpoint       runs/D/checkpoints/step_N.pt of a completed remote pretraining run D with N not the final step, not a
                   100k multiple, unreferenced, eval/step_N/summary.json synced locally; latest.pt is kept;
  verification_pytest_tmp
                   runtime/verification/verify-native-*/a<k>/evidence/tmp: the pytest --basetemp scratch of a finished
                   release-suite run (scripts/run_tests.py S4D_PYTEST_TMP=<evidence>/tmp). The evidence proper
                   (tests.json, tests.junit.xml, tests.log, pytest-cache, cache) is kept, and campaign acceptance reads
                   only those reports (ops/jobs/remote_backend.py acceptance_manifest, local copies). Kept entirely:
                   releases pinned by any job that is not done/failed (running, crashed/resumable, pending with a pin,
                   unknown), the adopted release, verification jobs not done/failed, trees modified within 24 h.
Run eligibility (RL and pretraining): not protected (common.PROTECT_PREFIXES / SUBSTRINGS, named under realworld-ops,
active admission pause); no live container (created/running/restarting/paused/removing) labelled with the run or its
evaluator; registry done, exit 0, not unknown, host remote with a verified final sync; the remote final.json equals the
verified sync (RL); final.json completed at the configured budget; evaluator done with a verified sync; eval history
ends at the budget with one row per interval; final snapshot and latest.pt present remotely; every other queue row
that references or depends on the run is done. Pretraining: completion.json (synced) completed with steps == max
remote step == configured train.steps.
Execution: one remote python3 (stdlib) call per batch re-checks every item on the host right before deleting (pattern,
regular file / real directory, realpath identity, no live container for the run, no process holding a file inside,
size/mtime unchanged, kept final present) and prints one JSON line per item; the authoritative result is a final
existence check. Paced (<= 200 files per call, 1 s apart; one verification tree per call, 3 s apart), nice/ionice.

    nice -n 19 .venv/bin/python -I runs/setup/tools/disk_cleanup_remote_20261111.py            # manifest only
    nice -n 19 .venv/bin/python -I runs/setup/tools/disk_cleanup_remote_20261111.py --execute [--category X]
        [--allow-user-manager]   # exempt only the non-dumpable `systemd --user` of the remote user from the
                                 # fail-closed own-process inspection (it blocked every tree on 2026-11-11)
"""
import base64
import json
import re
import shlex
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import disk_cleanup_common_20261111 as C  # noqa: E402

sys.path.insert(0, str(C.ROOT))
from ops.remote.access import RemoteClient  # noqa: E402

ROOT, REMOTE = C.ROOT, C.REMOTE_ROOT
EXECUTE = '--execute' in sys.argv
ONLY = sys.argv[sys.argv.index('--category') + 1] if '--category' in sys.argv else None
STEP = re.compile(r'^step_(\d{7})\.pt$')
VTMP = re.compile(r'^(verify-native-[0-9a-f]{7}[A-Za-z0-9-]*)/a(\d+)/evidence/tmp$')
MANIFEST = C.OUT / 'remote-models-manifest.json'
RECENT = 24 * 3600
LIVE = ('created', 'restarting', 'running', 'paused', 'removing')
CLIENT = RemoteClient()


def ssh(command, timeout=900, idempotent=True):
    r = CLIENT.ssh(command, wait=True, idempotent=idempotent, timeout=timeout)
    if r.returncode:
        raise RuntimeError(f'remote command exited {r.returncode}: {r.stderr[-500:]}')
    return r.stdout


LIVE_FILTER = ' '.join(f'--filter status={s}' for s in LIVE)
REMOTE_STATE = (
    f"set -eu; export LC_ALL=C; cd {REMOTE}/runs; df -B1 {REMOTE} | tail -1; echo __LIVE__; "
    f"docker ps -a {LIVE_FILTER} --format '{{{{.Label \"s4d.run_id\"}}}}|{{{{.Names}}}}'; echo __FILES__; "
    "nice -n 19 ionice -c3 find . -mindepth 3 -maxdepth 3 \\( -path './*/snapshots/step_*.pt' "
    "-o -path './*/checkpoints/step_*.pt' \\) -type f -printf '%k %s %T@ %U %p\\n'; echo __VTMP__; "
    f"cd {REMOTE}/runtime/verification; for t in */a*/evidence/tmp; do [ -d \"$t\" ] && [ ! -L \"$t\" ] || continue; "
    "printf '%s %s %s\\n' \"$(nice -n 19 ionice -c3 du -sk -- \"$t\" | cut -f1)\" "
    "\"$(nice -n 19 ionice -c3 find \"$t\" -printf '%T@\\n' | sort -n | tail -1)\" \"$t\"; done; echo __END__")


def remote_state():
    out = ssh(REMOTE_STATE, timeout=3000)
    assert out.rstrip().endswith('__END__'), 'incomplete remote state'
    head, rest = out.split('__LIVE__\n', 1)
    live, rest = rest.split('__FILES__\n', 1)
    listing, rest = rest.split('__VTMP__\n', 1)
    vtmp = rest.split('__END__')[0]
    files = []
    for line in listing.splitlines():
        kb, size, mtime, uid, path = line.split(' ', 4)
        rel = path[2:]
        files.append({'path': f'runs/{rel}', 'run': rel.split('/')[0], 'sub': rel.split('/')[1],
                      'name': rel.split('/')[2], 'bytes': int(kb) * 1024, 'size': int(size), 'mtime': float(mtime),
                      'owner_uid': int(uid)})
    trees = []
    for line in vtmp.splitlines():
        kb, newest, rel = line.split(' ', 2)
        trees.append({'rel': rel, 'bytes': int(kb) * 1024, 'newest_mtime': float(newest or 0)})
    live_runs = {x.split('|')[0] for x in live.split() if x.split('|')[0]}
    live_names = {x.split('|', 1)[1] for x in live.split() if '|' in x}
    return int(head.split()[3]), live_runs, live_names, files, trees


def eval_rows(run):
    for path in (ROOT / 'runs' / f'{run}-eval' / 'eval.jsonl', ROOT / 'runs' / run / 'eval.jsonl'):
        if path.is_file():
            return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return None


def verified(s):
    return bool(s.get('sync', {}).get('final') and s.get('sync', {}).get('verified'))


def common_check(run, state, rows, pauses, rw_runs, live):
    why = C.protected_prefix(run)
    if why:
        return f'protected: {why}'
    if run in rw_runs:
        return f'protected: named in realworld-ops ({rw_runs[run]})'
    p = C.paused(run, pauses)
    if p:
        return f'admission pause active: {p[0]}'
    if run in live or f'{run}-eval' in live:
        return 'live container'
    s = state.get(run)
    if s is None:
        return 'not in registry'
    if s['status'] != 'done' or s.get('unknown') or s.get('code') not in (0, None) or s.get('host') != 'remote':
        return f"registry status {s['status']} code {s.get('code')} host {s.get('host')}"
    if not verified(s):
        return 'final sync not verified'
    deps = [r['id'] for r in rows if (run in r.get('deps', []) or f'{run}-eval' in r.get('deps', []))
            and state.get(r['id'], {}).get('status') != 'done']
    if deps:
        return f'non-done dependent {deps[0]}'
    refs = [i for i in C.referencing_rows(run, rows) if state.get(i, {}).get('status') != 'done']
    if refs:
        return f'non-done referencing row {refs[0]}'
    return None


def rl_check(run, state, rows, byid, pauses, rw_runs, live, facts):
    why = common_check(run, state, rows, pauses, rw_runs, live)
    if why:
        return why, None
    s = state[run]
    if facts.get('final_sha') != s.get('sync', {}).get('files', {}).get('final.json'):
        return 'remote final.json differs from verified sync', None
    final = json.loads((ROOT / 'runs' / run / 'final.json').read_text())
    cfg = yaml.safe_load((ROOT / 'runs' / run / 'config.yaml').read_text())
    args = [str(a) for a in (byid.get(run) or {}).get('args', [])]
    queued = next((int(a.split('=', 1)[1]) for a in args if a.startswith('train.num_train_steps=')), None)
    configured = cfg.get('train', {}).get('num_train_steps')
    if queued is not None and configured != queued:
        return 'config budget differs from queue override', None
    total = queued if queued is not None else configured
    if final.get('completed') is not True or total is None or final.get('steps') != total:
        return f'budget not verified (final {final.get("steps")}, configured {total})', None
    every = int(cfg['eval']['every_steps'])
    ev = f'{run}-eval'
    if ev in byid or ev in state:
        es = state.get(ev, {})
        if es.get('status') != 'done' or not verified(es):
            return 'evaluator not closed out', None
    erows = eval_rows(run)
    if not erows or erows[-1]['step'] != total or len(erows) != total // every:
        return 'evaluation history incomplete', None
    if run.startswith('s2rl-') and not (ROOT / 'runs/setup' / f's2-{run}-completion.json').is_file():
        return 'Stage 2 closeout proof missing', None
    if not facts.get('final_snapshot') or not facts.get('latest_checkpoint'):
        return 'final snapshot or last checkpoint missing remotely', None
    return None, (int(total), {r['step'] for r in erows})


def pretrain_check(run, steps, state, rows, pauses, rw_runs, live):
    why = common_check(run, state, rows, pauses, rw_runs, live)
    if why:
        return why, None
    comp, cfg = ROOT / 'runs' / run / 'completion.json', ROOT / 'runs' / run / 'config.yaml'
    if not comp.is_file() or not cfg.is_file():
        return 'no synced completion.json / config.yaml', None
    c = json.loads(comp.read_text())
    cfg_steps = yaml.safe_load(cfg.read_text()).get('train', {}).get('steps')
    if c.get('status') != 'completed' or c.get('steps') != max(steps) or cfg_steps != max(steps):
        return f"completion not verified ({c.get('steps')}, max {max(steps)}, config {cfg_steps})", None
    return None, max(steps)


def release_pins(state, rows):
    """See disk_cleanup_common_20261111.release_pins (Codex review 2026-11-11 finding 3)."""
    return C.release_pins(state, rows)


def plan():
    free, live, live_names, files, trees = remote_state()
    state = C.read_registry()
    rows = C.load_queue_rows()
    byid = {r['id']: r for r in rows}
    pauses = C.active_pauses()
    rw_runs = C.realworld_ops_runs()
    by_run = {}
    for f in files:
        by_run.setdefault(f['run'], []).append(f)
    rl_runs = sorted(r for r, fs in by_run.items() if any(f['sub'] == 'snapshots' for f in fs))
    pre_runs = sorted(r for r in by_run if r not in rl_runs)
    # cheap prefilter before the per-run remote probe
    pre_verdict = {r: common_check(r, state, rows, pauses, rw_runs, live) for r in rl_runs}
    probe_runs = [r for r in rl_runs if pre_verdict[r] is None]
    facts = {}
    for i in range(0, len(probe_runs), 60):
        cmds = []
        for run in probe_runs[i:i + 60]:
            cfg_path = ROOT / 'runs' / run / 'config.yaml'
            total = (yaml.safe_load(cfg_path.read_text()) or {}).get('train', {}).get('num_train_steps') \
                if cfg_path.is_file() else 0
            q = shlex.quote(run)
            cmds.append(f"echo __RUN__ {q}; sha256sum {q}/final.json 2>/dev/null | cut -c1-64; "
                        f"test -f {q}/snapshots/step_{int(total or 0):07d}.pt && echo SNAP; "
                        f"test -f {q}/checkpoints/latest.pt && echo LATEST; true")
        out = ssh(f"cd {REMOTE}/runs && " + ' ; '.join(cmds))
        for block in out.split('__RUN__ ')[1:]:
            lines = block.splitlines()
            facts[lines[0].strip()] = {'final_sha': next((ln for ln in lines[1:] if len(ln) == 64), None),
                                       'final_snapshot': 'SNAP' in lines, 'latest_checkpoint': 'LATEST' in lines}
        time.sleep(1)
    verdicts, prelim = {}, {}
    for run in rl_runs:
        try:
            verdicts[run], prelim[run] = rl_check(run, state, rows, byid, pauses, rw_runs, live, facts.get(run, {}))
        except (OSError, KeyError, ValueError, TypeError) as exc:
            verdicts[run], prelim[run] = f'check error {type(exc).__name__}', None
    for run in pre_runs:
        steps = [int(STEP.match(f['name']).group(1)) for f in by_run[run]]
        try:
            verdicts[run], prelim[run] = pretrain_check(run, steps, state, rows, pauses, rw_runs, live)
        except (OSError, KeyError, ValueError, TypeError) as exc:
            verdicts[run], prelim[run] = f'check error {type(exc).__name__}', None
    eligible = {r for r, v in verdicts.items() if v is None}
    refs = C.reference_scan(eligible)
    cands, kept = [], {}

    def keep(reason, f):
        k = kept.setdefault(reason, {'files': 0, 'bytes': 0, 'examples': []})
        k['files'] += 1
        k['bytes'] += f['bytes']
        if len(k['examples']) < 4:
            k['examples'].append(f['path'])

    now = time.time()
    for run in rl_runs:
        if verdicts[run]:
            for f in by_run[run]:
                keep(f"RL run not eligible: {verdicts[run].split(' (')[0][:90]}", f)
            continue
        total, steps = prelim[run]
        for f in by_run[run]:
            m = STEP.match(f['name'])
            n = int(m.group(1))
            if f['sub'] != 'snapshots':
                keep('checkpoint file of an RL run', f)
            elif n >= total:
                keep('final RL policy snapshot', f)
            elif n % 100000 == 0:
                keep('100k-multiple milestone snapshot (kept as on 2026-11-06)', f)
            elif n not in steps:
                keep('snapshot step not in eval history', f)
            elif (r := C.referenced(f['path'], run, n, refs)):
                keep(f'referenced in campaign records ({r})', f)
            elif now - f['mtime'] < 3600:
                keep('modified within 1 h', f)
            else:
                cands.append({**f, 'path': f"{REMOTE}/{f['path']}", 'rel': f['path'], 'category': 'policy_snapshot',
                              'reason': f'intermediate RL policy snapshot of a completed, evaluated run (step {n} '
                                        f'evaluated; final step_{total:07d}.pt and checkpoints/latest.pt kept)',
                              'keep': f"runs/{run}/snapshots/step_{total:07d}.pt",
                              'evidence': {'registry': 'done', 'host': 'remote', 'final_sync_verified': True,
                                           'remote_final_json_equals_sync': True, 'budget': total,
                                           'eval_history_max': max(steps), 'live_container': False,
                                           'referenced': False}})
    for run in pre_runs:
        for f in by_run[run]:
            n = int(STEP.match(f['name']).group(1))
            if verdicts[run]:
                keep(f"pretraining run not eligible: {verdicts[run].split(' (')[0][:90]}", f)
            elif n == prelim[run]:
                keep('final pretrained model', f)
            elif n % 100000 == 0:
                keep('100k-multiple milestone (reported endpoints)', f)
            elif (r := C.referenced(f['path'], run, n, refs)):
                keep(f'referenced in campaign records ({r})', f)
            elif not (ROOT / 'runs' / run / 'eval' / f'step_{n:07d}' / 'summary.json').is_file():
                keep('no synced evaluation for this step', f)
            elif now - f['mtime'] < 3600:
                keep('modified within 1 h', f)
            else:
                cands.append({**f, 'path': f"{REMOTE}/{f['path']}", 'rel': f['path'], 'category': 'checkpoint',
                              'reason': f'intermediate pretraining checkpoint of a completed run (step {n}: not final, '
                                        f'not a 100k multiple, evaluation synced; final step_{prelim[run]:07d}.pt kept)',
                              'keep': f"runs/{run}/checkpoints/step_{prelim[run]:07d}.pt",
                              'evidence': {'registry': 'done', 'host': 'remote', 'final_sync_verified': True,
                                           'completion_steps': prelim[run], 'eval_summary_synced': True,
                                           'live_container': False, 'referenced': False}})
    pins = release_pins(state, rows)
    vkept = []
    for t in trees:
        m = VTMP.match(t['rel'])
        job, attempt = (m.group(1), int(m.group(2))) if m else (None, None)
        s = state.get(job or '', {})
        commit7 = re.match(r'verify-native-([0-9a-f]{7})', job).group(1) if job else None
        why = None
        if not m:
            why = 'name pattern'
        elif not s:
            why = 'verification job not in registry'
        elif s['status'] != 'done' or s.get('unknown'):  # failed suites' scratch kept (Codex review finding 1)
            why = f"verification job {s['status']}{' unknown' if s.get('unknown') else ''}"
        elif commit7 in pins:
            why = f'release {commit7} pinned ({pins[commit7]})'
        elif now - t['newest_mtime'] < RECENT:
            why = 'modified within 24 h'
        elif f's4d-{job}-a{attempt}' in live_names:
            why = 'verification container live'
        if why:
            vkept.append({**t, 'kept': why})
            k = kept.setdefault(f'verification tmp kept: {why.split(" (")[0]}', {'files': 0, 'bytes': 0, 'examples': []})
            k['files'] += 1
            k['bytes'] += t['bytes']
            continue
        cands.append({'path': f"{REMOTE}/runtime/verification/{t['rel']}", 'rel': f"runtime/verification/{t['rel']}",
                      'run': job, 'category': 'verification_pytest_tmp', 'bytes': t['bytes'], 'owner_uid': 1000,
                      'newest_mtime': t['newest_mtime'],
                      'reason': 'pytest --basetemp scratch of a finished release-suite run; evidence reports kept',
                      'evidence': {'verification_job': job, 'attempt': attempt, 'registry': s['status'],
                                   'release': commit7, 'pinned': False,
                                   'age_hours': round((now - t['newest_mtime']) / 3600, 1),
                                   'kept_alongside': ['tests.json', 'tests.junit.xml', 'tests.log', 'pytest-cache']}})
    by_cat = {}
    for c in cands:
        by_cat.setdefault(c['category'], {'files': 0, 'bytes': 0, 'runs': set()})
        by_cat[c['category']]['files'] += 1
        by_cat[c['category']]['bytes'] += c['bytes']
        by_cat[c['category']]['runs'].add(c['run'])
    return {'recorded_local': C.now_iso(), 'host': 'remote', 'root': REMOTE, 'tool': __file__,
            'request': 'user directive 2026-11-11 ~17:30 (disk cleanup, both hosts)', 'free_bytes_before': free,
            'live_containers': sorted(live_names), 'release_pins': pins,
            'reference_scan': {'files_scanned': refs['files_scanned'], 'paths': len(refs['paths']),
                               'pairs': len(refs['pairs'])},
            'candidates_by_category': {k: {'files': v['files'], 'bytes': v['bytes'], 'runs': sorted(v['runs'])}
                                       for k, v in by_cat.items()},
            'kept_by_reason': kept, 'verification_tmp_kept': vkept,
            'runs_not_eligible': {r: v for r, v in verdicts.items() if v}, 'candidates': cands}


# ---------------------------------------------------------------------------------------------------------------
# remote executor (python3 stdlib, runs ON the remote host; argv[1] = base64 JSON request). Hardened after the Codex
# review 2026-11-11 (findings 2, 4, 7): live containers and process holders are re-read right before EACH item; a tree
# is walked fail-closed (any walk error refuses), must stay on one filesystem (no mount point inside), be owned
# entirely by this user, match the manifest's newest mtime (root included) and allocated size (+-1 %); rmtree errors
# are reported as "partial".
EXECUTOR = r'''
import base64, json, os, re, shutil, subprocess, sys, time
ROOT = "/home/compu/kaist/sunho"
req = json.loads(base64.b64decode(sys.argv[1]).decode())
LIVE = ("created", "restarting", "running", "paused", "removing")
FILE = re.compile(r"runs/([A-Za-z0-9][A-Za-z0-9._-]*)/(snapshots|checkpoints)/step_\d{7}\.pt\Z")
TREE = re.compile(r"runtime/verification/(verify-native-[0-9a-f]{7}[A-Za-z0-9-]*)/a(\d+)/evidence/tmp\Z")
UID = os.getuid()
def live_containers():
    cmd = ["docker", "ps", "-a", "--format", '{{.Label "s4d.run_id"}}|{{.Names}}']
    for s in LIVE:
        cmd += ["--filter", "status=" + s]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=120)
    if r.returncode:
        raise RuntimeError("docker ps failed")
    runs, names = set(), set()
    for line in r.stdout.split():
        a, _, b = line.partition("|")
        if a: runs.add(a)
        if b: names.add(b)
    return runs, names
def holders(prefix):
    # processes of this user must be fully inspectable (fail closed); other users' processes cannot be inspected
    # without root and are counted (reported) instead
    hits, others = [], 0
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            own = os.stat("/proc/%s" % pid).st_uid == UID
        except OSError:
            continue
        try:
            cwd = os.readlink("/proc/%s/cwd" % pid)
            if cwd == prefix or cwd.startswith(prefix + "/"):
                hits.append((pid, cwd))
            for fd in os.listdir("/proc/%s/fd" % pid):
                try:
                    t = os.readlink("/proc/%s/fd/%s" % (pid, fd))
                except OSError:
                    continue
                if t == prefix or t.startswith(prefix + "/"):
                    hits.append((pid, t))
        except PermissionError:
            if own:
                # the per-user service manager is non-dumpable (fds/cwd unreadable) and never works in campaign
                # scratch; exempt ONLY with the explicit opt-in (2026-11-11: refused all 15 trees by default)
                try:
                    argv = open("/proc/%s/cmdline" % pid, "rb").read().split(b"\0")
                except OSError:
                    argv = []
                if req.get("allow_user_manager") and argv[:2] in ([b"/usr/lib/systemd/systemd", b"--user"],
                                                                  [b"/lib/systemd/systemd", b"--user"]):
                    others += 1
                    continue
                raise RuntimeError("cannot inspect own process %s" % pid)
            others += 1
        except OSError:
            continue
    return hits, others
def mount_inside(path):
    with open("/proc/self/mountinfo") as fh:
        for line in fh:
            mp = re.sub(r"\\([0-7]{3})", lambda m: chr(int(m.group(1), 8)), line.split()[4])
            if mp == path or mp.startswith(path + "/"):
                return mp
    return None
def scan(path):
    root = os.lstat(path)
    newest, total = root.st_mtime, root.st_blocks * 512
    if root.st_uid != UID:
        raise ValueError("tree root not owned by this user")
    if mount_inside(path):
        raise ValueError("mount point at or inside the tree")
    def fail(err):
        raise err
    for d, dirs, files in os.walk(path, onerror=fail, followlinks=False):
        for f in files + dirs:
            st = os.lstat(os.path.join(d, f))
            if st.st_dev != root.st_dev:
                raise ValueError("mount point inside the tree: " + os.path.join(d, f))
            if st.st_uid != UID:
                raise ValueError("entry not owned by this user: " + os.path.join(d, f))
            newest = max(newest, st.st_mtime)
            total += st.st_blocks * 512
    return newest, total
for it in req["items"]:
    rel, out = it["rel"], {"rel": it["rel"], "status": "refused"}
    full = os.path.join(ROOT, rel)
    try:
        if ".." in rel.split("/") or "//" in rel:
            raise ValueError("traversal")
        m = FILE.match(rel)
        t = TREE.match(rel)
        live_runs, live_names = live_containers()
        if m:
            run = m.group(1)
            if run in live_runs or run + "-eval" in live_runs:
                raise ValueError("live container for the run")
            if os.path.islink(full) or not os.path.isfile(full) or os.path.realpath(full) != full:
                raise ValueError("not a regular file")
            st = os.lstat(full)
            if st.st_uid != UID:
                raise ValueError("not owned by this user")
            if st.st_size != it["size"] or abs(st.st_mtime - it["mtime"]) > 1e-3:
                raise ValueError("size or mtime changed")
            keep = os.path.join(ROOT, it["keep"])
            if os.path.islink(keep) or not os.path.isfile(keep):
                raise ValueError("kept final file missing")
            busy, others = holders(full)
            out["uninspectable_foreign_processes"] = others
            if busy:
                raise ValueError("open by a process")
            if not req["dry_run"]:
                os.unlink(full)
            out.update(status="would_delete" if req["dry_run"] else "deleted", allocated=st.st_blocks * 512)
        elif t:
            job, attempt = t.group(1), t.group(2)
            if "s4d-%s-a%s" % (job, attempt) in live_names or job in live_runs:
                raise ValueError("verification container live")
            if os.path.islink(full) or not os.path.isdir(full) or os.path.realpath(full) != full:
                raise ValueError("not a real directory")
            ev = os.path.dirname(full)
            if os.lstat(ev).st_dev != os.lstat(full).st_dev:
                raise ValueError("tmp is a mount point")
            out["evidence_kept"] = sorted(f for f in os.listdir(ev) if f != "tmp")
            for need in ("tests.json", "tests.junit.xml", "tests.log"):
                if need not in out["evidence_kept"]:
                    raise ValueError(need + " missing next to tmp")
            latest, total = scan(full)
            out.update(newest_mtime=latest, allocated=total)
            if latest > it["newest_mtime"] + 1e-3 or time.time() - latest < 24 * 3600:
                raise ValueError("tree modified since the manifest / within 24 h")
            if abs(total - it["bytes"]) > 0.01 * it["bytes"] + 1048576:
                raise ValueError("allocated size differs from the manifest (%d vs %d)" % (total, it["bytes"]))
            busy, others = holders(full)
            out["uninspectable_foreign_processes"] = others
            if busy:
                raise ValueError("in use by a process: %s" % busy[:3])
            if not req["dry_run"]:
                errors = []
                shutil.rmtree(full, onerror=lambda fn, p, exc: errors.append("%s %s" % (p, exc[1])))
                gone = not os.path.lexists(full)
                out.update(status="deleted" if gone and not errors else "partial", gone=gone, errors=errors[:5])
            else:
                out["status"] = "would_delete"
        else:
            raise ValueError("path pattern")
    except Exception as exc:
        out["reason"] = ("%s: %s" % (type(exc).__name__, exc))[:300]
    print(json.dumps(out), flush=True)
print("__DONE__", flush=True)
'''
MAX_REQUEST = 60000  # bytes of base64 request per call (well below the 128 KiB per-argument limit)


def run_executor(items, dry_run, timeout=1800):
    req = base64.b64encode(json.dumps({'dry_run': dry_run, 'allow_user_manager': '--allow-user-manager' in sys.argv,
                                       'items': [
        {k: it[k] for k in ('rel', 'size', 'mtime', 'keep', 'newest_mtime', 'bytes') if k in it}
        for it in items]}).encode()).decode()
    assert len(req) <= MAX_REQUEST, 'request too large'
    script = base64.b64encode(EXECUTOR.encode()).decode()
    cmd = (f"cd {REMOTE} && nice -n 19 ionice -c3 python3 -I -c \"$(printf %s {script} | base64 -d)\" {req}")
    try:
        out = ssh(cmd, timeout=timeout, idempotent=dry_run)
    except Exception as exc:  # outcome unknown: reconciled by the existence check
        return [], f'{type(exc).__name__}: {exc}'[:300]
    lines = [json.loads(x) for x in out.splitlines() if x.startswith('{')]
    return lines, None if '__DONE__' in out else 'executor output incomplete'


def batches(items):
    # by encoded size (Codex review finding 8): items are added while the request stays below MAX_REQUEST
    cur = []
    for it in items:
        trial = cur + [it]
        size = len(base64.b64encode(json.dumps({'dry_run': False, 'items': [
            {k: x[k] for k in ('rel', 'size', 'mtime', 'keep', 'newest_mtime', 'bytes') if k in x}
            for x in trial]}).encode()))
        if cur and size > MAX_REQUEST:
            yield cur
            cur = [it]
        else:
            cur = trial
    if cur:
        yield cur


def exists_check(rels):
    present = set()
    for i in range(0, len(rels), 300):
        listing = '\n'.join(rels[i:i + 300])
        out = ssh(f"cd {REMOTE} || exit 9; while IFS= read -r p; do if [ -e \"$p\" ] || [ -L \"$p\" ]; then "
                  f"echo PRESENT \"$p\"; fi; done <<'S4DLIST'\n{listing}\nS4DLIST\necho __END__")
        if not out.rstrip().endswith('__END__'):
            raise RuntimeError('existence check incomplete')
        present |= {line[8:] for line in out.splitlines() if line.startswith('PRESENT ')}
        time.sleep(1)
    return present


def execute(m, dry_run=False):
    rw_runs = C.realworld_ops_runs()
    log = (C.OUT / 'executed-actions.jsonl').open('a')
    results = {}
    cats = [ONLY] if ONLY else ['verification_pytest_tmp', 'checkpoint', 'policy_snapshot']
    fresh = None
    for category in cats:
        batch = [c for c in m['candidates'] if c['category'] == category]
        if batch and category != 'verification_pytest_tmp' and fresh is None:
            # model files: the reviewed manifest intersected with a fresh plan incl. a fresh reference scan (Codex
            # review finding 5); anything the fresh plan no longer selects is skipped
            fresh = {c['rel'] for c in plan()['candidates']}
        skipped, todo = [], []
        for c in batch:
            if category != 'verification_pytest_tmp' and c['rel'] not in fresh:
                skipped.append((c, 'not selected by the fresh plan'))
            else:
                todo.append(c)
        reports, errors, called = {}, [], []
        if category == 'verification_pytest_tmp':
            for c in todo:  # one tree per call; controller state refreshed right before each
                if errors:  # remote reboot announced 2026-11-11: stop at the first failed call, defer the rest
                    skipped.append((c, 'deferred: an earlier call failed (remote unreachable / reboot?)'))
                    continue
                try:
                    CLIENT.check_tunnel(wait=False)
                except Exception as exc:
                    errors.append(f'unreachable before call: {exc}'[:200])
                    skipped.append((c, 'deferred: remote unreachable'))
                    continue
                state, rows = C.read_registry(), C.load_queue_rows()
                pins = release_pins(state, rows)
                s = state.get(c['run'], {})
                if s.get('status') != 'done' or s.get('unknown') or c['evidence']['release'] in pins:
                    skipped.append((c, f"controller recheck: status {s.get('status')}, pinned "
                                       f"{c['evidence']['release'] in pins}"))
                    continue
                time.sleep(3)
                called.append(c)
                lines, err = run_executor([c], dry_run)
                if err:
                    errors.append(err)
                for line in lines:
                    reports[line['rel']] = line
        else:
            state, rows, pauses = C.read_registry(), C.load_queue_rows(), C.active_pauses()
            for group in batches(todo):
                runs = {c['run'] for c in group}
                bad = {r for r in runs if common_check(r, state, rows, pauses, rw_runs, set()) is not None}
                for c in [c for c in group if c['run'] in bad]:
                    skipped.append((c, 'controller recheck failed'))
                group = [c for c in group if c['run'] not in bad]
                if not group:
                    continue
                time.sleep(1)
                called.extend(group)
                lines, err = run_executor(group, dry_run)
                if err:
                    errors.append(err)
                for line in lines:
                    reports[line['rel']] = line
                state, rows, pauses = C.read_registry(), C.load_queue_rows(), C.active_pauses()
        for c, why in skipped:
            log.write(json.dumps({'time': time.time(), 'host': 'remote', 'action': 'skip', 'category': category,
                                  'path': c['path'], 'reason': why}) + '\n')
        sent = called
        try:
            present = exists_check([c['rel'] for c in sent]) if (sent and not dry_run) else {c['rel'] for c in sent}
            unverified = False
        except Exception as exc:  # e.g. host rebooting: outcome reconciled later; nothing is claimed as deleted
            errors.append(f'existence check failed: {exc}'[:200])
            present, unverified = {c['rel'] for c in sent}, True
        done_b, done_n = 0, 0
        for c in sent:
            rep = reports.get(c['rel'], {})
            c['deleted'] = (not dry_run) and c['rel'] not in present
            if c['deleted']:
                done_b += c['bytes']
                done_n += 1
            action = 'dry_run' if dry_run else ('unverified' if unverified else 'delete' if c['deleted'] else
                                                 ('partial' if rep.get('status') == 'partial' else 'skip'))
            log.write(json.dumps({'time': time.time(), 'host': 'remote', 'action': action,
                                  'category': category, 'path': c['path'], 'allocated_bytes': c['bytes'],
                                  'owner_uid': c.get('owner_uid'), 'reason': c['reason'], 'evidence': c['evidence'],
                                  'remote_recheck': rep}) + '\n')
        log.flush()
        results[category] = {'files': done_n, 'bytes': done_b, 'of': len(batch),
                             'controller_skipped': [(c['rel'], why) for c, why in skipped], 'errors': errors[:10],
                             'refused': [r for r in reports.values() if r.get('status') == 'refused'][:50],
                             'partial': [r for r in reports.values() if r.get('status') == 'partial'][:50],
                             'would_delete': sum(1 for r in reports.values() if r.get('status') == 'would_delete')}
        print(category, json.dumps({k: v for k, v in results[category].items() if k not in ('refused', 'partial')}),
              'refused', len(results[category]['refused']), 'partial', len(results[category]['partial']))
    log.close()
    return results


if __name__ == '__main__':
    C.OUT.mkdir(parents=True, exist_ok=True)
    if not EXECUTE and '--dry-run-executor' not in sys.argv:
        m = plan()
        for k, v in m['candidates_by_category'].items():
            print(f"candidate {k:26s} {v['files']:6d} files {v['bytes'] / 1e9:8.2f} GB {len(v['runs'])} runs")
        for k, v in sorted(m['kept_by_reason'].items(), key=lambda x: -x[1]['bytes'])[:40]:
            print(f"kept {k[:85]:85s} {v['files']:6d} files {v['bytes'] / 1e9:8.2f} GB")
        MANIFEST.write_text(json.dumps(m, indent=1, default=sorted) + '\n')
        raise SystemExit(0)
    m = json.loads(MANIFEST.read_text())
    if '--dry-run-executor' in sys.argv:  # every on-host recheck, nothing deleted
        execute(m, dry_run=True)
        raise SystemExit(0)
    free0 = int(ssh(f"df -B1 {REMOTE} | tail -1").split()[3])
    res = execute(m)
    free1 = int(ssh(f"df -B1 {REMOTE} | tail -1").split()[3])
    m.setdefault('executions', []).append({'executed_local': C.now_iso(), 'category': ONLY or 'all', 'results': res,
                                           'free_before': free0, 'free_after': free1})
    MANIFEST.write_text(json.dumps(m, indent=1, default=sorted) + '\n')
    print(f'remote free {free0 / 1e9:.1f} -> {free1 / 1e9:.1f} GB')
