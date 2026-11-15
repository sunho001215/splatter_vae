"""Container-local phase A: immutable snapshots, established stages and stream protocol.

This is an integrity envelope, not a mount/security boundary. No Docker, SSH or
host namespace is used by execution. Only the publisher uses RemoteClient.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
import threading

import yaml

MAIN = Path('/home/ws/ws/splatter4d')
CAMPAIGN = Path('/home/ws/ws/realworld-ops/dsreset-20261113/pipeline')
ASSETS = Path('/home/ws/ws/validation-20261031')
PROCESSED = Path('/home/ws/data/realworld-processed/dsreset-20261113')
REMOTE = '/home/compu/kaist/sunho/realworld/processed/dsreset-20261113'
STAGES = ('ingest', 'sensor_depth', 'foundation_stereo_depth', 'sam3_masks')
SCRIPT = 'scripts/realworld_local_phase_a.py'
PINS = {'sam3': 'f1647e8d0c1d280991f4d1b7d974c2704537fe66',
        'foundation_stereo': '6e8806816b533e4d13ddbb95ffa907b797060a62',
        'dinov2': '7764ea0f912e53c92e82eb78a2a1631e92725fc8'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda: stream.read(8 * 2**20), b''):
            h.update(b)
    return h.hexdigest()


def atomic(path, doc):
    from s4d.realworld.io import atomic_write_json
    atomic_write_json(Path(path), doc)


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True,
                                   env={**os.environ, 'GIT_OPTIONAL_LOCKS': '0'}).strip()


def inventory(root):
    root = Path(root)
    result = {}
    for directory, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d != '.git')
        for name in sorted(files):
            p = Path(directory) / name
            require(not p.is_symlink(), f'snapshot symlink: {p}')
            require(p.is_file(), f'non-regular snapshot entry: {p}')
            result[str(p.relative_to(root))] = sha(p)
    return result


def readonly(root):
    for directory, dirs, files in os.walk(root):
        for name in files:
            p = Path(directory) / name
            p.chmod(p.stat().st_mode & ~0o222)
        Path(directory).chmod(Path(directory).stat().st_mode & ~0o222)


def clone_pin(source, destination, commit):
    require(not git(source, 'status', '--porcelain', '--untracked-files=no'), f'dirty source: {source}')
    subprocess.run(['git', 'init', '-q', str(destination)], check=True)
    subprocess.run(['git', '-C', str(destination), '-c', 'protocol.file.allow=always',
                    'fetch', '-q', '--depth=1', str(source), commit], check=True)
    subprocess.run(['git', '-C', str(destination), 'checkout', '-q', '--detach', 'FETCH_HEAD'], check=True)
    require(git(destination, 'rev-parse', 'HEAD') == commit, 'snapshot pin differs')


def prepare(source, bundle, audit=CAMPAIGN / 'local-weight-audit.json'):
    """Materialize NEW copies only; never chmod shared weights, environments or source."""
    source, bundle = Path(source).resolve(), Path(bundle).resolve()
    require(not bundle.exists(), 'bundle exists; reuse it rather than replacing an execution snapshot')
    commit = git(source, 'rev-parse', 'HEAD')
    bundle.mkdir(parents=True, mode=0o700)
    frozen = bundle / 'frozen'
    frozen.mkdir()
    clone_pin(source, frozen / 'source', commit)
    upstream = {'sam3': ASSETS / 'upstream/sam3',
                'foundation_stereo': ASSETS / 'upstream/FoundationStereo',
                'dinov2': ASSETS / 'cache/torch/hub/facebookresearch_dinov2_main'}
    for name, src in upstream.items():
        require(git(src, 'rev-parse', 'HEAD') == PINS[name], f'upstream pin: {name}')
        clone_pin(src, frozen / name, PINS[name])
    helper = Path('/home/ws/ws/realworld-ops/pre/tools/rwprep/rwprep_local.py')
    shutil.copyfile(helper, frozen / 'rwprep_local.py')
    weights = []
    for row in json.loads(Path(audit).read_text()):
        if 'path' not in row or not any(t in row['path'] for t in ('/sam3/', '/foundation_stereo/')):
            continue
        src = Path(row['path'])
        require(sha(src) == row['expected_sha256'], f'weight pin: {src}')
        dst = frozen / 'weights' / src.relative_to(ASSETS / 'downloads')
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        require(sha(dst) == row['expected_sha256'], f'copy checksum: {dst}')
        weights.append({'dest': str(dst), 'sha256': row['expected_sha256'],
                        'worker': 'sam3' if '/sam3/' in str(src) else 'foundation_stereo'})
    require(len(weights) == 3, 'phase A requires exactly three pinned weight/config files')
    envs = {n: str(ASSETS / 'envs' / n / 'bin/python') for n in ('sam3', 'foundation_stereo')}
    envs['s4d_kernels'] = str(MAIN / '.venv/bin/python')
    # Record interpreter bytes and package metadata. Shared environments are not modified.
    env_files = {}
    for name, py in envs.items():
        env_files[str(Path(py).resolve())] = sha(Path(py).resolve())
        prefix = Path(py).parent.parent
        for p in prefix.glob('lib/python*/site-packages/*.dist-info/RECORD'):
            env_files[str(p)] = sha(p)
    for name in ('sam3', 'foundation_stereo'):
        (frozen / f'{name}_launch.py').write_text(
            'import os,runpy,torch\n'
            'torch.cuda.set_per_process_memory_fraction(float(os.environ["S4D_PHASE_A_GPU_GIB"])*2**30/'
            'torch.cuda.get_device_properties(0).total_memory)\n'
            f'runpy.run_path({str(frozen / "source/s4d/realworld/workers" / (name + "_worker.py"))!r},run_name="__main__")\n')
    manifest = {'schema': 'local-phase-a-envelope/1', 'commit': commit,
                'frozen': str(frozen), 'files': inventory(frozen), 'environment_files': env_files,
                'upstream_pins': PINS, 'weights': weights, 'interpreters': envs,
                'helper_original_sha256': sha(helper),
                'limitations': ['chmod and hashes detect drift; root can bypass chmod',
                                 'campaign root/network/cgroups are inherited, not isolated',
                                 'shared installed environments are recorded, not copied']}
    readonly(frozen)
    atomic(bundle / 'envelope.json', manifest)
    (bundle / 'envelope.json').chmod(0o444)
    return {'bundle': str(bundle), 'envelope_sha256': sha(bundle / 'envelope.json'), 'commit': commit}


def verify_envelope(bundle, expected):
    path = Path(bundle) / 'envelope.json'
    require(sha(path) == expected, 'envelope manifest drift')
    doc = json.loads(path.read_text())
    require(inventory(doc['frozen']) == doc['files'], 'code/weight snapshot SHA drift')
    require(all(sha(p) == h for p, h in doc['environment_files'].items()), 'environment metadata/interpreter drift')
    require(git(Path(doc['frozen']) / 'source', 'rev-parse', 'HEAD') == doc['commit'], 'source HEAD drift')
    for name, pin in doc['upstream_pins'].items():
        require(git(Path(doc['frozen']) / name, 'rev-parse', 'HEAD') == pin, f'{name}: HEAD drift')
    for directory, dirs, files in os.walk(doc['frozen']):
        for p in [Path(directory), *(Path(directory) / n for n in files)]:
            require(not p.stat().st_mode & 0o222, f'writable frozen entry: {p}')
    return doc


def identity(episode):
    m = re.fullmatch(r'(drawer-open-v2|covering|pick-and-place)-(success|failure)-(episode_\d{6})', episode)
    require(m, f'invalid label-inclusive episode identity: {episode}')
    return m.groups()


def make_job(row, bundle, output=PROCESSED):
    bundle = Path(bundle).resolve()
    task, label, number = identity(row['episode'])
    cfg = yaml.safe_load(Path(row['config']).read_text())
    raw = Path(cfg['adapter']['config']['episodes'][row['episode']])
    require(raw.is_relative_to('/home/ws/data/real_world'), 'raw outside approved local root')
    cfg['output_root'] = str(Path(output) / task / label / number)
    return {'episode': row['episode'], 'raw': str(raw), 'raw_sha256': row['raw_sha256'],
            'pipeline': cfg, 'sam3_prompt_order': list(cfg['stages']['sam3_masks']['prompts']),
            'envelope_sha256': sha(bundle / 'envelope.json')}


def execution_pipeline(job):
    """JSON object order is not an inference contract; carry prompt order as a list."""
    import copy
    cfg = copy.deepcopy(job['pipeline'])
    prompts = cfg['stages']['sam3_masks']['prompts']
    order = job.get('sam3_prompt_order')
    require(isinstance(order, list) and len(order) == len(prompts) and
            len(set(order)) == len(order) and set(order) == set(prompts),
            'SAM3 prompt order must name every prompt exactly once')
    cfg['stages']['sam3_masks']['prompts'] = {name: prompts[name] for name in order}
    return cfg


def verified_job(bundle, jobfile, expected):
    bundle, jobfile = Path(bundle).resolve(), Path(jobfile).resolve()
    require(jobfile.is_relative_to(bundle / 'jobs'), 'job must be inside the declared bundle/jobs')
    require(sha(jobfile) == expected, 'job SHA drift')
    job = json.loads(jobfile.read_text())
    doc = verify_envelope(bundle, job['envelope_sha256'])
    identity(job['episode'])
    require(sha(job['raw']) == job['raw_sha256'], 'raw SHA drift')
    require(job['pipeline']['adapter']['config']['episodes'] == {job['episode']: job['raw']}, 'raw identity mismatch')
    return job, doc


def require_job_gate(bundle, expected, payload):
    from ops.jobs import local_phase_a_hybrid as h
    return h.require_gate(bundle, expected) if h.is_hybrid(payload) else require_gate(bundle, expected)


def scheduler_inputs(job, root=MAIN):
    """Pure-data resolver used by established local placement and admission."""
    args = job.get('args', [])
    require(len(args) == 8 and args[::2] == ['--bundle', '--job', '--job-sha256', '--gate-sha256'], 'invalid phase-A argv')
    bundle, path, checksum, gate_sha = args[1::2]
    require(job.get('id', '').startswith('rwv2-20261113-prep-ours-') and job.get('registration_tool') ==
            'runs/setup/tools/register_local_phase_a_20261115.py', 'unregistered phase-A row')
    payload, envelope = verified_job(bundle, path, checksum)
    from ops.jobs import local_phase_a_hybrid as h
    require(job['id'] == f"rwv2-20261113-prep-ours-{payload['episode']}-{h.suffix(payload)}", 'row/episode mismatch')
    require(Path(payload['pipeline']['output_root']) == PROCESSED.joinpath(*identity(payload['episode'])), 'noncanonical local output')
    require_job_gate(bundle, gate_sha, payload)
    if h.is_hybrid(payload):
        require(sha(Path(bundle)/'dedup-audit.json') == payload['dedup_audit_sha256'], 'dedup audit drift')
    gate_path = Path(bundle) / (h.REPORT if h.is_hybrid(payload) else "equivalence.json")
    return list(args), {'inputs': {payload['raw']: payload['raw_sha256']}, 'dependency_inputs': {},
                        'input_closure': {str(path): checksum, str(Path(bundle)/'envelope.json'): payload['envelope_sha256'],
                                          str(gate_path): gate_sha,
                                          **({str(Path(bundle)/'dedup-audit.json'): payload['dedup_audit_sha256']} if h.is_hybrid(payload) else {})}, 'phase_a_envelope_sha256': payload['envelope_sha256'], 'phase_a_job_sha256': checksum,
                        'raw_sha256': payload['raw_sha256'], 'equivalence_sha256': gate_sha}


def require_gate(bundle, expected):
    path = Path(bundle) / 'equivalence.json'
    require(sha(path) == expected, 'equivalence report drift')
    gate = json.loads(path.read_text())
    require(gate.get('qualified') is True and gate.get('coverage') == 'full-two-episodes', 'full equivalence gate has not passed')
    require(gate['envelope_sha256'] == sha(Path(bundle) / 'envelope.json'), 'gate belongs to another envelope')
    require({r['task'] for r in gate['episodes']} == {'covering', 'pick-and-place'} and len(gate['episodes']) == 2,
            'gate must cover both success episodes')
    for r in gate['episodes']:
        require(r['passed'] is True and r['label'] == 'success', 'gate episode failed')
        require(set(r['stages']) == set(STAGES) and all(s.get('passed') and s.get('files') for s in r['stages'].values()),
                'gate lacks complete stage comparisons')
        for p, h in r['evidence_files'].items():
            require(sha(p) == h, f'gate evidence drift: {p}')
    return gate


@contextlib.contextmanager
def output_lock(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.local-phase-a.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def executable_ram(path):
    """Native Python/Triton extensions need RAM backing AND executable mappings."""
    from s4d.realworld.workers._protocol import _mount_for
    ram = _mount_for(str(path), Path('/proc/self/mountinfo').read_text())
    require(ram['ok'], f'native code cache must be RAM-backed: {path}')
    require(not os.statvfs(path).f_flag & os.ST_NOEXEC, f'native code cache mount is noexec: {path}')
    return {**ram, 'executable': True}


@contextlib.contextmanager
def native_code_cache():
    # /dev is a small executable tmpfs in the campaign container. Never remount
    # shared filesystems; keep large frame/model scratch on /dev/shm instead.
    executable_ram('/dev')
    require(shutil.disk_usage('/dev').free >= 32 * 2**20, 'native code tmpfs needs at least 32 MiB free')
    with tempfile.TemporaryDirectory(prefix='s4d-localA-code-', dir='/dev') as name:
        path = Path(name)
        path.chmod(0o700)
        yield path, executable_ram(path)


def execute(bundle, jobfile, expected, gpu_gib=12, gate_sha=None, stages=STAGES):
    """Run inside pinned source; rehash in finally even on stage failure."""
    bundle = Path(bundle).resolve()
    job, doc = verified_job(bundle, jobfile, expected)
    from ops.jobs import local_phase_a_hybrid as h
    if h.is_hybrid(job):
        require(tuple(stages) in (STAGES, h.STAGES), 'hybrid jobs permit only qualified stages')
        stages = h.STAGES
        require(gate_sha is not None, 'hybrid execution requires its three-stage gate')
    if gate_sha is not None:
        require_job_gate(bundle, gate_sha, job)
    root = Path(job['pipeline']['output_root'])
    require(not any(p.is_symlink() for p in (root, *root.parents)), 'output symlink escape')
    require(root == PROCESSED.joinpath(*identity(job['episode'])) or root.is_relative_to(bundle / 'gate'), 'unsafe output root')
    from ops.resources.gpu_guard import MIG_DEVICES
    gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    require(gpu in MIG_DEVICES, 'assign exactly one approved local MIG UUID')
    import torch
    require(torch.cuda.device_count() == 1, 'exactly one CUDA device must be visible')
    require(torch.cuda.mem_get_info()[0] > (gpu_gib + 1) * 2**30, 'MIG has insufficient free memory')
    with output_lock(root), native_code_cache() as (code_cache, code_mount), \
            tempfile.TemporaryDirectory(prefix='s4d-localA-', dir='/dev/shm') as tmp:
        scratch = Path(tmp)
        scratch.chmod(0o700)
        os.environ.update(TMPDIR=tmp, TORCH_HOME=str(scratch / 'torch'), XDG_CACHE_HOME=str(scratch / 'cache'),
                          TRITON_CACHE_DIR=str(code_cache / 'triton'),
                          TORCH_EXTENSIONS_DIR=str(code_cache / 'extensions'),
                          TORCHINDUCTOR_CACHE_DIR=str(code_cache / 'inductor'),
                          CUDA_CACHE_PATH=str(scratch / 'cuda'),
                          HF_HOME=str(scratch / 'hf'), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                          S4D_CACHE_ROOT=str(scratch / 's4d'), S4D_RWPREP_SOURCE=str(Path(doc['frozen']) / 'source'),
                          S4D_PHASE_A_GPU_GIB=str(gpu_gib), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
        tempfile.tempdir = tmp
        hub = scratch / 'torch/hub'
        hub.mkdir(parents=True)
        (hub / 'facebookresearch_dinov2_main').symlink_to(Path(doc['frozen']) / 'dinov2', target_is_directory=True)
        (hub / 'trusted_list').write_text('facebookresearch_dinov2\n')
        workers = {name: {'python': py} for name, py in doc['interpreters'].items()}
        for name, subpath in (('sam3', 'sam3/sam3.pt'), ('foundation_stereo', 'foundation_stereo/23-51-11/model_best_bp2.pth')):
            workers[name].update(repo_dir=str(Path(doc['frozen']) / name),
                                 checkpoint=str(Path(doc['frozen']) / 'weights' / subpath),
                                 script=str(Path(doc['frozen']) / f'{name}_launch.py'), expected_commit=PINS[name])
        server = dict(ssh=['campaign-local'], container='campaign-local', workdir=tmp,
                      source_root=str(Path(doc['frozen']) / 'source'), workers=workers, env={}, retries=1,
                      backoff_s=[0], call_timeout_s=3600, hello_timeout_s=120, weights=doc['weights'])
        server_path = scratch / 'server.yaml'
        server_path.write_text(yaml.safe_dump(server))
        pipeline = scratch / 'pipeline.yaml'
        pipeline.write_text(yaml.safe_dump(execution_pipeline(job), sort_keys=False))
        spec = importlib.util.spec_from_file_location('established_rwprep', Path(doc['frozen']) / 'rwprep_local.py')
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        class IntegrityServer(helper.LocalServer):
            def verify_container(self):
                from s4d.realworld.workers._protocol import _mount_for
                ram = _mount_for(tmp, Path('/proc/self/mountinfo').read_text())
                require(ram['ok'], 'private worker scratch must be tmpfs')
                return {'envelope': 'local-phase-a-integrity/1', 'gpu': gpu, 'scratch': tmp,
                        'scratch_mode': oct(scratch.stat().st_mode & 0o777), 'tmpfs': ram,
                        'native_code_cache': {**code_mount, 'mode': oct(code_cache.stat().st_mode & 0o777)},
                        'limitations': doc['limitations']}
            def audit_start(self):
                return {'envelope_sha256': job['envelope_sha256'], 'started_unix': time.time()}
            def audit_end(self, start):
                verify_envelope(bundle, start['envelope_sha256'])
                return {'ok': True, 'code_weights_sha256_unchanged': True, 'scratch': tmp}
        helper.LocalServer = IntegrityServer
        helper.patch()
        from ops.inference_server import run_episode
        affinity = sorted(os.sched_getaffinity(0))[:4]
        os.sched_setaffinity(0, affinity)
        peak_rss = [0]
        stop = threading.Event()
        def monitor():
            while not stop.wait(.25):
                todo, rss = [os.getpid()], 0
                while todo:
                    pid = todo.pop()
                    try:
                        todo += [int(x) for x in Path(f'/proc/{pid}/task/{pid}/children').read_text().split()]
                        rss += int(Path(f'/proc/{pid}/statm').read_text().split()[1]) * os.sysconf('SC_PAGE_SIZE')
                    except (OSError, ValueError, IndexError):
                        pass
                peak_rss[0] = max(peak_rss[0], rss)
        watcher = threading.Thread(target=monitor, daemon=True)
        watcher.start()
        proof = {'status': 'running', 'episode': job['episode'], 'job_sha256': expected,
                 'envelope_sha256': job['envelope_sha256'], 'gpu': gpu, 'gpu_cap_gib': gpu_gib,
                 'started_unix': time.time(), 'stages': []}
        atomic(root / 'local-phase-a.json', proof)
        try:
            # Close the pool after each stage so FS and SAM3 never coexist on a MIG.
            for stage in stages:
                require(stage in STAGES, 'phase A only')
                proof['active_stage'] = stage
                atomic(root / 'local-phase-a.json', proof)
                print(f'[{stage}] starting; episode={job["episode"]}; pid={os.getpid()}; scratch={tmp}', flush=True)
                helper.LocalServer.needed_workers = tuple(helper.STAGE_WORKERS.get(stage, ()))
                code = run_episode.main(['--pipeline', str(pipeline), '--server', str(server_path),
                                         '--episode', job['episode'], '--stages', stage])
                require(code == 0, f'{stage}: exit {code}')
                proof['stages'].append(stage)
                atomic(root / 'local-phase-a.json', proof)
            proof.pop('active_stage', None)
            if h.is_hybrid(job):
                h.finish(root, job)
                proof['mode'] = h.MODE
                proof['remote_remaining_stages'] = ['sam3_masks']
            proof['status'] = 'ok'
        except BaseException as exc:
            proof.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            try:
                verify_envelope(bundle, job['envelope_sha256'])
                require(sha(job['raw']) == job['raw_sha256'], 'raw changed during episode')
                proof['integrity_after'] = True
            except Exception:
                proof['status'] = 'integrity-failed'
                raise
            finally:
                stop.set()
                watcher.join()
                proof['finished_unix'] = time.time()
                runs = [json.loads(p.read_text()) for p in (root/'runs').glob('*/run.json')]
                peak_cuda = max((st.get('peak_cuda_reserved_bytes') or 0 for r in runs for st in r.get('stages', {}).values()), default=0)
                proof['footprints'] = {'measured': bool(peak_cuda), 'gpu_peak_reserved_gib': peak_cuda / 2**30,
                                       'rss_peak_gib': peak_rss[0] / 2**30, 'cpu_affinity': affinity,
                                       'cpu_threads': len(affinity),
                                       'disk_gib': sum(p.stat().st_blocks*512 for p in root.rglob('*') if p.is_file())/2**30}
                if proof['status'] == 'running':
                    proof['status'] = 'failed'
                atomic(root / 'local-phase-a.json', proof)
        return proof


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    sub = p.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--source', type=Path, required=True)
    prep.add_argument('--bundle', type=Path, required=True)
    run = sub.add_parser('run')
    run.add_argument('--bundle', type=Path, required=True)
    run.add_argument('--job', type=Path, required=True)
    run.add_argument('--job-sha256', required=True)
    run.add_argument('--gate-sha256', required=True)
    run.add_argument('--gpu-gib', type=float, default=12)
    a = p.parse_args(argv)
    if a.command == 'prepare':
        print(json.dumps(prepare(a.source, a.bundle), indent=2))
    else:
        execute(a.bundle, a.job, a.job_sha256, a.gpu_gib, a.gate_sha256)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
