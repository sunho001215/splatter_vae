"""Offline prep container entry; fingerprinted stages provide restart idempotence."""
import json
import os
from pathlib import Path
import subprocess
import sys

SRC=Path('/opt/src/splatter4d')
sys.path.insert(0,str(SRC))
from ops.jobs.rwprep_mig import atomic


def main():
    job=json.loads(Path(sys.argv[1]).read_text())
    out=Path(job['result_dir'])/f"a{job['attempt']}"
    out.mkdir(parents=True,exist_ok=True)
    Path('/ram/work').mkdir(parents=True,exist_ok=True)
    proof={'status':'running','queue_sha256':job['queue_sha256'],'steps':[]}
    for n, step in enumerate(job['steps']):
        if step['kind']=='rwprep':
            argv=[sys.executable,'-I','-B','/opt/rwprep/rwprep_local.py',*step['argv']]
        elif step['kind']=='python':
            argv=[step.get('python',sys.executable),'-B',*step['argv']]
            if step.get('cuda_cap_gib'):
                cap=step['cuda_cap_gib']
                # Single-process inference bound; CPU rasterization keeps the
                # renderer out of the matcher's MIG budget. OOM fails visibly.
                code='import torch,runpy,sys; torch.cuda.set_per_process_memory_fraction('+str(cap)+ '*2**30/torch.cuda.get_device_properties(0).total_memory); m=sys.argv.pop(1); runpy.run_module(m,run_name="__main__")'
                if step['argv'][:1]!=['-m']:
                    raise ValueError('CUDA capped jobs must name a module')
                argv=[step.get('python',sys.executable),'-B','-c',code,*step['argv'][1:]]
        else:
            raise ValueError('unknown prep command kind')
        env=step.get('env',{})
        if set(env)-{'PYTHONPATH','TORCH_HOME'}:
            raise ValueError('unsupported worker environment')
        with (out/f'{n}.log').open('ab') as stream:
            result=subprocess.run(argv,cwd=SRC,env={**os.environ,'PYTHONPATH':str(SRC),
                'S4D_RWPREP_STEPDIR':str(out/str(n)),**env},stdout=stream,stderr=subprocess.STDOUT)
        proof['steps'].append({'argv':argv,'exit_code':result.returncode})
        proof['status']='failed' if result.returncode else 'running'
        atomic(out/'result.json',proof)
        if result.returncode:
            return result.returncode
    proof['status']='ok'
    atomic(out/'result.json',proof)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
