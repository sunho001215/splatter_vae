"""Invoke run_tests' existing worker/evidence writer in CPU-only mode.

The standard parent requires a CUDA device even for CPU shards. This local harness hides all CUDA devices and
replaces ONLY the startup GPU enumeration check in this process, then selects explicit CPU test files. It never
changes product guards, launches a subprocess runner, allocates CUDA tensors, or claims full-suite acceptance.
"""
import json
import os
from pathlib import Path
import runpy
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import ops.resources.gpu_guard as guard
# Required by coordinator's hard no-GPU rule; test selection contains CPU files only.
guard.enforce_allowed_gpus=lambda verbose=True: []
if __name__=='__main__':
    out=Path(os.environ.get('BH_TEST_OUT',str(ROOT/'runs/evidence-bh5/tests')));out.mkdir(parents=True,exist_ok=True)
    files=sys.argv[1:]
    from scripts.test_shards import FILE_CLASSES
    assert files and all(FILE_CLASSES.get(f)=='cpu' for f in files), 'CPU-file selection only'
    spec=out/'worker.json'
    spec.write_text(json.dumps({'pytest_args':files+['-k','not test_producers_refuse_without_data_before_trainer_imports','-q','--tb=short','--basetemp',str(out/'tmp'),'-o',f'cache_dir={out}/pytest-cache','--junitxml',str(out/'junit.xml')], 'json':str(out/'results.json')}))
    sys.argv=[str(ROOT/'scripts/run_tests.py'),'--worker',str(spec)]
    try:
        runpy.run_path(str(ROOT/'scripts/run_tests.py'),run_name='__main__')
    finally:
        import resource
        import torch
        usage=resource.getrusage(resource.RUSAGE_SELF)
        resources={'max_rss_kib':usage.ru_maxrss,'user_seconds':usage.ru_utime,'system_seconds':usage.ru_stime,
                   'cuda_initialized':torch.cuda.is_initialized(),'threads':torch.get_num_threads(),
                   'full_suite_passed':False,'selection':files,'gpu_guard_startup_replaced_for_cpu_tests':True,
                   'deselected_gpu_child_test':'test_producers_refuse_without_data_before_trainer_imports'}
        (out/'resources.json').write_text(json.dumps(resources,indent=2))
        assert resources['cuda_initialized'] is False
        assert resources['max_rss_kib'] < 8*1024*1024
