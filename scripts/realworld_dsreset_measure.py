"""Build native overlays and measure selected manifest TRAIN bounds/SNeRL parameters."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
if __name__ == '__main__':
    from s4d.rwlearn.dsreset_runtime import install
    install()

from s4d.rwlearn.dsreset_measure import measure, SampledNativeStore
from s4d.rwlearn.native_depth import build_native_depth
from s4d.rwlearn.store import open_store
from ops.jobs.rwprep_mig import atomic


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True,help='manifest-bound S or historical SF learning inputs')
    p.add_argument('--sample-native',action='store_true',help='verify raw/rig identity and replay exactly the measurement frames without building full overlays')
    p.add_argument('--raw-root',help='RECORDED_ROOT=LOCAL_ROOT mapping for verified local native replay')
    p.add_argument('--rig',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    native=a.out/'store-native.yaml'
    if a.sample_native:
        base=open_store(a.inputs/'store.yaml')
        raw_paths={}
        if a.raw_root:
            from s4d.rwlearn.native_depth_mapped import mapped_raw
            split=json.loads((a.inputs/'split_seed0.json').read_text())
            raw_paths={ep:mapped_raw(base.raw_recording(ep),base.raw_sha256(ep),a.raw_root) for ep in split['train']}
        store=SampledNativeStore(base,a.rig,raw_paths=raw_paths)
    else:
        if a.raw_root:
            from s4d.rwlearn.native_depth_mapped import build_native_depth_mapped
            build_native_depth_mapped(str(a.inputs/'store.yaml'),str(a.rig),str(a.out/'overlays'),str(native),
                                      workers=1,raw_root=a.raw_root)
        else:
            build_native_depth(str(a.inputs/'store.yaml'),str(a.rig),str(a.out/'overlays'),str(native),workers=1)
        store=open_store(native)
    report=measure(json.loads((a.inputs/'manifest.json').read_text()),store,
                   json.loads((a.inputs/'split_seed0.json').read_text()))
    atomic(a.out/'scene.json',report)


if __name__=='__main__':main()
