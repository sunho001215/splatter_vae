"""Build native overlays and measure shared SF-train bounds/SNeRL parameters."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from s4d.rwlearn.dsreset_measure import measure
from s4d.rwlearn.native_depth import build_native_depth
from s4d.rwlearn.store import open_store
from ops.jobs.rwprep_mig import atomic


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True,help='manifest-bound SF learning inputs')
    p.add_argument('--rig',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    native=a.out/'store-native.yaml'
    build_native_depth(str(a.inputs/'store.yaml'),str(a.rig),str(a.out/'overlays'),str(native),workers=1)
    report=measure(json.loads((a.inputs/'manifest.json').read_text()),open_store(native),
                   json.loads((a.inputs/'split_seed0.json').read_text()))
    atomic(a.out/'scene.json',report)


if __name__=='__main__':main()
