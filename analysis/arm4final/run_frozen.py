"""Run the FROZEN arm4 analysis scripts unchanged (arm4_rl.py / arm4_render_metrics.py) against live campaign data.

Deviation handled here (documented, 2026-11-11, agent ARM4): two hash-pinned reference files of the frozen analysis,
work/arch_compare/scripts/ranking.py and rl_common.py, were edited AFTER the arm4 registration by the CSW4 tube-ratio
finalisation (Codex session 2026-11-06 18:13 KST: S7 removal, label text, RNG-advance helper). The arm4 scripts never
import them (arm4_rl.ranking is self-contained); they are pinned only as provenance. The exact registered bytes were
reconstructed by reverse-applying the recorded patches (frozen_inputs/, sha256 equal to the registered values). This
runner makes arm4_common.verify_inputs hash those reconstructed copies for exactly these two paths; every other pinned
input is still verified against the live file, and the frozen analysis scripts are executed byte-for-byte.

    python -I -B run_frozen.py <src-dir> rl|render [script args...]

<src-dir> is a reports/results-artifact/src directory whose parent holds the arm4/ output directory (use a scratch
mirror to keep the registered reference-only outputs in the main checkout untouched).
"""
import hashlib
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REMAP = {
    '/home/ws/ws/presentation-assets-tools/work/arch_compare/scripts/ranking.py': HERE / 'frozen_inputs/ranking.py',
    '/home/ws/ws/presentation-assets-tools/work/arch_compare/scripts/rl_common.py': HERE / 'frozen_inputs/rl_common.py',
}
REGISTERED = {
    'ranking.py': '69b3d1b7935ee16e88e2cf5cf21540d4c77c90e94194ade52678c38d6982334e',
    'rl_common.py': 'b6e4a2867377d19370bc66fc7491fd8b27df734bc17cd30ba81107fd4ec202ce',
}


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    src, which, rest = Path(sys.argv[1]).resolve(), sys.argv[2], sys.argv[3:]
    for live, frozen in REMAP.items():
        assert _sha(frozen) == REGISTERED[frozen.name], f'reconstructed {frozen.name} does not match the registration'
    sys.path.insert(0, str(src))
    import arm4_common  # noqa: E402  (the frozen module, from src)
    orig_sha = arm4_common.sha

    def sha(path):
        p = str(path)
        return orig_sha(REMAP[p]) if p in REMAP else orig_sha(path)

    arm4_common.sha = sha
    script = src / {'rl': 'arm4_rl.py', 'render': 'arm4_render_metrics.py'}[which]
    sys.argv = [str(script), *rest]
    runpy.run_path(str(script), run_name='__main__')


if __name__ == '__main__':
    main()
