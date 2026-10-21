#!/bin/bash
# chamfer_lean smoke test and matched cost benchmark (2026-10-21), from the merged main checkout.
# 1. smoke: chamfer_lean on hammer to step 200 (checkpoints every 100, evaluation at the end), then resumed from its
#    checkpoint to step 300 (resumability check).
# 2. A/B: reference C + D0 + M3D (i2-m3d overrides) vs chamfer_lean, hammer, seed 0, interleaved ref, chl, ref, chl on
#    the same MIG slice (which also hosts one running pretrain, the same for all four runs), 500 steps each on the
#    200k schedule; step time, GPU time and peak allocated GPU memory are read from metrics.jsonl (logs at 150-500).
set -u
WT=/home/ws/ws/splatter4d  # main checkout (merged chamfer_lean, suite evidence c5918fb/8185c41); outputs must lie in its repo
OUT=/home/ws/ws/splatter4d/runs/bench/chl-ab-20261021
PY=/home/ws/ws/splatter4d/.venv/bin/python
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:?set one authorized MIG UUID}
mkdir -p "$OUT"
cd "$WT" || exit 1
BASE=(configs/metaworld/base.yaml configs/metaworld/tasks/hammer.yaml)
REF=(aug.synth_views=2 aug.synth_render=true loss.depth_valid_far=true loss.motion_space=gaussian)
run() {  # name, stop, extra --set..., -- configs...
  local name=$1 stop=$2; shift 2
  echo "=== $(date '+%F %T') $name stop=$stop"
  nice -n 5 "$PY" -I scripts/train.py --no-wandb --name "$name" --output-root "$OUT" --resume auto "$@" \
    >> "$OUT/$name.console.log" 2>&1
  echo "exit $? $(date '+%F %T')"
}
run chl-smoke 200 --config "${BASE[@]}" configs/metaworld/variants/chamfer_lean.yaml \
  --set train.steps=200000 train.stop_step=200 train.seed=0 train.eval_at_start=false train.save_every=100
run chl-smoke 300 --config "${BASE[@]}" configs/metaworld/variants/chamfer_lean.yaml \
  --set train.steps=200000 train.stop_step=300 train.seed=0 train.eval_at_start=false train.save_every=100
grep -h "resumed from" "$OUT/chl-smoke/log.txt" "$OUT/chl-smoke.console.log" 2>/dev/null | tail -1
COMMON=(train.steps=200000 train.stop_step=500 train.seed=0 train.eval_at_start=false train.eval_every=100000
        train.save_every=100000)
for rep in 1 2; do
  run ab-ref-$rep 500 --config "${BASE[@]}" --set "${COMMON[@]}" "${REF[@]}"
  run ab-chl-$rep 500 --config "${BASE[@]}" configs/metaworld/variants/chamfer_lean.yaml --set "${COMMON[@]}"
done
echo "=== done $(date '+%F %T')"
