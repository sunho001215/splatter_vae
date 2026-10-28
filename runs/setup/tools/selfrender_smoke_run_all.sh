#!/bin/bash
set -u
cd /home/ws/ws/splatter4d/.worktrees/selfrender
export CUDA_VISIBLE_DEVICES=MIG-c3511b41-5f78-52f1-9b93-9c6794cdbbdd WANDB_MODE=disabled
PY=/home/ws/ws/splatter4d/.venv/bin/python
D=/tmp/claude-0/-home-ws/738aaa26-ed64-46bb-9068-b4ef624cd2ac/scratchpad/sr/smoke
CSW4="configs/metaworld/base.yaml configs/metaworld/tasks/hammer.yaml configs/metaworld/variants/chamfer_lean.yaml configs/metaworld/variants/chamfer_lean_randmask.yaml configs/metaworld/variants/enc_cs.yaml configs/metaworld/variants/enc_w4.yaml"
COMMON="train.steps=300 train.log_every=50 train.seed=0 train.eval_at_start=false wandb.enabled=false"
run() { name=$1; shift; echo "=== $name $(date +%T)"; nice -n 5 $PY -E -s runs/setup/tools/selfrender_smoke_driver.py "$@" > $D/$name.log 2>&1; echo "exit $? $(date +%T)"; }
run smoke-csw4-hammer --config $CSW4 --name smoke-csw4-hammer --output-root runs/smoke --set $COMMON
run smoke-sr-full-hammer --config $CSW4 configs/metaworld/variants/selfrender_replace_v2.yaml --name smoke-sr-full-hammer --output-root runs/smoke --set $COMMON "aug.synth_ramp_steps=[0,150]"
run smoke-sr-warmup-hammer --config $CSW4 configs/metaworld/variants/selfrender_replace_v2.yaml --name smoke-sr-warmup-hammer --output-root runs/smoke --set $COMMON
