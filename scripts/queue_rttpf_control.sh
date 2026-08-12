#!/bin/bash
# Queue the re-calibration control matrix (7 myscenes x N rungs) on one GPU.
#
# usage: queue_rttpf_control.sh <out_root> <resolution> <iterations> <gpu> <rung>[,<rung>...] [extra args]
#   e.g. queue_rttpf_control.sh tmp/r8_control 8 7500 0 off,rttpf,noncentral,rttpf_z
#        queue_rttpf_control.sh tmp/r4_control 4 15000 1 rttpf
#
# Everything except `--camera_opt` is the baseline recipe (train_myscenes_at.sh), and the
# unfreeze point is 20 % of the run for every rung -- the recipe PROTOCOL.md measured for
# `noncentral`, reused verbatim so the control is not handicapped by a schedule it never
# got to choose. The intrinsic learning rate defaults to the swept optimum in config.py;
# pass --camera_opt_lr_intrinsics to override.
#
# -r 4 peaks at 11.8-16.5 GB, so it needs GPU 1 and a free-VRAM guard: a pueue group
# serializes its own tasks but does not reserve the card, and four FullCircle runs once
# died on OOM 13 s in because a job started outside the group held 12 GB.
set -e
OUT_ROOT="$1"
RESOLUTION="$2"
ITERATIONS="$3"
GPU="$4"
RUNGS="$5"
shift 5

WORKTREE=/workspace/gray/worktrees/rttpf-intrinsics
SCENES="atrium classroom forest library reception tunnel workshop"
FREEZE=$((ITERATIONS / 5))
[ "$GPU" = "1" ] && GUARD=17000 || GUARD=9000

for scene in $SCENES; do
    for rung in ${RUNGS//,/ }; do
        MODEL="$OUT_ROOT/${scene}_${rung}"
        EXTRA="--camera_opt $rung"
        [ "$rung" != "off" ] && EXTRA="$EXTRA --camera_opt_from_iter $FREEZE"
        pueue add --group "gpu$GPU" --print-task-id -- \
            "cd $WORKTREE && export PATH=/workspace/gray/.venv/bin:\$PATH \
             CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU && \
             for i in \$(seq 1 720); do \
               f=\$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $GPU); \
               [ \"\$f\" -gt $GUARD ] && break; sleep 10; done && \
             bash scripts/train_myscenes_at.sh $scene $MODEL $RESOLUTION $ITERATIONS $EXTRA $*"
    done
done
