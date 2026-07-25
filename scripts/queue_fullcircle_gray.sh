#!/usr/bin/env bash
# Usage: queue_fullcircle_gray.sh <scene> <gpu:0|1> [--control-too]
# Queues gray masked (and optionally control) training on a prepped FullCircle scene.
set -eu
scene=$1; gpu=$2; shift 2
ctl=${1:-}
WT=/workspace/gray/worktrees/person-masks
SRC=/workspace/dataset/fullcircle_baselines/$scene
PY=/workspace/gray/.venv/bin/python
# free-VRAM thresholds: GPU0 (12GB) just needs to be idle; GPU1 (24GB) shared
thr=$([ "$gpu" = 0 ] && echo 9500 || echo 14000)

mkcmd() { # $1 = out name, $2 = extra train flags
  echo "cd $WT && for i in \$(seq 1 720); do f=\$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $gpu); [ \"\$f\" -gt $thr ] && break; sleep 60; done; echo \"GPU$gpu free=\${f}MiB\"; [ \"\$f\" -gt $thr ] || exit 42; export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu; $PY train.py -s $SRC -r 4 -m out/fullcircle/$1 -y --camera_model opencv_fisheye --batch_size 2 --eval --llffhold 0 --vignetting_comp --vignetting_terms 3 --fisheye_mask_dir $SRC $2 --pruning_min_weight 1e-8 && $PY render.py -m out/fullcircle/$1 --eval-models opencv_fisheye && $PY metrics.py -m out/fullcircle/$1"
}

id1=$(pueue add --print-task-id -- "$(mkcmd "${scene}_masked" "--person_mask_dir $SRC/person_masks_4")")
echo "queued ${scene}_masked: $id1"
if [ "$ctl" = "--control-too" ]; then
  id2=$(pueue add --print-task-id --after "$id1" -- "$(mkcmd "${scene}_control" "")")
  echo "queued ${scene}_control: $id2"
fi
