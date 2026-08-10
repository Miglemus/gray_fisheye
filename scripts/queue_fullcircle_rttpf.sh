#!/usr/bin/env bash
# Train gray + the learnable residual camera model on a FullCircle `refit_rttpf` scene.
#
# The recipe is byte-for-byte dataset/fullcircle_code/queue_fullcircle_gray_rttpf.sh (the
# one that produced the published `gray` row of fullcircle_tracks/rttpf_masked.csv) with
# ONLY `--camera_opt` added, so the delta is attributable to the camera model and nothing
# else. The point cloud (`point_cloud.safetensors`), the masks, the golden split and the
# calibration all come from the shared track dir, so every rung starts from the same init.
#
# usage: queue_fullcircle_rttpf.sh <scene> <rung> [gpu] [extra train.py args...]
#   rung = off | passthrough | tilt | radial | ana | noncentral | central_matched
#   `off` is the control: it must reproduce the published gray row.
#
# Run dirs:  out/fullcircle_rttpf/<scene>_refit_rttpf         (rung = noncentral)
#            out/fullcircle_rttpf/<scene>_refit_rttpf_<rung>  (anything else)
# so that dataset/fullcircle_code/masked_eval_rttpf.py finds them under method `gray-nc`
# with its default suffix.
set -eu

scene=$1; rung=$2; gpu=${3:-1}
shift 3 2>/dev/null || shift $#

WT=/workspace/gray/worktrees/noncentral-camera
SRC=/workspace/dataset/fullcircle_tracks/refit_rttpf/$scene
PY=/workspace/gray/.venv/bin/python
[ -d "$SRC" ] || { echo "no such scene: $SRC" >&2; exit 2; }

name=$scene"_refit_rttpf"
[ "$rung" = noncentral ] || name="${name}_${rung}"
# NAME_SUFFIX tags a probe run (different lr, schedule, ...) so it lands in its own dir and
# is never picked up as the headline `gray-nc` row by masked_eval_rttpf.py.
name="${name}${NAME_SUFFIX:-}"
out=out/fullcircle_rttpf/$name

# `off` must not allocate the camera model at all; every other rung unfreezes at 20 % of
# the schedule (3000/15000), the measured optimum -- see PROTOCOL.md "Recipe".
copt="--camera_opt $rung"
[ "$rung" = off ] || copt="$copt --camera_opt_from_iter 3000"

# These scenes peak at ~13 GB, and the 2026-08-07 batch lost 4 runs to an OOM 13 s after
# start: a process OUTSIDE the gpu1 pueue group held 12 GB of the same card. pueue only
# serialises its own group, so wait for the card to actually be free before allocating.
# Same guard as dataset/fullcircle_code/queue_fullcircle_gray.sh.
thr=$([ "$gpu" = 0 ] && echo 9500 || echo 15000)
guard="for i in \$(seq 1 720); do f=\$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $gpu); [ \"\$f\" -gt $thr ] && break; sleep 60; done; echo \"GPU$gpu free=\${f}MiB\"; [ \"\$f\" -gt $thr ] || exit 42"

pueue add -g "gpu$gpu" --print-task-id -- "cd $WT && $guard && \
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu && \
$PY train.py -s $SRC -r 4 -m $out -y --camera_model rad_tan_thin_prism_fisheye \
  --batch_size 2 --eval --llffhold 0 --vignetting_comp --vignetting_terms 3 \
  --fisheye_mask_dir $SRC --person_mask_dir $SRC/person_masks_4 \
  --pruning_min_weight 1e-8 $copt $* && \
$PY render.py -m $out --eval-models rad_tan_thin_prism_fisheye && \
$PY metrics.py -m $out && \
$PY measure_fps.py -m $out"
