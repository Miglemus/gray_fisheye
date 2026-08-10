#!/usr/bin/env bash
# Re-measure FPS for every FullCircle refit_rttpf run, on ONE card, back to back.
#
# Why it exists:
#  * the published `gray` runs on this track have no fps.csv at all;
#  * the first `noncentral` batch was timed while a foreign process shared the card, so its
#    numbers range 63-245 FPS on comparable scenes and are meaningless.
# FPS is a throughput number: a run timed on the 2080 Ti against one timed on the TITAN RTX
# measures the cards. Everything here is pinned to gpu1, one job, sequential -- same reason
# dataset/fullcircle_code/queue_perf.sh hardcodes the group.
#
# Writes fps.csv next to each run (build_manifest.py's read_perf picks it up). It does NOT
# touch checkpoints, renders or metrics.
#
# usage: bash scripts/measure_fullcircle_fps.sh          # queues one gpu1 job
set -eu

PY=/workspace/gray/.venv/bin/python
NC=/workspace/gray/worktrees/noncentral-camera
GRAY=/workspace/gray/worktrees/fullcircle-erp
SCENES="room1 room2 room3 flat1 flat2 lab lounge dark persons"

body="for i in \$(seq 1 720); do f=\$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1); [ \"\$f\" -gt 15000 ] && break; sleep 60; done; [ \"\$f\" -gt 15000 ] || exit 42;
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1;
for s in $SCENES; do
  for m in \"\$s\"_refit_rttpf \"\$s\"_refit_rttpf_off; do
    d=$NC/out/fullcircle_rttpf/\$m
    [ -d \"\$d\" ] && { cd $NC && $PY measure_fps.py -m out/fullcircle_rttpf/\$m || echo \"FAILED \$m\"; }
  done
  d=$GRAY/out/fullcircle_rttpf/\"\$s\"_refit_rttpf
  [ -d \"\$d\" ] && { cd $GRAY && $PY measure_fps.py -m out/fullcircle_rttpf/\"\$s\"_refit_rttpf || echo \"FAILED gray \$s\"; }
done"

pueue add -g gpu1 --print-task-id -- "$body"
