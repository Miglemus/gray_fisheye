#!/usr/bin/env bash
# Print (or, with --queue, submit) the exact pueue lines for the provenanced FPS
# re-measurement and for the FoV sweep.
#
#   bash scripts/perf/queue_fov_sweep.sh              # print only  <-- the default
#   bash scripts/perf/queue_fov_sweep.sh --queue      # actually submit
#
# IT PRINTS BY DEFAULT AND SUBMITS NOTHING.  At the time of writing `pueue status` holds
# ~180 queued tasks belonging to four other sessions (worktrees/mip360-pinhole,
# worktrees/rttpf-intrinsics, dataset/rebaseline_2026-08, worktrees/noncentral-cuda).
# Submitting into that would create contention -- and contention is the exact defect this
# harness exists to remove.  Run with --queue only once the foreign work has drained.
#
# Every line carries, in order and on purpose:
#   * `--group gpu1`                pueue serialises its own group ...
#   * `gpuwait.sh 1 <MiB>`          ... but does NOT reserve the card, so also wait for VRAM
#   * CUDA_DEVICE_ORDER=PCI_BUS_ID  without it torch's index is not nvidia-smi's, and
#                                   CUDA_VISIBLE_DEVICES=1 lands on the 11 GB 2080 Ti
#   * CUDA_VISIBLE_DEVICES=1        the 24 GB TITAN RTX. EVERY speed row in one table must
#                                   come from ONE card or the table measures the cards.
# and `bench_fps.py` itself aborts if any foreign CUDA context is on the card, both before
# and after the timed region, and writes a provenance sidecar either way.
set -eu

QUEUE=0
[ "${1:-}" = "--queue" ] && QUEUE=1

WT=/workspace/gray/worktrees/noncentral-camera
PY=/workspace/gray/.venv/bin/python
GPUWAIT=/workspace/gray/worktrees/masked-efficiency/scripts/gpuwait.sh
ENV1='CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1'

# The sweep scene: `tunnel`, `--camera_opt off`, -r 4, 15k, 521 931 gaussians.
#   * `off` because the sweep must NOT put ray synthesis on the Python path (that is the
#     camera model's FPS tax, measured at 0.52-0.86x, and it is not what is being studied);
#   * tunnel because its `off`/`noncentral` pair is the reference of this whole branch, so
#     a speed curve on it joins to an existing quality curve without a second training.
SWEEP_RUN=/workspace/gray/tmp/r4/tunnel_off

emit() {
  local desc=$1; shift
  echo
  echo "# $desc"
  if [ "$QUEUE" = "1" ]; then
    pueue add --group gpu1 --print-task-id -- "$*"
  else
    printf 'pueue add --group gpu1 --print-task-id -- %s\n' "'$*'"
  fi
}

need_dir() {  # never emit a command whose run directory does not exist
  [ -d "$1" ] && return 0
  echo; echo "# SKIPPED (missing run): $1"; return 1
}

echo "############################################################################"
echo "# STAGE 0 - re-measure, with provenance, every FPS this branch would publish"
echo "#   Nothing on disk today is citable: 80 of 80 fps.csv / fps.json value files"
echo "#   have no provenance (scripts/perf/collect_fps.py --legacy-audit)."
echo "############################################################################"

# myscenes, the paired off/noncentral rows at the resolution the paper reports.
for pair in "tunnel:/workspace/gray/tmp/r4/tunnel_off:/workspace/gray/tmp/final/tunnel_noncentral" \
            "workshop:/workspace/gray/tmp/noncentral/fix15k_workshop:/workspace/gray/tmp/final/workshop_noncentral"; do
  scene=${pair%%:*}; rest=${pair#*:}
  off=${rest%%:*}; nc=${rest#*:}
  need_dir "$off" && emit "gray ${scene} off - provenanced FPS" \
"cd $WT && $ENV1 $GPUWAIT 1 14000 $PY scripts/perf/bench_fps.py \
-m $off --gpu 1 --repeats 3 --method gray --label ${scene}_off \
--out $WT/tmp/perf/myscenes/${scene}_off.perf.json" || true
  need_dir "$nc" && emit "gray ${scene} noncentral - provenanced FPS" \
"cd $WT && $ENV1 $GPUWAIT 1 14000 $PY scripts/perf/bench_fps.py \
-m $nc --gpu 1 --repeats 3 --method gray-noncentral --label ${scene}_noncentral \
--out $WT/tmp/perf/myscenes/${scene}_noncentral.perf.json" || true
done

# FullCircle rttpf: the track whose FPS column carries the five contention artefacts.
for scene in room1 room2 room3; do
  for suffix in "" "_off"; do
    run=$WT/out/fullcircle_rttpf/${scene}_refit_rttpf${suffix}
    need_dir "$run" && emit "gray FullCircle ${scene}${suffix} - provenanced FPS" \
"cd $WT && $ENV1 $GPUWAIT 1 17000 $PY scripts/perf/bench_fps.py \
-m $run --gpu 1 --repeats 3 --method gray${suffix:+-off} --label fc_${scene}${suffix} \
--out $WT/tmp/perf/fullcircle/${scene}${suffix}.perf.json" || true
  done
done

echo
echo "############################################################################"
echo "# STAGE 1 - the FoV sweep, ray-tracing arm (gray)"
echo "#   11 points: 7 equidistant-fisheye (60..175 deg) + 4 rectilinear controls"
echo "#   (60..120 deg). One checkpoint, one resolution, 30 fixed poses per point."
echo "#   Preview it with no GPU at all:"
echo "#     $PY scripts/perf/fov_sweep.py --plan -m $SWEEP_RUN"
echo "############################################################################"

need_dir "$SWEEP_RUN" && emit "FoV sweep - gray, 1024x1024, 30 poses" \
"cd $WT && $ENV1 $GPUWAIT 1 14000 $PY scripts/perf/fov_sweep.py \
-m $SWEEP_RUN --gpu 1 --width 1024 --height 1024 --n-views 30 --repeats 3 \
--out $WT/tmp/perf/fov_sweep_gray_1024" || true

need_dir "$SWEEP_RUN" && emit "FoV sweep - gray, 1920x1080 (resolution is not free: repeat it)" \
"cd $WT && $ENV1 $GPUWAIT 1 14000 $PY scripts/perf/fov_sweep.py \
-m $SWEEP_RUN --gpu 1 --width 1920 --height 1080 --n-views 30 --repeats 3 \
--out $WT/tmp/perf/fov_sweep_gray_1080p" || true

need_dir "$SWEEP_RUN" && emit "FoV sweep - gray, 512x512 (the small end: is the curve resolution-shaped?)" \
"cd $WT && $ENV1 $GPUWAIT 1 12000 $PY scripts/perf/fov_sweep.py \
-m $SWEEP_RUN --gpu 1 --width 512 --height 512 --n-views 30 --repeats 3 \
--out $WT/tmp/perf/fov_sweep_gray_512" || true

echo
echo "############################################################################"
echo "# STAGE 2 - transfer the SAME gaussians to the raster arm"
echo "#   convert/to_3dgrt.py moves tensors to .cuda(), so it needs a card: queue it."
echo "#   --match-kernel (DEFAULT TRUE) adds a log-scale offset so 3dgrut's generalized"
echo "#   gaussian reproduces gray's exp_power kernel. That CHANGES the scale tensor, so"
echo "#   it is not the identity transfer. Declare which you used; run both if the"
echo "#   footprint size turns out to matter for the raster cost (it will)."
echo "#   Its --template-checkpoint default (~/Desktop/3dgrut/...) does not exist here."
echo "############################################################################"

need_dir "$SWEEP_RUN" && emit "transfer tunnel_off -> 3dgrut checkpoint (kernel-matched)" \
"cd $WT && $ENV1 $GPUWAIT 1 8000 $PY convert/to_3dgrt.py $SWEEP_RUN \
$WT/tmp/perf/transfer/tunnel_off_3dgrt --downsample 1 \
--template-checkpoint <AN EXISTING 3dgrut ckpt: ls /workspace/3dgrut/worktrees/rttpf/out/*/*/ckpt_last.pt>" || true

need_dir "$SWEEP_RUN" && emit "transfer tunnel_off -> 3dgrut checkpoint (identity, no kernel match)" \
"cd $WT && $ENV1 $GPUWAIT 1 8000 $PY convert/to_3dgrt.py $SWEEP_RUN \
$WT/tmp/perf/transfer/tunnel_off_3dgrt_identity --downsample 1 --no-match-kernel \
--template-checkpoint <same>" || true

echo
echo "# CPU only, no card needed, run it now if you like:"
echo "#   cd $WT && $PY convert/to_3dgs.py $SWEEP_RUN $WT/tmp/perf/transfer/tunnel_off_3dgs"

echo
echo "############################################################################"
echo "# STAGE 3 - the FoV sweep, raster arm (3DGUT), same gaussians, same poses"
echo "#   3DGUT is the right rival: a RASTERISER that claims to support distorted"
echo "#   cameras, which is exactly the claim a FoV sweep tests. Its FThetaCamera"
echo "#   carries an explicit max_angle and is NOT capped at 90 deg, unlike gray"
echo "#   (cuda/core/opencv_fisheye.cuh:31) -- report that asymmetry, do not hide it."
echo "#   BLOCKED: scripts/perf/fov_sweep_3dgut.py does not exist yet. It needs a"
echo "#   synthetic-camera injection point in 3dgrut's dataset layer, which is a"
echo "#   different repo and therefore outside this task's write perimeter."
echo "#   See scripts/perf/README.md, 'What is still missing'."
echo "############################################################################"

echo
echo "############################################################################"
echo "# STAGE 4 - the confirmation pass on a modern RT core (Ampere / Ada)"
echo "#   Both cards on this machine are Turing sm_75: FIRST-generation RT cores"
echo "#   (2018). That is the single most attackable variable of a ray-tracing speed"
echo "#   claim, and no amount of care with pueue fixes it."
echo "#   Off-machine, not a pueue task: see README.md and the \`ccc\` skill."
echo "#   NOTE A100/H100 have NO RT cores at all -- the cluster must have RTX / L40S /"
echo "#   Ada nodes, or the confirmation pass measures software fallback."
echo "############################################################################"

echo
echo "############################################################################"
echo "# STAGE 5 - collect; the collector refuses anything unprovenanced"
echo "############################################################################"
echo
echo "cd $WT && $PY scripts/perf/collect_fps.py --roots tmp/perf --json-out tmp/perf/citable_rows.json"
echo "cd $WT && $PY scripts/perf/protocol.py --diff"
