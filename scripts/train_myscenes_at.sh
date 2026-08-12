#!/bin/bash
# train_myscenes.sh with the resolution and iteration count exposed, for ablation sweeps.
#
# `train_myscenes.sh` pins -r 4 / 15000, which are the baseline settings and must stay
# pinned for anything that ends up in a table. Recipe sweeps run at -r 8 / 7500 instead
# (PROTOCOL.md), which needs those two to be arguments -- passing a second `-r` to the same
# script would leave tyro deciding which one wins.
#
# usage: train_myscenes_at.sh <scene> <model_path> <resolution> <iterations> [extra args...]
set -e
SCENE="$1"
MODEL_PATH="$2"
RESOLUTION="$3"
ITERATIONS="$4"
shift 4

case "$SCENE" in
    atrium|library|reception|tunnel) SOURCE="data/myscenes/${SCENE}_undistortion" ;;
    classroom|forest|workshop)       SOURCE="data/myscenes/${SCENE}" ;;
    *) echo "unknown myscenes scene: $SCENE" >&2; exit 1 ;;
esac

python train.py -m "$MODEL_PATH" -y -s "$SOURCE" -r "$RESOLUTION" --iterations "$ITERATIONS" \
    --camera_model rad_tan_thin_prism_fisheye \
    --batch_size 2 --eval --vignetting_comp --vignetting_terms 3 "$@"

bash scripts/eval_rttpf.sh "$MODEL_PATH" "$SOURCE"
