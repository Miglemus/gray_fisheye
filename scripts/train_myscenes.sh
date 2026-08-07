#!/bin/bash
# Train + render + score one myscenes scene with the settings the baselines use.
#
# The 7 myscenes live under two different layouts -- atrium/library/reception/tunnel are
# `<scene>_undistortion`, classroom/forest/workshop are just `<scene>` -- which this
# resolves. It also pins vignetting_comp / vignetting_terms 3 / batch_size 2 for every
# scene: the published gray `workshop` baseline was the only one run without them
# (vignetting_comp=False, batch_size=1) and that alone cost it 0.71 dB.
#
# usage: train_myscenes.sh <scene> <model_path> [extra train.py args...]
set -e
SCENE="$1"
MODEL_PATH="$2"
shift 2

case "$SCENE" in
    atrium|library|reception|tunnel) SOURCE="data/myscenes/${SCENE}_undistortion" ;;
    classroom|forest|workshop)       SOURCE="data/myscenes/${SCENE}" ;;
    *) echo "unknown myscenes scene: $SCENE" >&2; exit 1 ;;
esac

python train.py -m "$MODEL_PATH" -y -s "$SOURCE" -r 4 \
    --camera_model rad_tan_thin_prism_fisheye \
    --batch_size 2 --eval --vignetting_comp --vignetting_terms 3 "$@"

bash scripts/eval_rttpf.sh "$MODEL_PATH" "$SOURCE"
