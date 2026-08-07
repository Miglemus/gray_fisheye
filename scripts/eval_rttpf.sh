#!/bin/bash
# Render + score a trained myscenes rttpf run.
#
# Why this exists: run.sh calls `render.py -m <model>` with no flags, and render.py's
# eval_models defaults to ["pinhole"] (render.py:26-29). A plain run.sh therefore never
# produces the rad_tan_thin_prism_fisheye renders that scripts/masked_eval.py globs, so the
# cross-method table cannot be built from it. Always go through this script.
#
# usage: eval_rttpf.sh <model_path> <source_path>
set -e
MODEL_PATH="$1"
SOURCE_PATH="$2"
python render.py -m "$MODEL_PATH" \
    --eval-models pinhole rad_tan_thin_prism_fisheye \
    --intrinsics "$SOURCE_PATH/distorted/sparse/0/cameras.bin"
python metrics.py -m "$MODEL_PATH"
python measure_fps.py -m "$MODEL_PATH"
