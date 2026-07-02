set -e
for arg in "$@"
do
    models=("THIN_PRISM_FISHEYE")

    for model in ${models[@]}
    do

        SOURCE_DIR="${arg}"
        
        echo "Processing folder: $SOURCE_DIR with model: $model"
        # python run_colmap_fixed.py -s "$SOURCE_DIR" -c "$SOURCE_DIR/cameras.txt" --no-gpu
        python run_colmap.py -s "$SOURCE_DIR" --camera $model --no-gpu

        python colmap_bin_to_txt.py -s "$SOURCE_DIR"
        python colmap_bin_to_txt.py -s "$SOURCE_DIR/distorted/sparse/0"

        python resize.py -s "$SOURCE_DIR" -i input -y
        python resize.py -s "$SOURCE_DIR" -y
        python third_party/edgs.py -s "$SOURCE_DIR" -r 1 --roma-model indoors -y

        SCENE_BASENAME=$(basename "$SOURCE_DIR")
        OUT_DIR="out/${SCENE_BASENAME}_${model}"
        RESULT_PATH="$OUT_DIR/results.json"
        
        echo "Training with output directory: $OUT_DIR"
        python train.py -s "$SOURCE_DIR" -r 1 -m $OUT_DIR --batch_size 2 --eval --vignetting_comp --vignetting_terms 3 -y --camera_model ${model,,}
        python render.py -m $OUT_DIR --eval-models pinhole ${model,,} --intrinsics "$SOURCE_DIR/distorted/sparse/0/cameras.bin"
        python metrics.py -m $OUT_DIR
        python result_to_csv.py -t "$RESULT_PATH"

    done
done