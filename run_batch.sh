set -e
for arg in "$@"
do
    for model in "THIN_PRISM_FISHEYE" "OPENCV_FISHEYE"
    do
        if [ "$model" == "THIN_PRISM_FISHEYE" ]; then
            echo "Running with THIN_PRISM_FISHEYE camera model"
            EXTENSION="_tpf"
        else
            echo "Running with OPENCV_FISHEYE camera model"
            EXTENSION="_ocv"
        fi
        SOURCE_DIR="${arg}${EXTENSION}"
        
        # echo "Processing folder: $SOURCE_DIR with model: $model"
        # python run_colmap.py -s "$SOURCE_DIR" --camera $model

        python colmap_bin_to_txt.py -s "$SOURCE_DIR"
        python colmap_bin_to_txt.py -s "$SOURCE_DIR/distorted/sparse/0"

        # python resize.py -s "$SOURCE_DIR" -i input -y
        # python resize.py -s "$SOURCE_DIR" -y
        # python third_party/edgs.py -s "$SOURCE_DIR" -r 1 --roma-model indoors -y

        SCENE_BASENAME=$(basename "$SOURCE_DIR")
        OUT_DIR="out/$SCENE_BASENAME"
        RESULT_PATH="$OUT_DIR/results.json"
        # echo "Training with output directory: $OUT_DIR"
        # python train.py -s "$SOURCE_DIR" -r 1 --camera_model "${model,,}" -m $OUT_DIR --batch_size 2 --eval --vignetting_comp --vignetting_terms 3 
        # python result_to_csv.py -t "$RESULT_PATH"
        # python render.py -m $OUT_DIR --eval-models "${model,,}"
        # python metrics.py -m $OUT_DIR
    done
done