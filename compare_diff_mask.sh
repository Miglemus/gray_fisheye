RESULTS_FOLDER="comp_results"

mkdir -p $RESULTS_FOLDER

for arg in "$@"
do
    folder_name=$(basename "$arg")

    for rad in 0.9 0.95 1.0
    do

        python render.py -m "$arg" --eval-models thin_prism_fisheye --fisheye_mask_radius_scale "$rad"
        python metrics.py -m "$arg"

        # Copie avec le nouveau format de nom
        cp "$arg"/results.json "./$RESULTS_FOLDER/result_${folder_name}_${rad}.json"

    done
done