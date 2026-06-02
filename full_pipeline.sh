#!/bin/bash

# Initialisation des variables (valeurs par defaut)
ROMA_MODEL="indoors"
DOWNSAMPLING_LEVEL="4"

# 1. Analyse des arguments nommés
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)
            ROMA_MODEL="$2"
            shift 2 # On décale de 2 pour éliminer l'option et sa valeur
            ;;
        -d|--downscale)
            DOWNSAMPLING_LEVEL="$2"
            shift 2
            ;;
        -*)
            echo "Option inconnue : $1"
            echo "Usage: $0 -m [indoors|outdoors] -d [niveau] dossier1 dossier2 ..."
            exit 1
            ;;
        *)
            # Si l'argument ne commence pas par un tiret, on a atteint la liste des dossiers
            break
            ;;
    esac
done

# 2. Validation spécifique pour le modèle Roma
if [ "$ROMA_MODEL" != "indoors" ] && [ "$ROMA_MODEL" != "outdoors" ]; then
    echo "❌ Erreur : Le modèle Roma doit être 'indoors' ou 'outdoors'."
    exit 1
fi

# Vérification qu'il reste au moins un dossier à traiter
if [ $# -eq 0 ]; then
    echo "❌ Erreur : Vous devez spécifier au moins un dossier à traiter."
    exit 1
fi

echo "Configuration :"
echo "  - Modèle RoMa : $ROMA_MODEL"
echo "  - Downsampling : $DOWNSAMPLING_LEVEL"
echo "  - Nombre de dossiers : $#"
echo "--------------------------------------------------"

# 3. Boucle sur tous les arguments restants (les dossiers)
for arg in "$@"
do
    echo "=== Traitement du dossier : $arg ==="
    # echo "Execution du colmap pour $arg"
    # python run_colmap.py -s "$arg"

    echo "Resizing des images dans $arg"
    python resize.py -s "$arg" -y

    # Le modèle dynamique est appliqué ici
    echo "Running EDGS on $arg"
    python third_party/edgs.py -s "$arg" --roma_model "$ROMA_MODEL"

    # Le niveau de downsampling dynamique est appliqué ici
    OUTPUT_DIR="out/$(basename "$arg")_d${DOWNSAMPLING_LEVEL}"

    echo "Training 3DRT on $arg with downsampling level $DOWNSAMPLING_LEVEL"
    python train.py -m "$OUTPUT_DIR" -s "$arg" -r "$DOWNSAMPLING_LEVEL" 
    python render.py -m "$OUTPUT_DIR"
    python metrics.py -m "$OUTPUT_DIR"
    python measure_fps.py -m "$OUTPUT_DIR"

done