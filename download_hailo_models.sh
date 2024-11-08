#!/bin/bash

# Directories where models will be saved
MODEL_DIR_HAILO8="models/hailo8"
MODEL_DIR_HAILO8L="models/hailo8l"

# Base URLs for each model set
BASE_URL_HAILO8="https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.13.0/hailo8"
BASE_URL_HAILO8L="https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.13.0/hailo8l"

# Model files for hailo8
models_hailo8=(
    "yolov10b.hef"
    "yolov10n.hef"
    "yolov10s.hef"
    "yolov10x.hef"
    "yolov5m.hef"
    "yolov5s.hef"
    "yolov8l.hef"
    "yolov8m.hef"
    "yolov8n.hef"
    "yolov8s.hef"
    "yolov8x.hef"
)

# Model files for hailo8l
models_hailo8l=models_hailo8

# Flags to control which models to download
download_hailo8=false
download_hailo8l=false

# Parse flags
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --hailo8) download_hailo8=true ;;
        --hailo8l) download_hailo8l=true ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

# If no flags are provided, download both
if ! $download_hailo8 && ! $download_hailo8l; then
    download_hailo8=true
    download_hailo8l=true
fi

# Function to download models
download_models() {
    local model_dir="$1"
    local base_url="$2"
    local -n models="$3"

    # Ensure the target directory exists
    mkdir -p "$model_dir"

    # Download each model in the list
    for model in "${models[@]}"; do
        echo "Downloading $model to $model_dir..."
        wget -q -O "$model_dir/$model" "$base_url/$model"
        if [[ $? -eq 0 ]]; then
            echo "$model downloaded successfully."
        else
            echo "Error downloading $model."
        fi
    done
}

# Download hailo8 models if specified
if $download_hailo8; then
    download_models "$MODEL_DIR_HAILO8" "$BASE_URL_HAILO8" models_hailo8
fi

# Download hailo8l models if specified
if $download_hailo8l; then
    download_models "$MODEL_DIR_HAILO8L" "$BASE_URL_HAILO8L" models_hailo8l
fi

echo "Download process completed."
