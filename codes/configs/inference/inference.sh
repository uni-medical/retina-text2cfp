#!/bin/bash

# Configuration paths
BASE_DIR="path/to/your/repo/codes"
SAVE_PATH="${BASE_DIR}/results/inference_results/"
CAPTION_BASE="${BASE_DIR}/configs/inference"

# Inference parameters
COMMON_PARAMS="--num_sampling_steps 50 \
    --samples_per_caption 1 \
    --resolution 512:512x512 1024:1024x1024 \
    --time_shifting_factor 1.0 \
    --batch_size 16 \
    --num_gpus 2" 

# GPU configuration
DEVICES="0,1"

# Caption files mapping
declare -A CAPTION_FILES=(  
    ["example"]="example_captions.txt" 
)

# Model checkpoint paths
declare -A CKPT_BASE_PATHS=(
    ["model_v1"]="${BASE_DIR}/results/model_chkpt/checkpoints"
)

# Available checkpoints
declare -A CHECKPOINTS=(
    ["model_v1"]="checkpoint_name"
)

extract_folder_name() {
    local ckpt_base=$1
    echo "$(basename "$(dirname "$ckpt_base")")"
}

run_inference() {
    local ckpt_base=$1
    local steps=$2
    local caption_name=$3
    local caption_file=${CAPTION_FILES[$caption_name]}
    local job_id="inference_job"
    local folder_name=$(extract_folder_name "$ckpt_base")
    local ckpt_path="${ckpt_base}/${steps}"
    echo "ckpt_path: ${ckpt_path}"

    CUDA_VISIBLE_DEVICES=${DEVICES} python sample_from_seeds_multigpus.py \
        --id "${folder_name}_${steps}steps_${caption_name}_${job_id}" \
        --ckpt "${ckpt_path}" \
        --image_save_path "${SAVE_PATH}" \
        --caption_path "${CAPTION_BASE}/${caption_file}" \
        ${COMMON_PARAMS}
}

# Main execution
for stage in "${!CKPT_BASE_PATHS[@]}"; do
    CKPT_BASE=${CKPT_BASE_PATHS[$stage]}
    echo "Processing stage: ${stage}"
    echo "Checkpoint base path: ${CKPT_BASE}"
    echo "------------------------"

    for steps in ${CHECKPOINTS[$stage]}; do
        echo "Processing checkpoint ${steps}"
        
        for caption_name in "${!CAPTION_FILES[@]}"; do
            echo "Running inference for ${caption_name}"
            run_inference "${CKPT_BASE}" "${steps}" "${caption_name}"
            echo ""
        done
    done
    
    echo "------------------------"
done