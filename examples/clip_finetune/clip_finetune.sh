#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# pre-trained model path
MODEL=/data0/users/tanganke/data/huggingface_models/openai/clip-vit-base-patch16
MODEL_SHORT_NAME=ViT-B-16
# TASKS: sun397 stanford_cars resisc45 eurosat svhn gtsrb mnist dtd

# Full fine-tune CLIP-ViT-B/16:
function full_finetune(){   
    for TASK in sun397 stanford_cars resisc45 eurosat svhn gtsrb mnist dtd; do
    fusion_bench \
        --config-dir ${SCRIPT_DIR}/config \
        method=clip_finetune \
        method.num_steps=2000 \
        method.learning_rate=1e-5 \
        modelpool=clip-finetune_${TASK} \
        modelpool.models.0.path=${MODEL} \
        taskpool=clip-vit-classification_TA8 \
        taskpool.clip_model=${MODEL} \
        fabric_logger.root_dir=outputs/${MODEL_SHORT_NAME}/full_finetune \
        fabric_logger.name=${TASK} \
        report_save_path=outputs/${MODEL_SHORT_NAME}/full_finetune_${TASK}.json
    done
}

function lora_finetune(){
    for TASK in sun397 stanford_cars resisc45 eurosat svhn gtsrb mnist dtd; do
    fusion_bench \
        --config-dir ${SCRIPT_DIR}/config \
        method=clip_finetune \
        method.num_steps=2000 \
        method.learning_rate=1e-5 \
        method.use_lora=true \
        modelpool=clip-finetune_${TASK} \
        modelpool.models.0.path=${MODEL} \
        taskpool=clip-vit-classification_TA8 \
        taskpool.clip_model=${MODEL} \
        fabric_logger.root_dir=outputs/${MODEL_SHORT_NAME}/lora_finetune \
        fabric_logger.name=${TASK} \
        report_save_path=outputs/${MODEL_SHORT_NAME}/lora_finetune_${TASK}.json
    done
}

function l_lora_finetune(){
    for TASK in sun397 stanford_cars resisc45 eurosat svhn gtsrb mnist dtd; do
    fusion_bench \
        --config-dir ${SCRIPT_DIR}/config \
        method=clip_finetune \
        method.num_steps=2000 \
        method.learning_rate=1e-5 \
        method.use_lora=true \
        method.use_l_lora=true \
        modelpool=clip-finetune_${TASK} \
        modelpool.models.0.path=${MODEL} \
        taskpool=clip-vit-classification_TA8 \
        taskpool.clip_model=${MODEL} \
        fabric_logger.root_dir=outputs/${MODEL_SHORT_NAME}/l_lora_finetune \
        fabric_logger.name=${TASK} \
        report_save_path=outputs/${MODEL_SHORT_NAME}/l_lora_finetune_${TASK}.json
    done
}

full_finetune
