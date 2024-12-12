#!/bin/bash
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
# pre-trained model path
MODEL=openai/clip-vit-base-patch32
MODEL_SHORT_NAME=ViT-B-32
# TASKS: sun397 stanford_cars resisc45 eurosat svhn gtsrb mnist dtd
TASK=kmnist

# Full fine-tune CLIP-ViT-B/16:
function full_finetune() {
    fusion_bench \
        --config-dir ${SCRIPT_DIR}/config \
        method=clip_finetune \
        method.num_steps=4000 \
        method.learning_rate=1e-5 \
        modelpool=clip-finetune_${TASK} \
        modelpool.base_model=${MODEL} \
        fabric.loggers.root_dir=outputs/${MODEL_SHORT_NAME}/full_finetune \
        fabric.loggers.name=${TASK}
}

function lora_finetune() {
    fusion_bench \
        --config-dir ${SCRIPT_DIR}/config \
        method=clip_finetune \
        method.num_steps=4000 \
        method.learning_rate=1e-5 \
        method.use_lora=true \
        modelpool=clip-finetune_${TASK} \
        modelpool.models.0.path=${MODEL} \
        fabric.loggers.root_dir=outputs/${MODEL_SHORT_NAME}/lora_finetune \
        fabric.loggers.name=${TASK}
}

function l_lora_finetune() {
    fusion_bench \
        --config-dir ${SCRIPT_DIR}/config \
        method=clip_finetune \
        method.num_steps=4000 \
        method.learning_rate=1e-5 \
        method.use_lora=true \
        method.use_l_lora=true \
        modelpool=clip-finetune_${TASK} \
        modelpool.models.0.path=${MODEL} \
        fabric.loggers.root_dir=outputs/${MODEL_SHORT_NAME}/l_lora_finetune \
        fabric.loggers.name=${TASK}
}

full_finetune
