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
        method=classification/clip_finetune \
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
        method=classification/clip_finetune \
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
        method=classification/clip_finetune \
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


function continual_full_finetune_8_tasks() {
    fusion_bench \
        --config-dir ${SCRIPT_DIR}/config \
        fabric.devices=4 \
        fabric.loggers.root_dir=outputs/${MODEL_SHORT_NAME}/continual_full_finetune \
        fabric.loggers.name=TA8 \
        method=classification/clip_continual_finetune \
        method.num_steps=4000 \
        method.batch_size=32 \
        method.save_interval=2000 \
        method.learning_rate=1e-5 \
        method.shuffle_order=true \
        modelpool=clip-finetune_TA8 \
        modelpool.base_model=${MODEL}
}

function continual_full_finetune_14_tasks() {
    fusion_bench \
        --config-dir ${SCRIPT_DIR}/config \
        fabric.devices=4 \
        fabric.loggers.root_dir=outputs/${MODEL_SHORT_NAME}/continual_full_finetune \
        fabric.loggers.name=TALL14 \
        method=classification/clip_continual_finetune \
        method.num_steps=4000 \
        method.batch_size=32 \
        method.save_interval=2000 \
        method.learning_rate=1e-5 \
        method.shuffle_order=true \
        modelpool=clip-finetune_TALL14 \
        modelpool.base_model=${MODEL}
}


function continual_full_finetune_20_tasks() {
    fusion_bench \
        --config-dir ${SCRIPT_DIR}/config \
        fabric.devices=4 \
        fabric.loggers.root_dir=outputs/${MODEL_SHORT_NAME}/continual_full_finetune \
        fabric.loggers.name=TALL20 \
        method=classification/clip_continual_finetune \
        method.num_steps=4000 \
        method.batch_size=32 \
        method.save_interval=2000 \
        method.learning_rate=1e-5 \
        method.shuffle_order=true \
        modelpool=clip-finetune_TALL20 \
        modelpool.base_model=${MODEL}
}

for version in 0 1 2 3 4 5 6 7 8 9; do
    continual_full_finetune_8_tasks
    continual_full_finetune_14_tasks
    continual_full_finetune_20_tasks
done
