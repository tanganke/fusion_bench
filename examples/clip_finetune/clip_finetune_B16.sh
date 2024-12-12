#!/bin/bash
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
# pre-trained model path
MODEL=openai/clip-vit-base-patch16
MODEL_SHORT_NAME=ViT-B-16
# TASKS: sun397 stanford_cars resisc45 eurosat svhn gtsrb mnist dtd

# Full fine-tune CLIP-ViT-B/16:
function full_finetune() {
    fusion_bench \
        --config-dir ${SCRIPT_DIR}/config \
        fabric.devices=8 \
        method=clip_finetune \
        method.num_steps=4000 \
        method.batch_size=16 \
        method.save_interval=2000 \
        method.learning_rate=1e-5 \
        modelpool=clip-finetune_${TASK} \
        modelpool.base_model=${MODEL} \
        fabric.loggers.root_dir=outputs/${MODEL_SHORT_NAME}/full_finetune \
        fabric.loggers.name=${TASK}
}

for TASK in oxford_flowers102 pcam fer2013 oxford-iiit-pet stl10 food101 fashion_mnist emnist_letters kmnist rendered-sst2
do
    full_finetune
done
