#! /usr/bin/env bash
# In this script, we evaluate the performance of CLIP models on single tasks.
# Including ViT-B-32, ViT-B-16, ViT-L-14.

# Pre-trained ViT-B-32
fusion_bench \
    method=dummy \
    fabric.loggers.root_dir=outputs/single_task_evaluation \
    fabric.loggers.name=vit-b-32 \
    fabric.loggers.version=pretrained \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_individual \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL20

# Single-task evaluation of ViT-B-32
for TASK in sun397 stanford-cars resisc45 eurosat svhn gtsrb mnist dtd oxford_flowers102 pcam fer2013 oxford-iiit-pet stl10 cifar100 cifar10 food101 fashion_mnist emnist_letters kmnist rendered-sst2; do
    # if outputs/single_task_evaluation/vit-b-32/${TASK}/report.json exists, skip
    if [ -f outputs/single_task_evaluation/vit-b-32/${TASK}/report.json ]; then
        echo "Skipping ${TASK} because report.json already exists"
        continue
    fi
    fusion_bench \
        method=dummy \
        fabric.loggers.root_dir=outputs/single_task_evaluation \
        fabric.loggers.name=vit-b-32 \
        fabric.loggers.version=${TASK} \
        modelpool=CLIPVisionModelPool/clip-vit-base-patch32_individual \
        modelpool.base_model=tanganke/clip-vit-base-patch32_${TASK} \
        taskpool=CLIPVisionModelTaskPool/clip-vit-single-task_${TASK}
done


# Pre-trained ViT-B-16
fusion_bench \
    method=dummy \
    fabric.loggers.root_dir=outputs/single_task_evaluation \
    fabric.loggers.name=vit-b-16 \
    fabric.loggers.version=pretrained \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch16_individual \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL20 \
    taskpool.base_model=openai/clip-vit-base-patch16

# Single-task evaluation of ViT-B-16
for TASK in sun397 stanford-cars resisc45 eurosat svhn gtsrb mnist dtd oxford_flowers102 pcam fer2013 oxford-iiit-pet stl10 cifar100 cifar10 food101 fashion_mnist emnist_letters kmnist rendered-sst2; do
    # if outputs/single_task_evaluation/vit-b-16/${TASK}/report.json exists, skip
    if [ -f outputs/single_task_evaluation/vit-b-16/${TASK}/report.json ]; then
        echo "Skipping ${TASK} because report.json already exists"
        continue
    fi
    fusion_bench \
        method=dummy \
        fabric.loggers.root_dir=outputs/single_task_evaluation \
        fabric.loggers.name=vit-b-16 \
        fabric.loggers.version=${TASK} \
        modelpool=CLIPVisionModelPool/clip-vit-base-patch16_individual \
        modelpool.base_model=tanganke/clip-vit-base-patch16_${TASK} \
        taskpool=CLIPVisionModelTaskPool/clip-vit-single-task_${TASK} \
        taskpool.base_model=openai/clip-vit-base-patch16
done


# Pre-trained ViT-L-14
fusion_bench \
    method=dummy \
    fabric.loggers.root_dir=outputs/single_task_evaluation \
    fabric.loggers.name=vit-l-14 \
    fabric.loggers.version=pretrained \
    modelpool=CLIPVisionModelPool/clip-vit-large-patch14_individual \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL20 \
    taskpool.base_model=openai/clip-vit-large-patch14

# Single-task evaluation of ViT-L-14
for TASK in sun397 stanford-cars resisc45 eurosat svhn gtsrb mnist dtd oxford_flowers102 pcam fer2013 oxford-iiit-pet stl10 cifar100 cifar10 food101 fashion_mnist emnist_letters kmnist rendered-sst2; do
    # if outputs/single_task_evaluation/vit-l-14/${TASK}/report.json exists, skip
    if [ -f outputs/single_task_evaluation/vit-l-14/${TASK}/report.json ]; then
        echo "Skipping ${TASK} because report.json already exists"
        continue
    fi
    fusion_bench \
        method=dummy \
        fabric.loggers.root_dir=outputs/single_task_evaluation \
        fabric.loggers.name=vit-l-14 \
        fabric.loggers.version=${TASK} \
        modelpool=CLIPVisionModelPool/clip-vit-large-patch14_individual \
        modelpool.base_model=tanganke/clip-vit-large-patch14_${TASK} \
        taskpool=CLIPVisionModelTaskPool/clip-vit-single-task_${TASK} \
        taskpool.base_model=openai/clip-vit-large-patch14
done
