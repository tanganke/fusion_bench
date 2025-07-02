#!/bin/bash

# Merge CLIP-ViT-B/32 models on eight image classification tasks using TSVM
fusion_bench \
    fabric.loggers.name=tsvm \
    method=task_singular_vector/TaskSingularVectorMerging \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8

# Merge CLIP-ViT-L/14 models on eight image classification tasks using TSVM
fusion_bench \
    fabric.loggers.name=tsvm \
    method=task_singular_vector/TaskSingularVectorMerging \
    modelpool=CLIPVisionModelPool/clip-vit-large-patch14_TA8 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8_L14

# Merge 20 CLIP-VIT-B/32 models with TSVM.
fusion_bench \
    method=task_singular_vector/TaskSingularVectorMerging \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TALL20_model_only \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL20
