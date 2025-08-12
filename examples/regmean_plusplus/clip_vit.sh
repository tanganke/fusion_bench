#!/bin/bash

# Merge CLIP-ViT-B/32 models on eight image classification tasks
fusion_bench \
    fabric.loggers.name="ViT-B-32/clip_regmean_plusplus" \
    method=regmean_plusplus/clip_regmean_plusplus \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8 \
        taskpool.base_model=openai/clip-vit-base-patch32

# Merge CLIP-ViT-B/16 models on eight image classification tasks
fusion_bench \
    fabric.loggers.name="ViT-B-16/clip_regmean_plusplus" \
    method=regmean_plusplus/clip_regmean_plusplus \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch16_TA8 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8 \
        taskpool.base_model=openai/clip-vit-base-patch16

# Merge CLIP-ViT-L/14 models on eight image classification tasks
fusion_bench \
    fabric.loggers.name="ViT-L-14/clip_regmean_plusplus" \
    method=regmean_plusplus/clip_regmean_plusplus \
    modelpool=CLIPVisionModelPool/clip-vit-large-patch14_TA8 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8 \
        taskpool.base_model=openai/clip-vit-large-patch14
