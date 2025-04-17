#!/bin/bash

# Merge CLIP-ViT-B/32 models on eight image classification tasks
fusion_bench \
    fabric.loggers.name=adamerging_surgery \
    method=surgery/adamerging_surgery \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8

# Merge CLIP-ViT-L/14 models on eight image classification tasks
fusion_bench \
    fabric.loggers.name=adamerging_surgery \
    fabric.devices=2 \
    method=surgery/adamerging_surgery \
    method.batch_size=8 \
    method.weights=/data0/users/tanganke/projects/smile_upscaling/outputs/logs/adamerging_surgery/version_3/merging_weights.pt \
    modelpool=CLIPVisionModelPool/clip-vit-large-patch14_TA8 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8_L14
