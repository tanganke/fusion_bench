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
    method=surgery/adamerging_surgery \
    modelpool=CLIPVisionModelPool/clip-vit-large-patch14_TA8 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8_L14
