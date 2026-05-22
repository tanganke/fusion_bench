#!/usr/bin/env bash
# Reproduce MagMax on the 14-task CLIP-ViT-B/32 vision benchmark (TALL14).
set -euo pipefail

fusion_bench \
    fabric.loggers.name=magmax/ViT-B-32_TALL14 \
    method=magmax/magmax \
    method.scaling_factor=0.5 \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TALL14 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL14 \
    report_save_path=outputs/magmax/clip-vit-base-patch32_TALL14.json
