#!/usr/bin/env bash
# Reproduce MagMax on the 8-task CLIP-ViT-B/16 vision benchmark.
set -euo pipefail

fusion_bench \
    fabric.loggers.name=magmax/ViT-B-16_TA8 \
    method=magmax/magmax \
    method.scaling_factor=0.5 \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch16_TA8 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8 \
    report_save_path=outputs/magmax/clip-vit-base-patch16_TA8.json
