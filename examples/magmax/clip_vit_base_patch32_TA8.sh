#!/usr/bin/env bash
# Reproduce MagMax (Marczak et al., ECCV 2024) on the 8-task CLIP-ViT-B/32
# vision benchmark used by Task Arithmetic.
set -euo pipefail

fusion_bench \
    fabric.loggers.name=magmax/ViT-B-32_TA8 \
    method=magmax/magmax \
    method.scaling_factor=0.5 \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8 \
    report_save_path=outputs/magmax/clip-vit-base-patch32_TA8.json
