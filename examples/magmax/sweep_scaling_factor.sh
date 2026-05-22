#!/usr/bin/env bash
# Sweep MagMax `scaling_factor` (alpha) on CLIP-ViT-B/32 TA8 to reproduce
# the alpha-sensitivity curve from the paper.
set -euo pipefail

for alpha in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
    fusion_bench \
        fabric.loggers.name=magmax/ViT-B-32_TA8/alpha_${alpha} \
        method=magmax/magmax \
        method.scaling_factor=${alpha} \
        modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8 \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8 \
        report_save_path=outputs/magmax/clip-vit-base-patch32_TA8_alpha_${alpha}.json
done
