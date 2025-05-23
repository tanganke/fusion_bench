#!/usr/bin/env bash
debug=0

for tall_mask_lambda in 0.2 0.3 0.4 0.5 0.6; do
    fusion_bench \
        fabric.loggers.name=tall_mask/ViT-B-32 \
        method=tall_mask/task_arithmetic \
        method.tall_mask_lambda=$tall_mask_lambda \
        method.debug=$debug \
        method.verbose=0 \
        modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8 \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8
done
