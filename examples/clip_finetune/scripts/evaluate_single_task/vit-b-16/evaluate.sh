TASK=rendered-sst2

# evaluate pretrained
fusion_bench --config-dir $PWD/config \
    method=dummy \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch16_individual \
    taskpool=clip-vit-classification_${TASK} \
    taskpool.base_model=openai/clip-vit-base-patch16

# evaluate fine-tuned
fusion_bench --config-dir $PWD/config \
    method=dummy \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch16_individual \
    modelpool.base_model=./tanganke/clip-vit-base-patch16_${TASK} \
    taskpool=clip-vit-classification_${TASK} \
    taskpool.base_model=openai/clip-vit-base-patch16

