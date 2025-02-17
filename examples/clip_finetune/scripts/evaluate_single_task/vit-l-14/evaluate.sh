TASK=fer2013

# evaluate pretrained
fusion_bench --config-dir $PWD/config \
    method=dummy \
    modelpool=CLIPVisionModelPool/clip-vit-large-patch14_individual \
    taskpool=clip-vit-classification_${TASK} \
    taskpool.base_model=openai/clip-vit-large-patch14

# evaluate fine-tuned
fusion_bench --config-dir $PWD/config \
    method=dummy \
    modelpool=CLIPVisionModelPool/clip-vit-large-patch14_individual \
    modelpool.base_model=./tanganke/clip-vit-large-patch14_${TASK} \
    taskpool=clip-vit-classification_${TASK} \
    taskpool.base_model=openai/clip-vit-large-patch14

