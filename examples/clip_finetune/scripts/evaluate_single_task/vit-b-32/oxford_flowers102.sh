# evaluate pretrained
fusion_bench --config-dir $PWD/config \
    method=dummy \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_individual \
    taskpool=clip-vit-classification_oxford_flowers102

# evaluate fine-tuned
fusion_bench --config-dir $PWD/config \
    method=dummy \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_individual \
    modelpool.base_model=tanganke/clip-vit-base-patch32_oxford_flowers102 \
    taskpool=clip-vit-classification_oxford_flowers102

# evaluate pretrained
fusion_bench --config-dir $PWD/config \
    method=dummy \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_individual \
    taskpool=clip-vit-classification_oxford_flowers102_val

# evaluate fine-tuned
fusion_bench --config-dir $PWD/config \
    method=dummy \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_individual \
    modelpool.base_model=tanganke/clip-vit-base-patch32_oxford_flowers102 \
    taskpool=clip-vit-classification_oxford_flowers102_val
