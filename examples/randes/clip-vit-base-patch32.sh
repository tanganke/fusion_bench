fusion_bench \
    fabric.loggers.name=randes_modelsoup/ViT-B-32_TA8 \
    method=randes/superposed_model_soup \
    method.mode=identity_matrix \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8_model_only \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8
