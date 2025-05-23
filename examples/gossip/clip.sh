# Layer-wise gossip
fusion_bench \
    method=gossip/layer_wise_clip \
    method.lr=1e-3 \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8  \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8
