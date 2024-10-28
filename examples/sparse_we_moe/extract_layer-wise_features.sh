# This script is used to extract layer-wise features of individual fine-tuned clip vit models on different tasks.
for TASK in sun397 stanford-cars resisc45 eurosat svhn gtsrb mnist dtd; do
    MODEL_PATH=tanganke/clip-vit-base-patch32_${TASK}
    fusion_bench \
        method=dummy \
        modelpool=CLIPVisionModelPool/clip-vit-base-patch32_individual \
        modelpool.models._pretrained_.pretrained_model_name_or_path=${MODEL_PATH} \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8 \
        taskpool.dataloader_kwargs.shuffle=true \
        taskpool.layer_wise_feature_save_path=outputs/layer_wise_features/clip-vit-base-patch32_${TASK} \
        taskpool.layer_wise_feature_first_token_only=true \
        taskpool.layer_wise_feature_max_num=1000
done

# extract layer-wise routing weights of the sparse we-moe model
fusion_bench \
    method=wemoe/sparse_weight_ensembling_moe \
    method.name=sparse_clip_weight_ensembling_moe \
    method.shared_gate=false \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8 \
    taskpool=CLIPVisionModelTaskPool/clip_sparse_wemoe_clip-vit-classification_TA8 \
    taskpool.dataloader_kwargs.shuffle=true \
    taskpool.layer_wise_routing_weights_save_path=outputs/layer_wise_routing_weights/clip-vit-base-patch32_TA8/ \
    taskpool.layer_wise_routing_weights_max_num=1000

# add option:
#    method.checkpoint=outputs/sparse_we_moe/clip-vit-base-patch32/clip-vit-base-patch32_TA8_sparse_we_moe_checkpoint.ckpt \
# to skip the training process
fusion_bench \
    method=wemoe/sparse_weight_ensembling_moe \
    method.name=sparse_clip_weight_ensembling_moe \
    method.shared_gate=true \
    method.save_checkpoint=outputs/sparse_we_moe/clip-vit-base-patch32/clip-vit-base-patch32_TA8_sparse_we_moe_checkpoint.ckpt \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8 \
    taskpool=CLIPVisionModelTaskPool/clip_sparse_wemoe_clip-vit-classification_TA8 \
    taskpool.dataloader_kwargs.shuffle=true \
    taskpool.layer_wise_routing_weights_save_path=outputs/layer_wise_routing_weights/clip-vit-base-patch32_TA8/ \
    taskpool.layer_wise_routing_weights_max_num=1000
