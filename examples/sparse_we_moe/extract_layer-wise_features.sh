for TASK in sun397 stanford-cars resisc45 eurosat svhn gtsrb mnist dtd; do
    MODEL_PATH=tanganke/clip-vit-base-patch32_${TASK}
    fusion_bench \
        method=dummy \
        modelpool=CLIPVisionModelPool/clip-vit-base-patch32_individual \
        modelpool.models._pretrained_.pretrained_model_name_or_path=${MODEL_PATH} \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8 \
        taskpool.layer_wise_feature_save_path=/results/layer_wise_features/clip-vit-base-patch32_${TASK}/
done
