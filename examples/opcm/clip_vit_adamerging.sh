# This is the script for running the continual layer-wise adamerging using the ViT-B/32 models.
for i in {1..6}; do
    OUTPUT_DIR=continual_clip_layer_wise_adamerging_adamerging/${i}
    fusion_bench \
        --config-dir ${PWD}/config \
        merged_model_save_path=outputs/logs/ViT-B-32/${OUTPUT_DIR}/version_${0}/merged_model \
        fabric.loggers.version=0 \
        method=adamerging/clip \
        method.name=clip_layer_wise_adamerging \
        method.save_merging_weights=merging_weights.pt \
        modelpool=clip-vit-base-patch32-round_${i} \
        fabric.loggers.root_dir=outputs/logs/ViT-B-32 \
        fabric.loggers.name=${OUTPUT_DIR}
done

fusion_bench method=dummy \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_individual \
    modelpool.models._pretrained_.pretrained_model_name_or_path=./outputs/logs/ViT-B-32/continual_clip_layer_wise_adamerging_adamerging/6/version_0/merged_model \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8
