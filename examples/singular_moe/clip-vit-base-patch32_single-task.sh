# project into different subspaces
for task in sun397 stanford-cars resisc45 eurosat svhn gtsrb mnist dtd
do
    CUDA_VISIBLE_DEVICES=5 fusion_bench \
        method=singular_projection_merging \
            method.device=cuda method.rank=low method.k=-1 method.full_matrices=false \
        modelpool=clip-vit-base-patch32_single_finetuned \
            modelpool.models.1.name=${task} \
            modelpool.models.1.path=tanganke/clip-vit-base-patch32_${task} \
        taskpool=clip-vit-classification_TA8.local \
        save_report="outputs/ViT-B-32/single-task/projection_merging_zone1_${task}.json" &

    CUDA_VISIBLE_DEVICES=6 fusion_bench \
        method=singular_projection_merging \
            method.device=cuda method.rank=high method.k=-1 method.full_matrices=false \
        modelpool=clip-vit-base-patch32_single_finetuned \
            modelpool.models.1.name=${task} \
            modelpool.models.1.path=tanganke/clip-vit-base-patch32_${task} \
        taskpool=clip-vit-classification_TA8.local \
        save_report="outputs/ViT-B-32/single-task/projection_merging_zone2_${task}.json" &

    CUDA_VISIBLE_DEVICES=7 fusion_bench \
        method=singular_projection_merging \
            method.device=cuda method.rank=high method.k=-1 method.full_matrices=true \
        modelpool=clip-vit-base-patch32_single_finetuned \
            modelpool.models.1.name=${task} \
            modelpool.models.1.path=tanganke/clip-vit-base-patch32_${task} \
        taskpool=clip-vit-classification_TA8.local \
        save_report="outputs/ViT-B-32/single-task/projection_merging_zone23_${task}.json" &
    wait
done

# evaluate singlue fine-tuned models
for task in sun397 stanford-cars resisc45 eurosat svhn gtsrb mnist dtd
do
    fusion_bench method=dummy \
        modelpool=clip-vit-base-patch32_individual \
            modelpool.models.0.path=tanganke/clip-vit-base-patch32_${task} \
        taskpool=clip-vit-classification_TA8.local \
        save_report="outputs/ViT-B-32/single-task/clip-vit-base-patch32_${task}.json"
done
