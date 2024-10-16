for model in openai/clip-vit-base-patch32 tanganke/clip-vit-base-patch32_sun397 tanganke/clip-vit-base-patch32_stanford-cars tanganke/clip-vit-base-patch32_resisc45 tanganke/clip-vit-base-patch32_eurosat tanganke/clip-vit-base-patch32_svhn tanganke/clip-vit-base-patch32_gtsrb tanganke/clip-vit-base-patch32_mnist tanganke/clip-vit-base-patch32_dtd
do
    model_name=$(echo $model | tr '/' '_')
    fusion_bench \
        method=dummy \
        modelpool=clip-vit-base-patch32_individual \
            modelpool.models.0.path=$model \
        taskpool=clip-vit-classification_TA8 \
        report_save_path=outputs/clip-vit-base-patch32_individual_${model_name}_dummy.json
done

for model in openai/clip-vit-large-patch14 tanganke/clip-vit-large-patch14_sun397 tanganke/clip-vit-large-patch14_stanford-cars tanganke/clip-vit-large-patch14_resisc45 tanganke/clip-vit-large-patch14_eurosat tanganke/clip-vit-large-patch14_svhn tanganke/clip-vit-large-patch14_gtsrb tanganke/clip-vit-large-patch14_mnist tanganke/clip-vit-large-patch14_dtd
do
    model_name=$(echo $model | tr '/' '_')
    fusion_bench \
        method=dummy \
        modelpool=clip-vit-large-patch14_individual \
            modelpool.models.0.path=$model \
        taskpool=clip-vit-classification_TA8 \
            taskpool.clip_model=openai/clip-vit-large-patch14 \
        report_save_path=outputs/clip-vit-large-patch14_individual_${model_name}_dummy.json
done
