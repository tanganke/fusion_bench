# the eight tasks
for hf_base_model_name in clip-vit-base-patch32 clip-vit-base-patch16 clip-vit-large-patch14; do
    for task_name in sun397 stanford-cars resisc45 eurosat svhn gtsrb mnist dtd; do
        if [ -d "./output/${hf_base_model_name}_${task_name}" ]; then
            echo "Skipping ${hf_base_model_name}_${task_name} because it already exists"
            continue
        fi
        python convert_openclip_model.py \
            --hf_base_model_name $hf_base_model_name \
            --task_name $task_name \
            --save_directory ./outputs/open-${hf_base_model_name}_${task_name}
    done
done
