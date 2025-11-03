#! /usr/bin/env bash
function finetune() {
    local output_dir="outputs/${model}/${dataset}"
    local version="batch_size_${batch_size}_lr_${lr}_training_data_ratio_${training_data_ratio}"
    if [ -d "$output_dir/$version" ]; then
        echo "$output_dir/$version exists, skip finetuning."
        continue
    fi
    rich-run fusion_bench --config-name model_fusion \
        path.log_dir=${output_dir}/$version seed=0 \
        method=classification/image_classification_finetune \
            method.max_epochs=-1 \
            method.max_steps=4000 \
            method.save_top_k=-1 \
            method.save_interval=1000 \
            method.save_on_train_epoch_end=false \
            method.training_data_ratio=${training_data_ratio} \
            method.optimizer.lr=${lr} \
            method.dataloader_kwargs.batch_size=${batch_size} \
        modelpool=ConvNextForImageClassification/${model} \
        modelpool.models._pretrained_.dataset_name=${dataset} \
        +dataset/image_classification/train@modelpool.train_datasets=${dataset} \
        +dataset/image_classification/test@modelpool.val_datasets=${dataset}
}

training_data_ratio=1.0
for model in convnext-base-224; do
    for dataset in sun397 stanford-cars resisc45 eurosat svhn gtsrb mnist dtd oxford_flowers102 pcam fer2013 oxford-iiit-pet stl10 cifar100 cifar10 food101 fashion_mnist emnist_letters kmnist rendered-sst2; do
        device_idx=0
        for batch_size in 64 128 256; do
            for lr in 0.005 0.001; do
                CUDA_VISIBLE_DEVICES=${device_idx} finetune &
                device_idx=$((device_idx + 1))
            done
        done
        wait
    done
done
