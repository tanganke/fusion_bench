#! /usr/bin/env bash
function finetune() {
    for batch_size in 64 128 256; do
        for lr in 0.01 0.005 0.001; do
            for training_data_ratio in 0.5 0.8 1.0; do
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
                        method.lr_scheduler=null \
                        method.dataloader_kwargs.batch_size=${batch_size} \
                    modelpool=ResNetForImageClassification/transformers/${model}_${dataset}
            done
        done
    done
}

for model in resnet18 resnet50 resnet152; do
    for dataset in sun397 stanford-cars resisc45 eurosat svhn gtsrb mnist dtd oxford_flowers102 pcam fer2013 oxford-iiit-pet stl10 cifar100 cifar10 food101 fashion_mnist emnist_letters kmnist rendered-sst2; do
        finetune
    done
done
