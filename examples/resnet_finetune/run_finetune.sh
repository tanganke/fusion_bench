#! /usr/bin/env bash
for model in resnet18 resnet50 resnet152; do
    for dataset in sun397 stanford-cars resisc45 eurosat svhn gtsrb mnist dtd oxford_flowers102 pcam fer2013 oxford-iiit-pet stl10 cifar100 cifar10 food101 fashion_mnist emnist_letters kmnist rendered-sst2; do
        output_dir="outputs/${model}/${dataset}"
        if [ -d "$output_dir" ]; then
            echo "$output_dir exists, skip finetuning."
            continue
        fi
        rich-run fusion_bench --config-name model_fusion \
            path.log_dir=$output_dir \
            method=classification/image_classification_finetune \
            modelpool=ResNetForImageClassfication/transformers/${model}_${dataset}
    done
done
