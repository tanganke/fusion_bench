#! /usr/bin/env bash
# if CUDA_VISIBLE_DEVICES is set, get the number of GPUs from it
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    NUM_GPUS=$(nvidia-smi -L | wc -l)
else
    NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')
fi
echo "Using $NUM_GPUS GPUs"

function finetune() {
    local output_dir="outputs/${model}/${dataset}"
    local version="batch_size=${batch_size},lr=${lr}"
    if [ -d "$output_dir/$version" ]; then
        echo "$output_dir/$version exists, skip finetuning."
    else
        # the per-GPU batch size is global batch size divided by number of GPUs
        rich-run fusion_bench --config-name model_fusion \
            path.log_dir=\"${output_dir}/$version\" seed=0 \
            method=classification/image_classification_finetune \
                method.max_epochs=-1 \
                method.max_steps=4000 \
                method.save_top_k=-1 \
                method.save_interval=1000 \
                method.save_on_train_epoch_end=false \
                method.optimizer.lr=${lr} \
                method.lr_scheduler=null \
                method.dataloader_kwargs.batch_size=$((${batch_size} / ${NUM_GPUS})) \
            modelpool=ConvNextForImageClassification/${model} \
            modelpool.models._pretrained_.dataset_name=${dataset} \
            +dataset/image_classification/train@modelpool.train_datasets=${dataset} \
            +dataset/image_classification/test@modelpool.val_datasets=${dataset}
    fi
}

for model in convnext-base-224; do
    for dataset in sun397 stanford-cars resisc45 eurosat svhn gtsrb mnist dtd oxford_flowers102 pcam fer2013 oxford-iiit-pet stl10 cifar100 cifar10 food101 fashion_mnist emnist_letters kmnist rendered-sst2; do
        # global batch size
        for batch_size in 64 128 256; do
            for lr in 0.001 0.005 0.01; do
                finetune
            done
        done
    done
done
