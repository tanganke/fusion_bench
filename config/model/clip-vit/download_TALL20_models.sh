#! /bin/bash
for MODEL in clip-vit-base-patch32 clip-vit-base-patch16 clip-vit-large-patch14; do
    for TASK in sun397 stanford-cars resisc45 eurosat svhn gtsrb mnist dtd oxford_flowers102 pcam fer2013 oxford-iiit-pet stl10 cifar100 cifar10 food101 fashion_mnist emnist_letters kmnist rendered-sst2; do
        huggingface-cli download --local-dir tanganke/${MODEL}_${TASK} tanganke/${MODEL}_${TASK}
    done
done
