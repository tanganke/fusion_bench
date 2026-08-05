#! /usr/bin/env bash
# This script is used to generate the model configuration file for ViT models.
# for example, you can run the following command to generate the model configuration file for ViT-B/32:

# clip-vit-base-patch32_sun397.yaml:
# sun397:
#   _target_: transformers.CLIPVisionModel.from_pretrained
#   path: tanganke/clip-vit-base-patch32_sun397

for model in clip-vit-base-patch32 clip-vit-base-patch16 clip-vit-large-patch14; do
    # generate pretrained model config
    file="${model}.yaml"
    echo "_pretrained_:" >${file}
    echo "  _target_: transformers.CLIPVisionModel.from_pretrained" >>${file}
    echo "  pretrained_model_name_or_path: openai/${model}" >>${file}

    for task in sun397 stanford-cars resisc45 eurosat svhn gtsrb mnist dtd; do
        file="${model}_${task}.yaml"
        echo "${task}:" >${file}
        echo "  _target_: transformers.CLIPVisionModel.from_pretrained" >>${file}
        echo "  pretrained_model_name_or_path: tanganke/${model}_${task}" >>${file}
    done
done
