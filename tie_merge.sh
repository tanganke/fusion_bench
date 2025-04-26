#!/bin/bash

# 遍历从0到30
for i in $(seq 15 20)
do
  # 设置CUDA_VISIBLE_DEVICES环境变量
  CUDA_VISIBLE_DEVICES=0 fusion_bench \
    modelpool=clip-vit-base-patch32_generalization_exp1 \
    method=gossip/clip_ties_merging \
    method.accuracy_test_interval=0 \
    method.gossip_max_steps=20 \
    method.scaling_factor=0.343 \
    method.ties_merging_steps=$i \
    taskpool=clip-vit-classification_TA8.yaml
done