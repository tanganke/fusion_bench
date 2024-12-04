#!/bin/bash

# 遍历从0到30
for i in $(seq 0 30)
do
  # 设置CUDA_VISIBLE_DEVICES环境变量
  CUDA_VISIBLE_DEVICES=0 fusion_bench \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8.yaml \
    method=gossip/clip_ties_merging \
    method.accuracy_test_interval=0 \
    method.ties_merging_steps=$i \
    taskpool=clip-vit-classification_TA8.yaml
done