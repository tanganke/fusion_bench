#!/bin/bash

# 从 0.1 到 1.0 的步长
for value in $(seq 0.1 0.1 1.0); do
    # 执行命令并将当前的 value 插入命令中
    echo "Running with method.init_values=$value"
    python fusion_bench/scripts/cli.py --config-name=gossip method.gossip_skip_adamerging=true method.init_values=$value
done
