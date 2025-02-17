#!/bin/bash

# 检查当前仓库是否设置了 http.proxy（仅 global 配置）
current_proxy=$(git config --global --get http.proxy)

if [ -n "$current_proxy" ]; then
    # 如果已设置全局代理，则关闭全局代理
    echo "检测到全局代理已开启，现在关闭全局代理..."
    git config --global --unset http.proxy
    git config --global --unset https.proxy
    echo "全局代理已关闭。"
else
    # 如果未设置全局代理，则开启全局代理
    echo "全局代理未开启，现在开启全局代理..."
    git config --global http.proxy http://127.0.0.1:7890
    git config --global https.proxy http://127.0.0.1:7890
    echo "全局代理已开启。"
fi
