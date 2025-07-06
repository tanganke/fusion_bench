#!/bin/bash

SCRIPT_DIR=$(dirname $(realpath $0))

if [ -d /mnt/huggingface_cache ]; then
    export HF_HOME=/mnt/huggingface_cache/
else
    export HF_HOME=$SCRIPT_DIR/.cache/huggingface
fi

echo "HF_HOME set to $HF_HOME"

# if `HF_HOME` does not exist, create it
if [ ! -d $HF_HOME ]; then
    mkdir -p $HF_HOME
fi
