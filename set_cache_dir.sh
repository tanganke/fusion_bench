#!/bin/bash

SCRIPT_DIR=$(dirname $(realpath $0))

export HF_HOME=$SCRIPT_DIR/.cache/huggingface
echo "HF_HOME set to $HF_HOME"
