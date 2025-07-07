#!/bin/bash

SCRIPT_DIR=$(dirname $(realpath $0))

source $SCRIPT_DIR/evaluate.sh

MODELS=(
  "meta-llama/Llama-3.2-3B"
  "MergeBench/Llama-3.2-3B_instruction"
  "MergeBench/Llama-3.2-3B_math"
  "MergeBench/Llama-3.2-3B_coding"
  "MergeBench/Llama-3.2-3B_multilingual"
  "MergeBench/Llama-3.2-3B_safety"
)

evaluate_all_models
