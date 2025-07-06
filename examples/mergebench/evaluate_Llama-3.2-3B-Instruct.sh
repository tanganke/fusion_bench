#!/bin/bash

SCRIPT_DIR=$(dirname $(realpath $0))

source $SCRIPT_DIR/evaluate.sh

MODELS=(
  "MergeBench/Llama-3.2-3B-Instruct_instruction"
  "MergeBench/Llama-3.2-3B-Instruct_math"
  "MergeBench/Llama-3.2-3B-Instruct_coding"
  "MergeBench/Llama-3.2-3B-Instruct_multilingual"
  "MergeBench/Llama-3.2-3B-Instruct_safety"
)

evaluate_all_models
