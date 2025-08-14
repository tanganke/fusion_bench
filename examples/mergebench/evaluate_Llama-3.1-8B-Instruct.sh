#!/bin/bash

SCRIPT_DIR=$(dirname $(realpath $0))

source $SCRIPT_DIR/evaluate.sh

MODELS=(
  "meta-llama/Llama-3.1-8B-Instruct"
  "MergeBench/Llama-3.1-8B-Instruct_instruction"
  "MergeBench/Llama-3.1-8B-Instruct_math"
  "MergeBench/Llama-3.1-8B-Instruct_coding"
  "MergeBench/Llama-3.1-8B-Instruct_multilingual"
  "MergeBench/Llama-3.1-8B-Instruct_safety"
)

LM_EVAL_ARGS="--apply_chat_template"

evaluate_all_models
