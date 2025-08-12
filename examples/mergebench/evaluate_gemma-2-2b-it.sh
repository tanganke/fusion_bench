#!/bin/bash

SCRIPT_DIR=$(dirname $(realpath $0))

source $SCRIPT_DIR/evaluate.sh

MODELS=(
  "google/gemma-2-2b-it"
  "MergeBench/gemma-2-2b-it_instruction"
  "MergeBench/gemma-2-2b-it_math"
  "MergeBench/gemma-2-2b-it_coding"
  "MergeBench/gemma-2-2b-it_multilingual"
  "MergeBench/gemma-2-2b-it_safety"
)

LM_EVAL_ARGS="--apply_chat_template"

evaluate_all_models
