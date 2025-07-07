#!/bin/bash

SCRIPT_DIR=$(dirname $(realpath $0))

source $SCRIPT_DIR/evaluate.sh

MODELS=(
  "google/gemma-2-2b"
  "MergeBench/gemma-2-2b_instruction"
  "MergeBench/gemma-2-2b_math"
  "MergeBench/gemma-2-2b_coding"
  "MergeBench/gemma-2-2b_multilingual"
  "MergeBench/gemma-2-2b_safety"
)

evaluate_all_models
