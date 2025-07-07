#! /bin/bash

LM_EVAL_ARGS=""

function run_command() {
  echo "Running $@"
  $@
}

function lm_eval_evaluate_task() {
  # if output file exists, skip
  MODEL_PATH=$(echo $MODEL | sed 's/\//__/g')
  if [ -d $OUTPUT_DIR/$TASK/$MODEL_PATH ]; then
    echo "Skipping $MODEL on $TASK because output file exists"
  else
    echo "Evaluating $MODEL on $TASK"
    run_command lm_eval \
      --model_args pretrained="$MODEL",dtype='bfloat16',parallelize=True \
      $LM_EVAL_ARGS \
      --tasks $TASK \
      --batch_size $BATCH_SIZE \
      --confirm_run_unsafe_code \
      --output_path $OUTPUT_DIR/$TASK
  fi
}

function evaluate_all_models() {
  for MODEL in ${MODELS[@]}; do
    # math
    for TASK in gsm8k_cot; do
      lm_eval_evaluate_task
    done

    # multilingual
    for TASK in m_mmlu_fr arc_fr hellaswag_fr m_mmlu_es arc_es hellaswag_es m_mmlu_de arc_de hellaswag_de m_mmlu_ru arc_ru hellaswag_ru; do
      lm_eval_evaluate_task
    done

    # instruction following
    for TASK in ifeval; do
      lm_eval_evaluate_task
    done

    # coding
    for TASK in humaneval_plus mbpp_plus; do
      lm_eval_evaluate_task
    done

    # safety
    for TASK in truthfulqa toxigen winogender; do
      lm_eval_evaluate_task
    done
  done
}

OUTPUT_DIR="results/"
BATCH_SIZE=8

if [ ! -d $OUTPUT_DIR ]; then
  mkdir -p $OUTPUT_DIR
fi
