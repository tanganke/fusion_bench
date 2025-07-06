#! /bin/bash

function lm_eval_evaluate_task() {
  # if output file exists, skip
  if [ -f $OUTPUT_DIR/$MODEL/$TASK ]; then
    echo "Skipping $MODEL on $TASK because output file exists"
  else
    echo "Evaluating $MODEL on $TASK"
    lm_eval \
      --model_args pretrained="$MODEL",dtype='bfloat16',parallelize=True \
      --apply_chat_template \
      --tasks $TASK \
      --batch_size $BATCH_SIZE \
      --output_path $OUTPUT_DIR/$MODEL/$TASK
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
    for TASK in mbpp; do
      lm_eval_evaluate_task
    done

    # safety
    for TASK in truthfulqa; do
      lm_eval_evaluate_task
    done
  done
}

OUTPUT_DIR="results/"
BATCH_SIZE=8

if [ ! -d $OUTPUT_DIR ]; then
  mkdir -p $OUTPUT_DIR
fi
