#! /usr/bin/env bash
# This script is used to generate the model configuration file for Flan-T5 models.
# for example, you can run the following command to generate the model configuration file for Flan-T5-base:
#
# flan-t5-base.yaml:
# _pretrained_:
#   _target_: transformers.AutoModelForSeq2SeqLM.from_pretrained
#   pretrained_model_name_or_path: google/flan-t5-base
# glue-cola:
#  _target_: transformers.AutoModelForSeq2SeqLM.from_pretrained
#  pretrained_model_name_or_path: tanganke/flan-t5-base_glue-cola
for model in flan-t5-base; do
    # generate pretrained model config
    file="${model}.yaml"
    echo "_pretrained_:" >${file}
    echo "  _target_: transformers.AutoModelForSeq2SeqLM.from_pretrained" >>${file}
    echo "  pretrained_model_name_or_path: google/${model}" >>${file}

    for task in glue-cola glue-sst2 glue-mrpc glue-qqp glue-stsb glue-mnli glue-qnli glue-rte; do
        file="${model}_${task}.yaml"
        echo "${task}:" >${file}
        echo "  _target_: transformers.AutoModelForSeq2SeqLM.from_pretrained" >>${file}
        echo "  pretrained_model_name_or_path: tanganke/${model}_${task}" >>${file}
    done
done

for model in flan-t5-base flan-t5-large; do
    for task in glue-cola glue-sst2 glue-mrpc glue-qqp glue-stsb glue-mnli glue-qnli glue-rte; do
        file="${model}_${task}_lora-16.yaml"
        if [ -f ${file} ]; then
            rm ${file}
        fi
        echo "${task}:" >${file}
        echo "  _target_: fusion_bench.modelpool.seq2seq_lm.modelpool.load_lora_model" >>${file}
        echo "  base_model_path: google/${model}" >>${file}
        echo "  peft_model_path: tanganke/${model}_${task}_lora-16" >>${file}
    done
done
