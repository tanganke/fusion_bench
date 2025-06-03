#######################llama3.2 3B task5#############################
fusion_bench \
    method=s2_moe/s2_moe_llama_upscaling \
    method.device=cpu \
    modelpool=CausalLMPool/mergebench/Llama-3.2-3B-Instruct \
    taskpool=LMEvalHarnessTaskPool/lm_eval \
    taskpool.tasks="[truthfulqa]"

#dummy
HYDRA_FULL_ERROR=1 fusion_bench \
    method=s2_moe/s2_moe_llama_upscaling \
    method.device=cpu \
    modelpool=CausalLMPool/mergebench/Llama-3.2-3B-Instruct \
    taskpool=dummy

#eval 
lm_eval --model_args pretrained="save_models/llama3.2-3B_S2MoE",dtype='bfloat16',parallelize=True --tasks gsm8k_cot --batch_size 8