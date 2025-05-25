#######################llama3.2 3B task5#############################
fusion_bench \
    method=s2_moe/s2_moe_llama_upscaling \
    method.device=cpu method.gate_k=8 method.k=8 \
    modelpool=CausalLMPool/mergebench/Llama-3.2-3B-Instruct \
    taskpool=LMEvalHarnessTaskPool/lm_eval \
    taskpool.tasks="[truthfulqa]"
