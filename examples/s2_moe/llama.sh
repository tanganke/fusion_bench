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


#bigcode
accelerate launch  main.py   --model /home/hfh/fusionbench_0517/save_models/llama3.2-3B_S2MoE --modeltype S2MoE   --max_length_generation 512   --precision bf16   --tasks humanevalplus,mbppplus   --temperature 0.2   --n_samples 10   --batch_size 10   --allow_code_execution   --metric_output_path outputs/code_eval.json   --use_auth_token


#safety
CUDA_VISIBLE_DEVICES=0,1,2,3 python evaluation/eval.py generators   --model_name_or_path MergeBench/Llama-3.2-3B-Instruct_safety   --model_input_template_path_or_name llama3   --tasks wildguardtest,harmbench,xstest,do_anything_now   --report_output_path outputs/safety_eval.json   --save_individual_results_path outputs/safety_generation.json   --batch_size 8