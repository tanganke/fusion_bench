#######################llama3.2 3B task5#############################
CUDA_VISIBLE_DEVICES=1 fusion_bench method=s2_moe/s2_moe_llama_upscaling method.device=cuda method.gate_k=8 method.k=8 modelpool=llama32_3b/llama-32-3B_TA8 taskpool=LMEvalHarnessTaskPool/lm_eval taskpool.tasks=[MergeBench/instruction_val,MergeBench/coding_val,MergeBench/math_val,MergeBench/multilingual_val,MergeBench/safety_val] report_save_path="outputs/llama32_3b/llama32_3b_TA5.json"


fusion_bench method=s2_moe/s2_moe_llama_upscaling method.device=cpu method.gate_k=8 method.k=8 modelpool=CausalLMPool/mergebench/Llama-3.2-3B-Instruct taskpool=dummy