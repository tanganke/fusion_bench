function run_eight_tasks() {
    fusion_bench \
        method=s2_moe/s2_moe_upscaling \
        method.device=cuda \
        method.gate_k=$gate_k method.k=$k \
        modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8 \
        taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8 \
        report_save_path="outputs/ViT-B-32/eight_tasks/gate_k\=${gate_k}_k\=${k}.json"
}

gate_k=16 k=32 run_eight_tasks


#######################llama3.2 3B task5#############################
CUDA_VISIBLE_DEVICES=1 fusion_bench method=s2_moe/s2_moe_llama_upscaling method.device=cuda method.gate_k=8 method.k=8 modelpool=llama32_3b/llama-32-3B_TA8 taskpool=LMEvalHarnessTaskPool/lm_eval taskpool.tasks=[MergeBench/instruction_val,MergeBench/coding_val,MergeBench/math_val,MergeBench/multilingual_val,MergeBench/safety_val] report_save_path="outputs/llama32_3b/llama32_3b_TA5.json"