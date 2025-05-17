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
