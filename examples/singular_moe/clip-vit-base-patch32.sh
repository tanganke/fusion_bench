# hyperparameter search
for gate_k in 1 2 4 8 16 32; do
    for k in 4 8 16 32 64 128 -1; do
        fusion_bench \
            method=singular_moe_upscaling \
            method.device=cuda \
            method.gate_k=$gate_k method.k=$k \
            modelpool=clip-vit-base-patch32_TA8 \
            taskpool=clip-vit-classification_TA8.local \
            save_report="outputs/ViT-B-32/eight_tasks/gate_k\=${gate_k}_k\=${k}.json"
    done
done
