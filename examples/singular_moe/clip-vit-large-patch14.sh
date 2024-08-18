# hyperparameter search
for gate_k in 1 2 4 8 16 32 64 128; do
    for k in 4 8 16 32 64 128 -1; do
        fusion_bench \
            method=singular_moe_upscaling \
            method.gate_k=$gate_k method.k=$k \
            modelpool=clip-vit-large-patch14_TA8 \
            taskpool=clip-vit-classification_TA8.local \
            taskpool.clip_model=openai/clip-vit-large-patch14 \
            save_report="outputs/ViT-L-14/eight_tasks/gate_k\=${gate_k}_k\=${k}.json"
    done
done
