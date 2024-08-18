# hyperparameter search for full fine-tuned flan-t5-base
for gate_k in 4 8 16 32; do
    for k in 16 32 64 128; do
        fusion_bench \
            method=singular_moe_upscaling \
                method.device=cpu \
                method.gate_k=$gate_k method.k=$k \
            modelpool=flan-t5-base_glue \
            taskpool=flan-t5_glue_text_generation \
            save_report="outputs/flan-t5-base/glue_text_generation/gate_k\=${gate_k}_k\=${k}.json"
    done
done

# hyperparameter search for lora fine-tuned flan-t5-base
for gate_k in 2 4 8; do
    for k in 4 8 16; do
        fusion_bench \
            method=singular_moe_upscaling \
                method.device=cuda \
                method.gate_k=$gate_k method.k=$k \
            modelpool=flan-t5-base_glue_lora16 \
            taskpool=flan-t5_glue_text_generation \
            save_report="outputs/flan-t5-base_lora16/glue_text_generation/gate_k\=${gate_k}_k\=${k}.json"
    done
done


