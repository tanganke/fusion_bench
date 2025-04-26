# Layer-wise gossip
fusion_bench \
    method=gossip/layer_wise_flan_t5 \
    method.lr=1e-3 \
    modelpool=Seq2SeqLMPool/flan-t5-base_glue_lora16_tta \
    taskpool=taskpool=flan-t5_glue_text_generation
