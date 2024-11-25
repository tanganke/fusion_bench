# Layer-wise adamerging
fusion_bench \
    method=adamerging/layer_wise_flan_t5 \
    method.optimizer.lr=1e-3 \
    modelpool=Seq2SeqLMPool/flan-t5-base_glue_lora16_tta \
    taskpool=flan-t5_glue_text_generation
