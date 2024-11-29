fusion_bench --config-name llama_full_finetune \
  fabric.loggers.name=llama_full_finetune \
  method=lm_finetune/fullfinetune_sft \
  method.dataloader_kwargs.batch_size=3 \
  modelpool=SeqenceClassificationModelPool/llama_preference700k
