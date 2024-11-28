# finetune the full model on alpaca data
fusion_bench --config-name llama_full_finetune \
  fabric.loggers.name=llama_full_finetune \
  method=lm_finetune/fullfinetune_sft \
  method.dataloader_kwargs.batch_size=8 \
  modelpool=CausalLMPool/llama_alpaca_cleaned
