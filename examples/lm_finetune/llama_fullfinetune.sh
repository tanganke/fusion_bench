# finetune the full model on alpaca data
fusion_bench --config-name llama_full_finetune \
  fabric.loggers.name=llama_full_finetune \
  method=lm_finetune/fullfinetune_sft \
  method.dataloader_kwargs.batch_size=8 \
  modelpool=CausalLMPool/llama_alpaca_cleaned

# full finetune on codealpaca
fusion_bench --config-name llama_full_finetune \
  fabric.loggers.name=llama_full_finetune \
  method=lm_finetune/fullfinetune_sft \
  method.dataloader_kwargs.batch_size=8 \
  modelpool=CausalLMPool/llama_codealpaca

# full finetune on metamathqa
fusion_bench --config-name llama_full_finetune \
  fabric.loggers.name=llama_full_finetune \
  method=lm_finetune/fullfinetune_sft \
  method.dataloader_kwargs.batch_size=4 \
  method.checkpoint_save_interval=step \
  method.checkpoint_save_frequency=2000 \
  method.max_epochs=1 \
  modelpool=CausalLMPool/llama_metamathqa
