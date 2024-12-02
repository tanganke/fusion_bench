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

# full finetune on ultrachat
fusion_bench --config-name llama_full_finetune \
  fabric=llama_peft_fsdp \
  fabric.loggers.name=llama_lora_finetune \
  method=lm_finetune/peftfinetune_sft \
  method.dataloader_kwargs.batch_size=1 \
  method.max_epochs=1 \
  method.gradient_clip_val=1.0 \
  method.accumulate_grad_batches=16 \
  method.checkpoint_save_interval=step \
  method.checkpoint_save_frequency=2000 \
  modelpool=CausalLMPool/llama_ultrachat \
  modelpool.pretrained_model_name_or_path=meta-llama/Meta-Llama-3-8B-Instruct
