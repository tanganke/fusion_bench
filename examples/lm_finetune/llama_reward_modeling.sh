fusion_bench --config-name llama_full_finetune \
  fabric.loggers.name=llama_full_bradley_terry_rm \
  method=lm_finetune/bradley_terry_rm \
  method.dataloader_kwargs.batch_size=8 \
  method.accumulate_grad_batches=16 \
  method.lr_scheduler.min_lr=1e-7 \
  method.lr_scheduler.max_lr=5e-6 \
  method.lr_scheduler.warmup_steps=100 \
  method.optimizer.lr=0 \
  method.optimizer.weight_decay=0.001 \
  method.gradient_clip_val=1 \
  method.max_epochs=2 \
  method.checkpoint_save_interval=epoch \
  method.checkpoint_save_frequency=1 \
  modelpool=SequenceClassificationModelPool/llama_preference700k
