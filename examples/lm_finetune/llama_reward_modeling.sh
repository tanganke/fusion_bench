fusion_bench --config-name llama_full_finetune \
  fabric.loggers.name=llama_full_bradly_terry_rm \
  method=lm_finetune/bradly_terry_rm \
  method.dataloader_kwargs.batch_size=8 \
  method.optimizer.lr=5e-6 \
  method.optimizer.weight_decay=0.001 \
  method.max_epochs=1 \
  method.checkpoint_save_interval=step \
  method.checkpoint_save_frequency=2000 \
  modelpool=SeqenceClassificationModelPool/llama_preference700k
