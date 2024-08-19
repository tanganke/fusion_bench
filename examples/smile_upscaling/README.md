# SMILE Upscaling Examples

This directory contains pre-run experimental results related to the SMILE (Sparse MIxture of Low-rank Experts) upscaling method.

## Contents

1. CLIP-ViT Models
   - [`clip-vit-base-patch32.ipynb`](./clip-vit-base-patch32.ipynb): Experiments with CLIP-ViT-B/32 on eight tasks
   - [`clip-vit-large-patch14.ipynb`](./clip-vit-large-patch14.ipynb): Experiments with CLIP-ViT-L/14 on eight tasks
   - [`clip-vit-base-patch32-ablations-topk.ipynb`](./clip-vit-base-patch32-ablations-topk.ipynb): Ablation studies on number of experts per token (Top-K)

2. Flan-T5 Models
   - [`flan-t5-base.ipynb`](./flan-t5-base.ipynb): Hyperparameter search for full fine-tuned Flan-T5 models
   - [`flan-t5-base-lora16.ipynb`](./flan-t5-base-lora16.ipynb): Hyperparameter search for LoRA fine-tuned Flan-T5 models

3. Mistral-7B Models
   - [`mistral_gsm8k.ipynb`](./mistral_gsm8k.ipynb): Evaluation of upscaled Mistral-7B models on the GSM8K task

4. Projection Merge Experiments (Motivation Experiments)
   - [`clip-vit-base-patch32_single-task_projection-merging.ipynb`](./clip-vit-base-patch32_single-task_projection-merging.ipynb): Experiments with projection merging on CLIP-ViT-B/32 models
