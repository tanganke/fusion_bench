# MagMax

Reproduction scripts for **MagMax: Leveraging Model Merging for Seamless
Continual Learning** (Marczak et al., ECCV 2024,
[arXiv:2407.06322](https://arxiv.org/abs/2407.06322)) on FusionBench.

The merge rule is element-wise selection of the task vector with the
largest absolute magnitude across tasks:

```
tau_t        = theta_t - theta_0
tau_max[j]   = tau_{argmax_t |tau_t[j]|}[j]
theta_merged = theta_0 + alpha * tau_max
```

## Two fine-tuning protocols

The MagMax paper evaluates the merge rule under two distinct
fine-tuning protocols, and the paper's headline numbers come from the
*sequential* one:

- **Independent FT (`ind-ft`)** — each task is fine-tuned from the
  pretrained model in isolation. This is what the original Task
  Arithmetic / TIES papers use, and what the publicly hosted CLIP
  checkpoints from `tanganke/...` provide. The shell scripts below run
  MagMax on these ready-made checkpoints.
- **Sequential FT (`seq-ft`)** — each task is fine-tuned starting from
  the *previous* task's weights. The paper's "seamless continual
  learning" claim is about this setting, and the official
  `merge_8datasets.py` is wired for it. We reproduce this end-to-end
  with `sequential_ft_and_merge.py` below.

## Files

| File | Setting |
|---|---|
| `clip_vit_base_patch32_TA8.sh` | `ind-ft`, CLIP-ViT-B/32, TA8 |
| `clip_vit_base_patch16_TA8.sh` | `ind-ft`, CLIP-ViT-B/16, TA8 |
| `clip_vit_base_patch32_TALL14.sh` | `ind-ft`, CLIP-ViT-B/32, TALL14 |
| `sweep_scaling_factor.sh` | `ind-ft` $\alpha$ sweep on TA8 |
| `sequential_ft_and_merge.py` | `seq-ft` — trains the 8 snapshots itself, then merges and evaluates |

## Independent FT (quick start)

```bash
bash examples/magmax/clip_vit_base_patch32_TA8.sh
```

Override the scaling factor from the command line:

```bash
fusion_bench \
    method=magmax/magmax \
    method.scaling_factor=0.5 \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8
```

## Sequential FT (paper-faithful reproduction)

`sequential_ft_and_merge.py` mirrors the official
`tmp/magmax/finetune_8datasets.py + merge_8datasets.py` pair end-to-end.

Task order (paper): `Cars -> MNIST -> EuroSAT -> SVHN -> RESISC45 -> SUN397 -> DTD -> GTSRB`.

```bash
# ViT-B/32, ~50 min on one H100
python examples/magmax/sequential_ft_and_merge.py \
    --base-model openai/clip-vit-base-patch32 \
    --output-dir outputs/magmax/seq_ft_b32 \
    --num-steps-per-task 2500 \
    --scaling-factor 0.5

# ViT-B/16 — closer to the paper's setting, ~2-3h
python examples/magmax/sequential_ft_and_merge.py \
    --base-model openai/clip-vit-base-patch16 \
    --output-dir outputs/magmax/seq_ft_b16 \
    --num-steps-per-task 2500 \
    --scaling-factor 0.5
```

Per-task snapshots are written to
`<output-dir>/{idx:02d}_{task_name}/` in HuggingFace format; the merged
model lands under `merged_alpha<...>/`; the evaluation report is the
JSON next to the merged model.

To re-run only the merge step on existing snapshots:

```bash
python examples/magmax/sequential_ft_and_merge.py \
    --output-dir outputs/magmax/seq_ft_b32 \
    --skip-training \
    --scaling-factor 0.6   # try a different alpha
```

## Notes on closing the gap to the paper

The MagMax paper trains its own CLIP checkpoints (with `open_clip` and a
per-task epoch schedule) and reports numbers against those. Running MagMax
against the public `tanganke/clip-vit-base-patch{16,32}_*` checkpoints
(which is what the `*_TA8.sh` scripts do) lands a few points below the
paper because the underlying checkpoints differ — not because the merge
rule deviates. The `sequential_ft_and_merge.py` path trains the
checkpoints inside this repo, so numbers there are directly comparable
to the paper's seq-ft column once you match the iteration budget.
