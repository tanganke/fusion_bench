# MagMax

[MagMax](https://arxiv.org/abs/2407.06322) (Marczak et al., ECCV 2024) is a
training-free model merging method that selects, **element-wise**, the task
vector with the **largest absolute magnitude** across tasks.

The intuition is that each fine-tuned model encodes its task knowledge in a
sparse set of large-magnitude parameter updates. For a given parameter
position, the task whose update has the highest magnitude is the one that
"cares most" about that parameter, and its value is the one most likely to
contain task-relevant signal. Summing or averaging would dilute that signal;
MagMax keeps it intact.

## Merging rule

Given a pretrained model $\theta_0$ and $T$ fine-tuned models
$\{\theta_t\}_{t=1}^T$, compute the per-task task vectors:

$$\tau_t = \theta_t - \theta_0$$

For every parameter index $j$ select the task with the maximum absolute
value at that position:

$$t^\star(j) = \arg\max_{t \in \{1,\dots,T\}} \lvert \tau_t[j] \rvert,
\qquad
\tau_\text{max}[j] = \tau_{t^\star(j)}[j]$$

The merged model is then

$$\theta_\text{merged} = \theta_0 + \alpha \cdot \tau_\text{max}$$

where $\alpha$ (`scaling_factor` in the config) scales the merged task
vector. The official 8-dataset reproduction script uses $\alpha = 0.5$,
which we adopt as the default.

!!! note "Implementation details"
    - **Tie-breaking** uses `>=` (later task vectors win on ties), matching
      the reference implementation.
    - Integer / boolean buffer keys (e.g. `position_ids`) are skipped and
      copied through from the pretrained state — the merge only touches
      floating-point parameters.

## Configuration

```yaml title="config/method/magmax/magmax.yaml"
--8<-- "config/method/magmax/magmax.yaml"
```

| Key | Type | Description |
|---|---|---|
| `scaling_factor` | float | $\alpha$. Scales the merged max-magnitude task vector before adding it back to $\theta_0$. |
| `inplace` | bool | If `true`, the loaded pretrained model is mutated in place; otherwise a copy is returned. |

## Examples

### CLI

```bash
fusion_bench \
    method=magmax/magmax \
    method.scaling_factor=0.5 \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8
```

### Reproduction scripts

Bundled scripts live in [`examples/magmax/`](https://github.com/tanganke/fusion_bench/tree/main/examples/magmax). The paper distinguishes between two protocols (see the example README for details):

- **Independent FT** (default for these CLI scripts) — runs MagMax on the publicly hosted per-task CLIP checkpoints. Faster, no training required, but a few points below the paper because the checkpoints differ.
    - `clip_vit_base_patch32_TA8.sh`
    - `clip_vit_base_patch16_TA8.sh`
    - `clip_vit_base_patch32_TALL14.sh`
    - `sweep_scaling_factor.sh` — $\alpha$ sweep
- **Sequential FT** (paper-faithful, headline setting): `sequential_ft_and_merge.py` performs sequential fine-tuning of CLIP on the 8 tasks in the paper's order, saves each task's snapshot, then merges with MagMax and evaluates.

### Programmatic use

```python
from fusion_bench.method import MagMaxAlgorithm, magmax_merge

# As an Algorithm (with a BaseModelPool):
algorithm = MagMaxAlgorithm(scaling_factor=0.5)
merged = algorithm.run(modelpool)

# As a one-shot function on bare nn.Modules:
merged = magmax_merge(pretrained_model, [m1, m2, m3], scaling_factor=0.5)
```

## Reproduction results

Numbers obtained on CLIP-ViT-B/32 (TA8: SUN397, Stanford-Cars, RESISC45, EuroSAT, SVHN, GTSRB, MNIST, DTD).

### Independent fine-tuning protocol (`ind-ft`)

Uses the publicly hosted per-task `tanganke/clip-vit-base-patch32_*` checkpoints (each fine-tuned from `_pretrained_` in isolation). $\alpha$ sweep:

| $\alpha$ | 0.3 | 0.4 | 0.5 | **0.6** | 0.7 | 0.8 | 1.0 |
|---|---|---|---|---|---|---|---|
| Avg Acc (%) | 66.49 | 69.23 | 70.70 | **71.10** | 70.31 | 68.80 | 63.59 |

For reference, on the same checkpoints: Task Arithmetic ($\alpha=0.3$) → 77.14 % and TIES Merging ($\alpha=0.3$) → 77.60 % on CLIP-ViT-B/16 — i.e. all three task-vector methods sit within ~2 points of one another. The MagMax paper's headline numbers (~84 % on ViT-B/16) come from the `seq-ft` protocol below, not from `ind-ft` checkpoints.

### Sequential fine-tuning protocol (`seq-ft`, paper-faithful)

Trained end-to-end with `examples/magmax/sequential_ft_and_merge.py` on CLIP-ViT-B/32, 2500 steps per task in the paper's task order, then merged with `scaling_factor = 0.5`:

| Stanford-Cars | MNIST | EuroSAT | SVHN | RESISC45 | SUN397 | DTD | GTSRB | **Average** |
|---|---|---|---|---|---|---|---|---|
| 71.74 | 99.09 | 96.67 | 94.39 | 85.59 | 68.80 | 58.99 | 69.21 | **80.56** |

`seq-ft` outperforms `ind-ft` by ~10 percentage points on B/32, mirroring the paper's central observation that MagMax recovers near-multi-task performance from sequentially fine-tuned snapshots. The paper reports ~84 % on CLIP-ViT-B/16 in the same setting; our B/32 result sits in the expected B/32-to-B/16 scaling band.

## Reference

```bibtex
@inproceedings{marczak2024magmax,
  title     = {{MagMax}: Leveraging Model Merging for Seamless Continual Learning},
  author    = {Marczak, Daniel and Twardowski, Bart{\l}omiej and
               Trzci{\'n}ski, Tomasz and Cygert, Sebastian},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2024},
}
```
