# SWUDI / ASWUDI

[![arXiv](https://img.shields.io/badge/arXiv-2606.07289-b31b1b.svg)](https://arxiv.org/abs/2606.07289)

**SWUDI** and its adaptive variant **ASWUDI** (called SWUDI-A in the paper) are data-free model merging methods from *Closed-Form Spectral Regularization for Multi-Task Model Merging*.
They start from the per-layer interference proxy of [WUDI-Merging](wudi_merging.md) and observe that its iterative solver works because early stopping acts as a spectral regularizer of an ill-posed normal equation; running the optimizer to convergence is actually worse.
SWUDI applies this regularization explicitly and in closed form: one symmetric eigendecomposition per linear layer, without optimizer state or data.

## Merging rule

For a linear layer with task vectors $\tau_i = W_i - W_0 \in \mathbb{R}^{d_o \times d_i}$, the WUDI loss

$$\mathcal{L}(\tau) = \sum_{i=1}^N \frac{1}{\|\tau_i\|_F^2} \left\|(\tau - \tau_i)\,\tau_i^\top\right\|_F^2$$

is a quadratic whose stationary points satisfy the normal equation $\tau C = D$, with

$$C = \sum_{i=1}^N \frac{\tau_i^\top \tau_i}{\|\tau_i\|_F^2}, \qquad D = \sum_{i=1}^N \frac{\tau_i\, \tau_i^\top \tau_i}{\|\tau_i\|_F^2}.$$

With the eigendecomposition $C = Q \operatorname{diag}(\lambda_1 \ge \dots \ge \lambda_{d_i})\, Q^\top$, both methods compute

$$\tau_m = \tau_\text{init} + \left(D - \tau_\text{init}\, C\right) Q \operatorname{diag}\!\left(\frac{h_k}{\lambda_k}\right) Q^\top, \qquad \tau_\text{init} = \sum_{i=1}^N \tau_i,$$

and differ only in the spectral filter $h_k \in [0, 1]$:

| Method | Spectral filter $h_k$ | Hyperparameters |
| ------ | --------------------- | --------------- |
| SWUDI  | $\mathbf{1}[k \le \lceil r\, d_i \rceil] \cdot \left(1 - e^{-t \lambda_k}\right)$ | `exp_time` $t$, `rank_ratio` $r$ |
| ASWUDI | $\mathbf{1}[k \le K]$, with $K = \left\lceil \left(\sum_k \sqrt{\lambda_k}\right)^2 / \sum_k \lambda_k \right\rceil$ for each layer | none |

The soft factor $1 - e^{-t \lambda_k}$ is exactly the gradient flow of the WUDI loss stopped at time $t$, started from $\tau_\text{init}$.
The hard mask removes the weakly supported tail directions, whose inverse gain $1/\lambda_k$ amplifies the noise of the data-free proxy.
ASWUDI drops the soft factor and chooses the rank of each layer from its own spectrum, as the participation ratio of the singular values $\sqrt{\lambda_k}$ of the stacked normalized task vectors.

Parameters that are not 2-D (biases, LayerNorm, the patch-embedding convolution, ...) are merged by averaging their task vectors.
The merged model is $\theta_0 + s\, \tau_m$, where $s$ is `scaling_factor`.

## Configuration

Both methods are implemented by [`SWUDIMerging`][fusion_bench.method.SWUDIMerging]; ASWUDI corresponds to `exp_time: null` and `rank_ratio: null`.

=== "SWUDI"

    ```yaml title="config/method/swudi/swudi.yaml"
    --8<-- "config/method/swudi/swudi.yaml"
    ```

=== "ASWUDI"

    ```yaml title="config/method/swudi/aswudi.yaml"
    --8<-- "config/method/swudi/aswudi.yaml"
    ```

| Key | Type | Description |
| --- | ---- | ----------- |
| `exp_time` | float or null | Time $t$ of the soft filter $1 - e^{-t\lambda_k}$. `null` disables the soft filter. |
| `rank_ratio` | float in (0, 1] or null | Fraction $r$ of eigendirections kept in every linear layer. `null` selects the rank of each layer adaptively (ASWUDI). |
| `scaling_factor` | float | Multiplier $s$ of the merged task vector. |
| `exclude_keys` | list or null | 2-D parameters that are averaged instead of solved in closed form. |

The defaults (`exp_time=1300`, `rank_ratio=0.65`, `scaling_factor=1.0`) are the values used for CLIP-ViT-B/32 and ViT-L/14 in the paper; CLIP-ViT-B/16 uses `exp_time=1800`.

## Examples

### CLI Usage

Merge the eight CLIP-ViT-B/32 models of the TA8 benchmark with SWUDI and with ASWUDI:

```bash
fusion_bench \
    method=swudi/swudi \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8

fusion_bench \
    method=swudi/aswudi \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8
```

Other backbones and the 20-task benchmark:

```bash
# CLIP-ViT-B/16
fusion_bench \
    method=swudi/swudi method.exp_time=1800 \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch16_TA8 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8_B16

# CLIP-ViT-L/14
fusion_bench \
    method=swudi/swudi \
    modelpool=CLIPVisionModelPool/clip-vit-large-patch14_TA8 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8_L14

# CLIP-ViT-B/32, TALL20
fusion_bench \
    method=swudi/aswudi \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TALL20 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL20
```

### API Usage

```python
from fusion_bench.method import SWUDIMerging, swudi_merging

# SWUDI and ASWUDI as algorithms on a model pool
merged_model = SWUDIMerging(exp_time=1300, rank_ratio=0.65).run(modelpool)
merged_model = SWUDIMerging(exp_time=None, rank_ratio=None).run(modelpool)

# Directly on a list of task vectors (state dicts of finetuned - pretrained)
merged_task_vector = swudi_merging(task_vectors, exp_time=None, rank_ratio=None)
```

## Results

Average accuracy (%) reported in the paper, with `scaling_factor=1.0`:

| Method | TA8 ViT-B/32 | TA8 ViT-B/16 | TA8 ViT-L/14 | TALL20 ViT-B/32 |
| ------ | ------------ | ------------ | ------------ | --------------- |
| WUDI-Merging | 84.63 | 89.17 | 92.16 | 61.06 |
| SWUDI  | 85.55 | 89.57 | 92.51 | 75.60 |
| ASWUDI | 85.53 | 89.49 | 92.52 | 75.62 |

Accuracy (%) of this implementation on CLIP-ViT-B/32 TA8, obtained with the default configurations and the CLI commands above:

| Method | SUN397 | Cars | RESISC45 | EuroSAT | SVHN | GTSRB | MNIST | DTD | Average |
| ------ | ------ | ---- | -------- | ------- | ---- | ----- | ----- | --- | ------- |
| SWUDI  | 70.06 | 72.96 | 87.29 | 95.85 | 94.22 | 95.00 | 99.29 | 69.68 | 85.54 |
| ASWUDI | 69.98 | 72.86 | 87.25 | 95.74 | 94.36 | 95.00 | 99.30 | 69.84 | 85.54 |

Merging the 73 two-dimensional weight matrices of CLIP-ViT-B/32 takes about 2 seconds on a single NVIDIA H20 GPU, excluding model loading.

## Implementation Details

- [fusion_bench.method.SWUDIMerging][]
- [fusion_bench.method.swudi_merging][]

## Reference

```bibtex
@article{wei2026swudi,
  title   = {Closed-Form Spectral Regularization for Multi-Task Model Merging},
  author  = {Wei, Yongxian and Cheng, Runxi and Zhang, Xingxuan and Shen, Li and
             Yuan, Chun and Cui, Peng and Tao, Dacheng},
  journal = {arXiv preprint arXiv:2606.07289},
  year    = {2026},
}
```
