R"""
Closed-Form Spectral Regularization for Multi-Task Model Merging (SWUDI / ASWUDI).

Reference:
    Wei, Y., Cheng, R., Zhang, X., Shen, L., Yuan, C., Cui, P., Tao, D.
    "Closed-Form Spectral Regularization for Multi-Task Model Merging."
    arXiv:2606.07289, 2026.
    https://arxiv.org/abs/2606.07289

For a linear layer with task vectors tau_i = W_i - W_0 of shape (d_out, d_in),
WUDI-Merging (arXiv:2503.08099) minimizes the data-free interference proxy

    L(tau) = sum_i ||(tau - tau_i) tau_i^T||_F^2 / ||tau_i||_F^2

with a few hundred Adam steps. Its stationary points solve the normal equation
tau C = D, where

    C = sum_i tau_i^T tau_i / ||tau_i||_F^2,    D = sum_i tau_i tau_i^T tau_i / ||tau_i||_F^2.

With the eigendecomposition C = Q diag(lambda_1 >= ... >= lambda_{d_in}) Q^T,
SWUDI replaces the iterations by a closed-form spectral filter h_k in [0, 1]:

    tau_m = tau_init + (D - tau_init C) Q diag(h_k / lambda_k) Q^T,    tau_init = sum_i tau_i.

- SWUDI:  h_k = 1[k <= ceil(r * d_in)] * (1 - exp(-t * lambda_k)).
  The soft factor is the gradient-flow solution stopped at time t (the
  early-stopping effect of iterative WUDI), and the hard mask removes the
  weakly supported tail directions that amplify proxy noise.
- ASWUDI (SWUDI-A in the paper): h_k = 1[k <= K], with a per-layer rank
  K = ceil((sum_k sqrt(lambda_k))^2 / sum_k lambda_k), i.e. the participation
  ratio of the singular values of the stacked normalized task vectors. It has
  no hyperparameters.

Parameters that are not 2-D (biases, normalization layers, convolutions, ...)
are merged by averaging their task vectors.
"""

import logging
import math
from typing import Dict, List, Optional, Union

import torch
from torch import Tensor, nn
from tqdm.auto import tqdm

from fusion_bench import BaseAlgorithm, BaseModelPool, auto_register_config
from fusion_bench.mixins import LightningFabricMixin
from fusion_bench.utils.type import StateDictType

log = logging.getLogger(__name__)

# Floor on the eigenvalues of C, keeps h_k / lambda_k finite on rank-deficient layers.
_MIN_EIGENVALUE = 1e-8


def _spectral_merge(
    task_vectors: Tensor,
    exp_time: Optional[float],
    rank_ratio: Optional[float],
) -> Tensor:
    """Closed-form spectral merge of the stacked task vectors ``(N, d_out, d_in)`` of one linear layer."""
    sq_norms = torch.norm(task_vectors.flatten(1), p=2, dim=-1).square()
    if sq_norms.max() < 1e-12:
        return task_vectors.mean(dim=0)
    weights = 1.0 / sq_norms.clamp(min=1e-12)

    gram = torch.einsum("nab,nac->nbc", task_vectors, task_vectors)  # tau_i^T tau_i
    C = torch.einsum("n,nbc->bc", weights, gram)
    D = torch.einsum(
        "n,nac->ac", weights, torch.einsum("nab,nbc->nac", task_vectors, gram)
    )

    eigvals, Q = torch.linalg.eigh(C)  # ascending order
    eigvals = eigvals.clamp(min=_MIN_EIGENVALUE)

    d_in = eigvals.numel()
    if rank_ratio is None:
        sigma = eigvals.flip(0).sqrt()
        rank = math.ceil((sigma.sum().square() / sigma.square().sum()).item())
    else:
        rank = math.ceil(rank_ratio * d_in)
    rank = min(max(rank, 1), d_in)

    h = torch.zeros_like(eigvals)
    if exp_time is None:
        h[-rank:] = 1.0
    else:
        h[-rank:] = 1.0 - torch.exp(-exp_time * eigvals[-rank:])

    tau_init = task_vectors.sum(dim=0)
    # (D - tau_init C) Q diag(h / lambda) Q^T, using C Q = Q diag(lambda)
    return tau_init + ((D @ Q) * (h / eigvals) - (tau_init @ Q) * h) @ Q.mT


@torch.no_grad()
def swudi_merging(
    task_vectors: List[StateDictType],
    exp_time: Optional[float] = 1300.0,
    rank_ratio: Optional[float] = 0.65,
    exclude_keys: Optional[List[str]] = None,
    accelerator: Union[str, torch.device] = "cuda",
) -> StateDictType:
    """
    Merge task vectors with SWUDI, or with ASWUDI when ``exp_time`` and ``rank_ratio`` are both ``None``.

    Args:
        task_vectors: One state dict of ``finetuned - pretrained`` per task.
        exp_time: Time ``t`` of the soft filter ``1 - exp(-t * lambda)``.
            ``None`` disables the soft filter (hard truncation).
        rank_ratio: Keep the top ``ceil(rank_ratio * d_in)`` eigendirections of every linear layer.
            ``None`` selects the rank of each layer from its spectrum (participation ratio).
        exclude_keys: 2-D parameters that are averaged instead of solved in closed form.
        accelerator: Device on which the per-layer eigendecompositions are computed.

    Returns:
        The merged task vector.
    """
    if exp_time is not None and exp_time < 0:
        raise ValueError(f"exp_time must be non-negative or None, got {exp_time}")
    if rank_ratio is not None and not 0 < rank_ratio <= 1:
        raise ValueError(f"rank_ratio must be in (0, 1] or None, got {rank_ratio}")
    exclude_keys = set(exclude_keys or [])

    merged_tv = {}
    for key in tqdm(task_vectors[0], desc="SWUDI Merging", leave=False):
        tv = task_vectors[0][key]
        if tv.dim() == 2 and key not in exclude_keys:
            stacked = torch.stack(
                [
                    t[key].to(device=accelerator, dtype=torch.float32)
                    for t in task_vectors
                ]
            )
            merged_tv[key] = _spectral_merge(stacked, exp_time, rank_ratio).to(
                device=tv.device, dtype=tv.dtype
            )
        else:
            merged_tv[key] = torch.stack([t[key] for t in task_vectors]).mean(dim=0)
    return merged_tv


@auto_register_config
class SWUDIMerging(
    LightningFabricMixin,
    BaseAlgorithm,
):
    """
    SWUDI / ASWUDI: closed-form spectral regularization for data-free model merging.

    Args:
        exp_time: Time ``t`` of the soft filter ``1 - exp(-t * lambda)``; ``None`` disables it.
        rank_ratio: Fraction of eigendirections kept in every linear layer;
            ``None`` selects the rank of each layer adaptively.
        scaling_factor: Multiplier of the merged task vector before adding it to the pretrained model.
        exclude_keys: 2-D parameters that are averaged instead of solved in closed form.

    ``exp_time=None, rank_ratio=None`` gives ASWUDI.
    """

    def __init__(
        self,
        exp_time: Optional[float] = 1300.0,
        rank_ratio: Optional[float] = 0.65,
        scaling_factor: float = 1.0,
        exclude_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

    @torch.no_grad()
    def run(self, modelpool: Union[BaseModelPool, Dict[str, nn.Module]]) -> nn.Module:
        if not isinstance(modelpool, BaseModelPool):
            modelpool = BaseModelPool(modelpool)

        log.info(
            f"Fusing {len(modelpool.model_names)} models with SWUDI "
            f"(exp_time={self.exp_time}, rank_ratio={self.rank_ratio})."
        )
        pretrained_model = modelpool.load_pretrained_model()
        pretrained_sd = pretrained_model.state_dict()
        keys = [k for k, v in pretrained_sd.items() if v.is_floating_point()]

        task_vectors = []
        for model_name in modelpool.model_names:
            finetuned_sd = modelpool.load_model(model_name).state_dict()
            task_vectors.append({k: finetuned_sd[k] - pretrained_sd[k] for k in keys})
            del finetuned_sd

        merged_tv = swudi_merging(
            task_vectors,
            exp_time=self.exp_time,
            rank_ratio=self.rank_ratio,
            exclude_keys=self.exclude_keys,
            accelerator=self.fabric.device,
        )

        merged_sd = dict(pretrained_sd)
        for k in keys:
            merged_sd[k] = pretrained_sd[k] + self.scaling_factor * merged_tv[k]
        pretrained_model.load_state_dict(merged_sd)
        return pretrained_model
