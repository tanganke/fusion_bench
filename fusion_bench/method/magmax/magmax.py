R"""
MagMax: Leveraging Model Merging for Seamless Continual Learning.

Reference:
    Marczak, D., Twardowski, B., Trzcinski, T., Cygert, S.
    "MagMax: Leveraging Model Merging for Seamless Continual Learning."
    ECCV 2024.
    https://arxiv.org/abs/2407.06322

The MagMax merging rule is element-wise maximum-magnitude task-vector
selection:

    For each task t in {1, ..., T} compute the task vector
        tau_t = theta_t - theta_0
    where theta_t is a fine-tuned model and theta_0 is the pretrained model.

    For every parameter index j choose the task vector with the largest
    absolute value at that position:
        tau_max[j] = tau_{t*(j)}[j],   t*(j) = argmax_t |tau_t[j]|

    The merged model is then:
        theta_merged = theta_0 + scaling_factor * tau_max

This contrasts with Task Arithmetic, which sums all task vectors instead of
selecting per-element by magnitude.

Implementation notes (matched to the official release at
https://github.com/danielm1405/magmax):
- Tie-breaking is done with ``>=``, i.e. later task vectors win when their
  absolute magnitude equals the current best (see
  ``src/merging/task_vectors.py::merge_max_abs``).
- Integer / boolean buffer keys (e.g. ``position_ids``) are skipped during
  task-vector computation and copied through from the pretrained state.
- The released 8-dataset reproduction script uses ``scaling_coef = 0.5``
  for independent fine-tunes; we expose this as ``scaling_factor`` and use
  it as the default.
"""

import logging
from copy import deepcopy
from typing import Dict, List, Mapping, Optional, Union

import torch
from torch import Tensor, nn

from fusion_bench import LazyStateDict
from fusion_bench.method.base_algorithm import BaseAlgorithm
from fusion_bench.mixins import SimpleProfilerMixin, auto_register_config
from fusion_bench.modelpool import BaseModelPool
from fusion_bench.utils.type import StateDictType, TorchModelType

log = logging.getLogger(__name__)


_SKIP_DTYPES = (torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8, torch.bool)


def _is_float_param(t: Tensor) -> bool:
    """Return True if a parameter is a floating-point tensor we should merge.

    Mirrors the reference implementation, which skips integer / boolean
    buffers such as ``position_ids``.
    """
    return t.dtype not in _SKIP_DTYPES


@torch.no_grad()
def _magmax_select(
    task_vectors: List[StateDictType],
) -> StateDictType:
    """Element-wise maximum-magnitude selection across task vectors.

    For each parameter tensor key, fold the task vectors in order using the
    ``>=`` tie-breaking rule from the reference implementation: a later task
    vector replaces the running selection when its absolute value at a given
    position is greater than or equal to the current best.
    """
    if len(task_vectors) == 0:
        raise ValueError("magmax requires at least one task vector.")

    merged: Dict[str, Tensor] = {k: v.clone() for k, v in task_vectors[0].items()}
    for tv in task_vectors[1:]:
        for key, current in tv.items():
            running = merged[key]
            mask = current.abs() >= running.abs()
            merged[key] = torch.where(mask, current, running)
    return merged


@torch.no_grad()
def magmax_merge(
    pretrained_model: TorchModelType,
    finetuned_models: List[TorchModelType],
    scaling_factor: float = 1.0,
    inplace: bool = True,
) -> TorchModelType:
    """Merge fine-tuned models into the pretrained model using MagMax.

    Args:
        pretrained_model: Base / pretrained model that supplies theta_0.
        finetuned_models: A list of task-specific fine-tuned models. Each
            must have the same parameter shapes as ``pretrained_model``.
        scaling_factor: Multiplier applied to the merged task vector before
            adding it back to the pretrained weights.
        inplace: If True, mutate ``pretrained_model``; otherwise operate on
            a deep copy.

    Returns:
        The merged model with the MagMax task vector applied.
    """
    if not inplace:
        pretrained_model = deepcopy(pretrained_model)

    ptm_sd = pretrained_model.state_dict(keep_vars=True)
    merge_keys = [k for k, v in ptm_sd.items() if _is_float_param(v)]
    task_vectors: List[StateDictType] = []
    for model in finetuned_models:
        ft_sd = model.state_dict(keep_vars=True)
        task_vectors.append({k: ft_sd[k] - ptm_sd[k] for k in merge_keys})

    tau_max = _magmax_select(task_vectors)
    merged_sd = dict(ptm_sd)
    for k in merge_keys:
        merged_sd[k] = ptm_sd[k] + scaling_factor * tau_max[k]
    pretrained_model.load_state_dict(merged_sd)
    return pretrained_model


@auto_register_config
class MagMaxAlgorithm(
    SimpleProfilerMixin,
    BaseAlgorithm,
):
    """MagMax model merging algorithm.

    Computes per-task task vectors against the pretrained model, selects
    element-wise the value with the maximum absolute magnitude across
    tasks, scales by ``scaling_factor`` and adds the result back to the
    pretrained weights.

    Attributes:
        scaling_factor: Multiplier applied to the merged task vector before
            adding it back to the pretrained model (alpha in the paper).
        inplace: If True, mutate the loaded pretrained model in place.
    """

    def __init__(
        self,
        scaling_factor: float = 0.5,
        inplace: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

    @torch.no_grad()
    def run(self, modelpool: Union[BaseModelPool, Dict[str, nn.Module]]) -> nn.Module:
        if not isinstance(modelpool, BaseModelPool):
            modelpool = BaseModelPool(modelpool)

        log.info(
            "Fusing %d models using MagMax (scaling_factor=%s).",
            len(modelpool.model_names),
            self.scaling_factor,
        )

        with self.profile("load pretrained"):
            pretrained_model = modelpool.load_model("_pretrained_")
            ptm_sd: Mapping[str, Tensor] = pretrained_model.state_dict(keep_vars=True)
            merge_keys = [k for k, v in ptm_sd.items() if _is_float_param(v)]

        # Stream task vectors: hold only the running max-magnitude selection
        # in memory instead of all task vectors. Tie-breaking uses ``>=`` to
        # match the reference implementation.
        tau_max: Optional[Dict[str, Tensor]] = None
        for model_name in modelpool.model_names:
            with self.profile("load model"):
                model = modelpool.load_model(model_name)
            with self.profile("merge weights"):
                ft_sd = model.state_dict(keep_vars=True)
                if tau_max is None:
                    tau_max = {k: (ft_sd[k] - ptm_sd[k]).clone() for k in merge_keys}
                else:
                    for k in merge_keys:
                        new_tv = ft_sd[k] - ptm_sd[k]
                        mask = new_tv.abs() >= tau_max[k].abs()
                        tau_max[k] = torch.where(mask, new_tv, tau_max[k])

        assert tau_max is not None, "modelpool produced no finetuned models"

        with self.profile("merge weights"):
            merged_sd = dict(ptm_sd)
            for k in merge_keys:
                merged_sd[k] = ptm_sd[k] + self.scaling_factor * tau_max[k]

        self.print_profile_summary()

        if isinstance(pretrained_model, nn.Module):
            model = pretrained_model if self.inplace else deepcopy(pretrained_model)
            model.load_state_dict(merged_sd)
        elif isinstance(pretrained_model, LazyStateDict):
            model = deepcopy(pretrained_model.meta_module)
            model = model.to_empty(device=pretrained_model._device)
            result = model.load_state_dict(merged_sd, strict=False)
            if result.unexpected_keys:
                raise ValueError(
                    f"Unexpected keys in state dict: {result.unexpected_keys}"
                )
            if result.missing_keys:
                log.warning(f"Missing keys in state dict: {result.missing_keys}")
        else:
            raise TypeError(f"Unsupported model type: {type(pretrained_model)}")
        return model
