R"""
Examples:

The following command merges eight clip-ViT models using a weighted average approach.
Because `method.normalize` is set to true, the weights are normalized to sum to 1, thus equivalent to simple average.

```bash
fusion_bench \
    method=linear/weighted_average \
    method.normalize=true \
    method.weights=[0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3] \
    modelpool=... \
    taskpool=...
```
"""

import logging
from typing import List, Mapping, Optional, Union  # noqa: F401

import numpy as np
import torch
from typing_extensions import override

from fusion_bench.method import BaseAlgorithm
from fusion_bench.mixins import SimpleProfilerMixin
from fusion_bench.modelpool import BaseModelPool
from fusion_bench.utils.state_dict_arithmetic import state_dict_add, state_dict_mul
from fusion_bench.utils.type import StateDictType

log = logging.getLogger(__name__)


class WeightedAverageAlgorithm(BaseAlgorithm, SimpleProfilerMixin):

    _config_mapping = BaseAlgorithm._config_mapping | {
        "normalize": "normalize",
        "weights": "weights",
    }

    def __init__(
        self,
        normalize: bool,
        weights: List[float],
        verbose: bool = True,
        **kwargs,
    ):
        self.normalize = normalize
        self.weights = weights
        self.verbose = verbose
        log.disabled = not self.verbose
        super().__init__(**kwargs)

    @override
    @torch.no_grad()
    def run(self, modelpool: BaseModelPool):
        """
        Fuses the models in the model pool using a weighted average approach.

        Parameters
            modelpool (ModelPool): The pool of models to be fused.

        Raises
            ValueError: If the number of weights does not match the number of models in the model pool.

        Returns
            forward_model (torch.nn.Module): The resulting model after fusion.
        """
        if not isinstance(modelpool, BaseModelPool):
            modelpool = BaseModelPool(modelpool)

        log.info("Fusing models using weighted average.")
        weights = np.asarray(self.weights)
        if len(weights) != len(modelpool.model_names):
            raise ValueError(
                "Number of weights must match the number of models.,"
                f"but got {len(weights)} weights and {len(modelpool.model_names)} models."
                f"weights: {weights}, models: {modelpool.model_names}"
            )
        if self.normalize:
            weights = weights / np.sum(weights)
        if self.verbose:
            print(f"weights: {weights}, normalized: {self.normalize}")

        sd: Optional[StateDictType] = None
        forward_model = None

        for model_name, weight in zip(modelpool.model_names, weights):
            with self.profile("load_model"):
                model = modelpool.load_model(model_name)
            with self.profile("merge weights"):
                if sd is None:
                    sd = state_dict_mul(model.state_dict(keep_vars=True), weight)
                    forward_model = model
                else:
                    sd = state_dict_add(
                        sd, state_dict_mul(model.state_dict(keep_vars=True), weight)
                    )

        forward_model.load_state_dict(sd)
        if self.verbose:
            self.print_profile_summary()
        return forward_model
