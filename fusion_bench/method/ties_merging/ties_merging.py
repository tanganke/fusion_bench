import logging
from copy import deepcopy
from typing import Dict, List, Mapping, Union

import torch
from torch import Tensor, nn

from ...modelpool import ModelPool, to_modelpool
from ...utils.type import _StateDict
from ..base_algorithm import ModelFusionAlgorithm
from .ties_merging_utils import state_dict_to_vector, ties_merging, vector_to_state_dict

log = logging.getLogger(__name__)


class TiesMergingAlgorithm(ModelFusionAlgorithm):
    @torch.no_grad()
    def run(self, modelpool: ModelPool | Dict[str, nn.Module]):
        log.info("Fusing models using ties merging.")
        modelpool = to_modelpool(modelpool)
        remove_keys = self.config.get("remove_keys", [])
        merge_func = self.config.get("merge_func", "sum")
        scaling_factor = self.config.scaling_factor
        threshold = self.config.threshold

        pretrained_model = modelpool.load_model("_pretrained_")

        # load the state dicts of the models
        ft_checks: List[_StateDict] = [
            modelpool.load_model(model_name).state_dict(keep_vars=True)
            for model_name in modelpool.model_names
        ]
        ptm_check: _StateDict = pretrained_model.state_dict(keep_vars=True)

        # compute the task vectors
        flat_ft = torch.vstack(
            [state_dict_to_vector(check, remove_keys) for check in ft_checks]
        )
        flat_ptm = state_dict_to_vector(ptm_check, remove_keys)
        tv_flat_checks = flat_ft - flat_ptm

        # Ties Merging
        merged_tv = ties_merging(
            tv_flat_checks,
            reset_thresh=threshold,
            merge_func=merge_func,
        )
        merged_check = flat_ptm + scaling_factor * merged_tv
        merged_state_dict = vector_to_state_dict(
            merged_check, ptm_check, remove_keys=remove_keys
        )
        merged_state_dict = vector_to_state_dict(
            merged_check, ptm_check, remove_keys=remove_keys
        )

        pretrained_model.load_state_dict(merged_state_dict)
        return pretrained_model
