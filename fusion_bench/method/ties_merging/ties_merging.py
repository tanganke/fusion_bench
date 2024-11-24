import logging
from typing import Dict, List, Literal, Mapping, Union  # noqa: F401

import torch
from torch import Tensor, nn

from fusion_bench.compat.modelpool import to_modelpool
from fusion_bench.method import BaseAlgorithm
from fusion_bench.modelpool import BaseModelPool
from fusion_bench.utils.type import StateDictType

from .ties_merging_utils import state_dict_to_vector, ties_merging, vector_to_state_dict

log = logging.getLogger(__name__)


class TiesMergingAlgorithm(BaseAlgorithm):
    """
    TiesMergingAlgorithm is a class for fusing multiple models using the TIES merging technique.

    Attributes:
        scaling_factor (float): The scaling factor to apply to the merged task vector.
        threshold (float): The threshold for resetting values in the task vector.
        remove_keys (List[str]): List of keys to remove from the state dictionary.
        merge_func (Literal["sum", "mean", "max"]): The merge function to use for disjoint merging.
    """

    _config_mapping = BaseAlgorithm._config_mapping | {
        "scaling_factor": "scaling_factor",
        "threshold": "threshold",
        "remove_keys": "remove_keys",
        "merge_func": "merge_func",
    }

    def __init__(
        self,
        scaling_factor: float,
        threshold: float,
        remove_keys: List[str],
        merge_func: Literal["sum", "mean", "max"],
        **kwargs,
    ):
        """
        Initialize the TiesMergingAlgorithm with the given parameters.

        Args:
            scaling_factor (float): The scaling factor to apply to the merged task vector.
            threshold (float): The threshold for resetting values in the task vector.
            remove_keys (List[str]): List of keys to remove from the state dictionary.
            merge_func (Literal["sum", "mean", "max"]): The merge function to use for disjoint merging.
            **kwargs: Additional keyword arguments for the base class.
        """
        self.scaling_factor = scaling_factor
        self.threshold = threshold
        self.remove_keys = remove_keys
        self.merge_func = merge_func
        super().__init__(**kwargs)

    @torch.no_grad()
    def run(self, modelpool: BaseModelPool | Dict[str, nn.Module], **kwargs):
        """
        Run the TIES merging algorithm to fuse models in the model pool.

        Args:
            modelpool (BaseModelPool | Dict[str, nn.Module]): The model pool containing the models to fuse.

        Returns:
            nn.Module: The fused model.
        """
        log.info("Fusing models using ties merging.")
        modelpool = to_modelpool(modelpool)
        remove_keys = self.config.get("remove_keys", [])
        merge_func = self.config.get("merge_func", "sum")
        scaling_factor = self.scaling_factor
        threshold = self.threshold

        # Load the pretrained model
        pretrained_model = modelpool.load_model("_pretrained_")

        # Load the state dicts of the models
        ft_checks: List[StateDictType] = [
            modelpool.load_model(model_name).state_dict(keep_vars=True)
            for model_name in modelpool.model_names
        ]
        ptm_check: StateDictType = pretrained_model.state_dict(keep_vars=True)

        # Compute the task vectors
        flat_ft: Tensor = torch.vstack(
            [state_dict_to_vector(check, remove_keys) for check in ft_checks]
        )
        flat_ptm: Tensor = state_dict_to_vector(ptm_check, remove_keys)
        tv_flat_checks = flat_ft - flat_ptm

        # Perform TIES Merging
        merged_tv = ties_merging(
            tv_flat_checks,
            reset_thresh=threshold,
            merge_func=merge_func,
        )
        merged_check = flat_ptm + scaling_factor * merged_tv
        merged_state_dict = vector_to_state_dict(
            merged_check, ptm_check, remove_keys=remove_keys
        )

        # Load the merged state dict into the pretrained model
        pretrained_model.load_state_dict(merged_state_dict)
        return pretrained_model
