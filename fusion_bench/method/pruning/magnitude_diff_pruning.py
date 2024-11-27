import functools
import logging
import re
from copy import deepcopy
from typing import Dict, List, Literal, Optional, Union  # noqa: F401

import torch
from torch import Tensor, nn
from tqdm.auto import tqdm

from fusion_bench.method import BaseAlgorithm
from fusion_bench.mixins.simple_profiler import SimpleProfilerMixin
from fusion_bench.modelpool import BaseModelPool

from .prune_utils import unstructured_magnitude_prune_

log = logging.getLogger(__name__)


def _is_name_matched(name: str, extract_names: List[str]):
    """
    Check if the parameter name matches any of the provided regular expressions.

    Args:
        name (str): The name of the parameter.
        extract_names (List[str]): List of regular expressions to match the parameter names.

    Returns:
        bool: True if the name matches any of the regular expressions, False otherwise.
    """
    for extract_name in extract_names:
        # extract_name is a regular expression
        if re.match(extract_name, name):
            return True
    return False


class MagnitudeDiffPruningAlgorithm(
    BaseAlgorithm,
    SimpleProfilerMixin,
):
    """
    Implements magnitude-based pruning on the difference between pretrained and fine-tuned model parameters.

    This class supports pruning the difference between the pretrained and fine-tuned model parameters
    based on their magnitude. It allows specifying the ratio of weights to prune and the names of
    parameters to extract for pruning.

    Methods:
        run(modelpool: BaseModelPool) -> nn.Module:
            Executes the pruning process on the model pool and returns the pruned model.
        magnitude_prune(pretrained_model: nn.Module, finetuned_model: nn.Module, in_place: bool = True) -> nn.Module:
            Prunes the difference between the pretrained and fine-tuned model parameters.
    """

    _config_mapping = BaseAlgorithm._config_mapping | {
        "prune_ratio": "prune_ratio",
        "extract_names": "extract_names",
    }

    def __init__(
        self,
        prune_ratio: float,
        rescale: Optional[Union[bool, float]] = None,
        extract_names: List[str] = None,
        prune_type: Literal["minor", "major"] = "minor",
        **kwargs,
    ):
        """
        Initialize the MagnitudeDiffPruningAlgorithm with the given configuration.

        Args:
            prune_ratio (float): The ratio of weights to prune.
            extract_names (List[str], optional): List of regular expressions to match the parameter names for pruning. Defaults to None.
            **kwargs: Additional keyword arguments.
        """
        self.prune_ratio = prune_ratio
        self.rescale = rescale
        self.extract_names = extract_names
        self.prune_type = prune_type
        super().__init__(**kwargs)

    @torch.no_grad()
    def run(self, modelpool: BaseModelPool):
        """
        Execute the pruning process on the model pool.

        This method loads the pretrained and fine-tuned models from the model pool,
        prunes the difference between their parameters, and returns the pruned model.

        Args:
            modelpool (BaseModelPool): The model pool containing the models to prune.

        Returns:
            nn.Module: The pruned model.
        """
        if not isinstance(modelpool, BaseModelPool):
            modelpool = BaseModelPool(modelpool)

        assert (
            len(modelpool.model_names) == 1
        ), "Only one fine-tuned model is allowed in the model pool."
        with self.profile("load pretrained model"):
            pretrained_model = modelpool.load_model("_pretrained_")
        with self.profile("load fine-tuned model"):
            finetuned_model = modelpool.load_model(modelpool.model_names[0])

        with self.profile("prune model"):
            model = self.magnitude_prune(pretrained_model, finetuned_model)

        self.print_profile_summary()
        return model

    @torch.no_grad()
    def magnitude_prune(
        self,
        pretrained_model: nn.Module,
        finetuned_model: nn.Module,
        in_place: bool = True,
    ):
        """
        Prune the difference between the pretrained and fine-tuned model parameters.

        This method calculates the difference between the pretrained and fine-tuned model parameters,
        prunes the difference based on their magnitude, and updates the pretrained model parameters
        with the pruned difference.

        Args:
            pretrained_model (nn.Module): The pretrained model.
            finetuned_model (nn.Module): The fine-tuned model.
            in_place (bool, optional): Whether to perform the pruning in place. Defaults to True.

        Returns:
            nn.Module: The pruned model.
        """
        if in_place:
            model = pretrained_model
        else:
            model = deepcopy(pretrained_model)

        if self.extract_names is not None:
            extract_names: List[str] = (
                self.extract_names
            )  # regular expressions for the names of the parameters
        else:
            # extract the weight matrix of each linear layer
            extract_names = []
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    extract_names.append(f"{name}.weight")

        ft_state_dict = finetuned_model.state_dict()
        for name, param in tqdm(
            model.named_parameters(),
            "Magnitude Pruning On Parameter Difference",
            total=len(tuple(model.named_parameters())),
        ):
            if not param.requires_grad:
                continue

            # Prune the diff parameter if its name matches
            if _is_name_matched(name, extract_names):
                w_diff = ft_state_dict[name] - param
                w_diff = unstructured_magnitude_prune_(
                    w_diff,
                    (
                        torch.abs
                        if self.prune_type == "minor"
                        else lambda x: -torch.abs(x)
                    ),
                    sparsity_ratio=self.prune_ratio,
                )
                if self.rescale is not None:
                    rescale = (
                        1 / self.prune_ratio if self.rescale == True else self.rescale
                    )
                    w_diff = w_diff * rescale
                param.data = param + w_diff

        return model
