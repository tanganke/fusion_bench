"""
This file contains the implementation of the regmean method for model merging.
modified from https://github.com/yule-BUAA/MergeLM/blob/6d49ad96fd69c92013654b837041b868aa806564/model_merging_methods/merging_methods.py
"""

import logging
import re
from collections import defaultdict
from typing import Dict, List, cast

import torch
from torch import Tensor, nn
from tqdm.autonotebook import tqdm

from fusion_bench.method import BaseAlgorithm
from fusion_bench.mixins import SimpleProfilerMixin
from fusion_bench.modelpool import BaseModelPool

log = logging.getLogger(__name__)


def get_param_names_to_merge(
    input_param_names: List[str], exclude_param_names_regex: list
):
    """
    get the names of parameters that need to be merged
    :param input_param_names: list, names of input parameters
    :param exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
    :return:
    """
    param_names_to_merge = []
    for param_name in input_param_names:
        exclude = any(
            [
                re.match(exclude_pattern, param_name)
                for exclude_pattern in exclude_param_names_regex
            ]
        )
        if not exclude:
            param_names_to_merge.append(param_name)
    return param_names_to_merge


def get_modules_to_merge(model: nn.Module, include_module_types: list):
    """
    get the model modules that need to be merged, whose type is in include_module_types
    :param model: nn.Module, input model
    :param include_module_types: list, module types that want to include
    :return:
    """
    modules_to_merge: Dict[str, nn.Module] = {}
    for module_name, module in model.named_modules():
        is_valid_type = not include_module_types or any(
            [
                isinstance(module, include_module_type)
                for include_module_type in include_module_types
            ]
        )
        if is_valid_type:
            modules_to_merge[module_name] = module
    return modules_to_merge


def reduce_non_diagonal_elements(
    regmean_weights: torch.Tensor, reduce_non_diagonal_ratio: float
):
    """
    reduce the non-diagonal elements in regmean_weights
    :param regmean_weights: Tensor, shape (hidden_dim, hidden_dim), input regmean weights
    :param reduce_non_diagonal_ratio: float, reduce non-diagonal elements in regmean weights by multiplying this scalar
    :return:
    """
    # diagonal matrix with (1 - reduce_non_diagonal_ratio) as elements
    diag_weights = torch.diag(
        torch.ones(regmean_weights.shape[0]) - reduce_non_diagonal_ratio
    ).to(regmean_weights.device)
    # matrix with reduce_non_diagonal_ratio as elements
    non_diag_weights = torch.zeros_like(diag_weights).fill_(reduce_non_diagonal_ratio)
    # diagonal elements are unchanged, while non-diagonal elements are multiplied by reduce_non_diagonal_ratio
    return regmean_weights * (diag_weights + non_diag_weights)


def merging_with_regmean_weights(
    models_to_merge_param_dict: dict,
    models_to_merge_regmean_weights_list: list,
    reduce_non_diagonal_ratio: float = 1.0,
    weight_transpose: bool = True,
):
    """
    merge parameters of different models with computed regmean weights
    :param models_to_merge_param_dict: dict, dictionary of list, where key is the parameter name,
    value is a list of the corresponding parameters of all the models that need to be merged
    :param models_to_merge_regmean_weights_list: list, list of dictionaries with length len(models_to_merge),
    each dictionary records the regmean weights (matrix) of parameters for each model that needs to be merged, key is module name
    :param reduce_non_diagonal_ratio: float, reduce non-diagonal elements in regmean weights by multiplying this scalar
    :return:
    """
    # dict, dictionary of model parameters
    merged_params = {}

    for param_name, param_value_list in models_to_merge_param_dict.items():
        merged_by_regmean = False
        # only perform regmean merging on the "weight" parameter of Linear module
        if param_name.endswith(".weight"):
            module_name = param_name[: -len(".weight")]
            if module_name in models_to_merge_regmean_weights_list[0].keys():
                # two lists with length num_models_to_merge
                param_multiplied_results, module_regmean_weights_list = [], []
                for model_idx, model_to_merge_regmean_weights in enumerate(
                    models_to_merge_regmean_weights_list
                ):
                    # Tensor, shape (hidden_dim, hidden_dim)
                    module_regmean_weights = model_to_merge_regmean_weights[module_name]

                    # reduce non-diagonal elements
                    module_regmean_weights = reduce_non_diagonal_elements(
                        regmean_weights=module_regmean_weights,
                        reduce_non_diagonal_ratio=reduce_non_diagonal_ratio,
                    )
                    module_regmean_weights_list.append(module_regmean_weights)

                    model_to_merge_param = param_value_list[model_idx]
                    # since the weight shape of Linear module is (output_size, input_size), we need to transpose it
                    param_multiplied_results.append(
                        torch.matmul(
                            module_regmean_weights,
                            (
                                model_to_merge_param.transpose(0, 1)
                                if weight_transpose
                                else model_to_merge_param
                            ),
                        )
                    )

                # sum up module_regmean_weights and param_multiplied_results over all individual models
                sum_module_regmean_weights = sum(module_regmean_weights_list)
                sum_param_multiplied_results = sum(param_multiplied_results)

                # get the inverse matrix
                inv_sum_module_regmean_weights = torch.inverse(
                    sum_module_regmean_weights
                )
                # merge parameters with regmean
                merged_param = torch.matmul(
                    inv_sum_module_regmean_weights, sum_param_multiplied_results
                )
                # transpose to the original shape of "weight" in Linear module
                merged_params[param_name] = (
                    merged_param.transpose(0, 1) if weight_transpose else merged_param
                )
                merged_by_regmean = True
        # use average merging for parameters whose names are not end with ".weight" or not in Linear module
        if not merged_by_regmean:
            merged_params[param_name] = torch.stack(param_value_list, dim=0).mean(dim=0)

    return merged_params


def regmean_merging(
    models_to_merge: list,
    trainers: list,
    exclude_param_names_regex: list,
    nums_regmean_examples: list,
    reduce_non_diagonal_ratio: float = 1.0,
):
    """
    regmean merging method
    :param models_to_merge: list, individual models that need to be merged
    :param trainers: list, trainers of individual models
    :param exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
    :param nums_regmean_examples: list, numbers of examples to compute regmean weights
    :param reduce_non_diagonal_ratio: float, reduce non-diagonal elements in regmean weights by multiplying this scalar
    :return:
    """

    def compute_regmean_weights(module_name: str):
        """
        compute the regmean weights, a hook function to deal with each module's input
        :param module_name: str, module name
        :return:
        """

        def hook(module: nn.Module, input: tuple, output: torch.Tensor):
            # Tensor, shape (batch_size, sequence_length, hidden_dim)
            x = input[0].detach()
            batch_num_actual_examples = x.shape[0]
            # Tensor, shape (batch_size * sequence_length, hidden_dim)
            x = x.reshape(-1, x.shape[-1])
            # Tensor, shape (hidden_dim, hidden_dim)
            xtx = torch.matmul(x.transpose(0, 1), x)
            # store the averaged weights in regmean_weights
            if module_name not in regmean_weights.keys():
                regmean_weights[module_name] = xtx / x.shape[0]
                num_computed_examples[module_name] = x.shape[0]
                num_actual_examples[module_name] = batch_num_actual_examples
            else:
                regmean_weights[module_name] = (
                    regmean_weights[module_name] * num_computed_examples[module_name]
                    + xtx
                ) / (num_computed_examples[module_name] + x.shape[0])
                num_computed_examples[module_name] += x.shape[0]
                num_actual_examples[module_name] += batch_num_actual_examples

        return hook

    # dictionary of list, where key is the parameter name,
    # value is a list of the corresponding parameters of all the models that need to be merged
    models_to_merge_param_dict = defaultdict(list)

    # list of dictionaries with length len(models_to_merge),
    # each dictionary records the regmean weights (matrix) of parameters for each model that needs to be merged
    models_to_merge_regmean_weights_list = []

    # iterate each individual model that needs to be merged
    with torch.no_grad():
        for model_idx, (model_to_merge, trainer, num_regmean_examples) in enumerate(
            zip(models_to_merge, trainers, nums_regmean_examples)
        ):
            param_dict = {
                param_name: param_value
                for param_name, param_value in model_to_merge.named_parameters()
            }
            # exclude parameter whose name matches element in exclude_param_names_regex
            param_names_to_merge = get_param_names_to_merge(
                input_param_names=list(param_dict.keys()),
                exclude_param_names_regex=exclude_param_names_regex,
            )

            for param_name in param_names_to_merge:
                models_to_merge_param_dict[param_name].append(param_dict[param_name])

            linear_modules_to_merge = get_modules_to_merge(
                model=model_to_merge, include_module_types=[nn.Linear]
            )
            handles = []
            # dictionary, regmean matrices for each linear module inputs
            regmean_weights = {}
            # dictionary, number of examples (multiplied the sequence length) used for computing regmean matrices
            num_computed_examples = {}
            # dictionary, number of actual examples used for computing regmean matrices
            num_actual_examples = {}

            for module_name, linear_module_to_merge in linear_modules_to_merge.items():
                # register a hook in the forward process
                handle = linear_module_to_merge.register_forward_hook(
                    compute_regmean_weights(module_name=module_name)
                )
                handles.append(handle)

            train_dataloader = trainer.get_train_dataloader()
            if num_regmean_examples % trainer._train_batch_size != 0:
                print(
                    f"warning: the number of examples for computing regmean cannot be fully divided by the batch size for model {model_idx}, "
                    "which may lead to a slightly different number of the actually used examples."
                )
            for step, inputs in tqdm(
                enumerate(train_dataloader),
                desc=f"computing regmean weights for model {model_idx}",
            ):
                if (
                    len(num_actual_examples) > 0
                    and list(num_actual_examples.values())[0] >= num_regmean_examples
                ):
                    break
                inputs = trainer._prepare_inputs(inputs)
                outputs = model_to_merge(**inputs)

            models_to_merge_regmean_weights_list.append(regmean_weights)

            # remove the added hook
            for handle in handles:
                handle.remove()
        # merging with regmean weights
        merged_params = merging_with_regmean_weights(
            models_to_merge_param_dict=models_to_merge_param_dict,
            models_to_merge_regmean_weights_list=models_to_merge_regmean_weights_list,
            reduce_non_diagonal_ratio=reduce_non_diagonal_ratio,
        )

    return merged_params


class RegMeanAlgorithm(BaseAlgorithm, SimpleProfilerMixin):
    _include_module_type = [nn.Linear]
    _config_mapping = {
        "num_regmean_examples": "num_regmean_examples",
        "exclude_param_names_regex": "exclude_param_names_regex",
        "reduce_non_diagonal_ratio": "reduce_non_diagonal_ratio",
        "weight_transpose": "weight_transpose",
    }

    def __init__(
        self,
        *,
        num_regmean_examples: int,
        exclude_param_names_regex: list,
        reduce_non_diagonal_ratio: float,
        weight_transpose: bool,
        **kwargs,
    ):
        self.num_regmean_examples = num_regmean_examples
        self.exclude_param_names_regex = exclude_param_names_regex
        self.reduce_non_diagonal_ratio = reduce_non_diagonal_ratio
        self.weight_transpose = weight_transpose
        super().__init__(**kwargs)

    def run(self, modelpool: BaseModelPool, **kwargs):
        if not isinstance(modelpool, BaseModelPool):
            modelpool = BaseModelPool(modelpool)
        self.modelpool = modelpool
        self.on_regmean_start()

        # dictionary of list, where key is the parameter name,
        # value is a list of the corresponding parameters of all the models that need to be merged
        models_to_merge_param_dict = defaultdict(list)

        # list of dictionaries with length len(models_to_merge),
        # each dictionary records the regmean weights (matrix) of parameters for each model that needs to be merged
        models_to_merge_regmean_weights_list = []

        param_names_to_merge = None

        with torch.no_grad():
            for name, model in modelpool.named_models():
                param_dict = model.state_dict()

                # exclude parameter whose name matches element in exclude_param_names_regex
                if param_names_to_merge is None:
                    param_names_to_merge = get_param_names_to_merge(
                        input_param_names=list(param_dict.keys()),
                        exclude_param_names_regex=self.config.get(
                            "exclude_param_names_regex", []
                        ),
                    )

                for param_name in param_names_to_merge:
                    models_to_merge_param_dict[param_name].append(
                        param_dict[param_name]
                    )

                linear_modules_to_merge = get_modules_to_merge(
                    model=model, include_module_types=self._include_module_type
                )
                assert len(linear_modules_to_merge) > 0, "No linear modules to merge"

                with (
                    self.profile("merging models"),
                    self.profile("computing regmean weights"),
                ):
                    regmean_weights = self.get_regmean_weights(
                        name,
                        model,
                        train_dataset=modelpool.load_train_dataset(name),
                        linear_modules_to_merge=linear_modules_to_merge,
                    )
                    models_to_merge_regmean_weights_list.append(regmean_weights)

        with self.profile("merging models"):
            # merging with regmean weights
            merged_params = merging_with_regmean_weights(
                models_to_merge_param_dict=models_to_merge_param_dict,
                models_to_merge_regmean_weights_list=models_to_merge_regmean_weights_list,
                reduce_non_diagonal_ratio=self.reduce_non_diagonal_ratio,
                weight_transpose=self.config.get("weight_transpose", True),
            )

            merged_model = modelpool.load_model("_pretrained_")
            merged_model.load_state_dict(merged_params, strict=False)

        self.print_profile_summary()
        return merged_model

    def on_regmean_start(self):
        pass

    def get_regmean_weights(
        self,
        model_name: str,
        model: nn.Module,
        train_dataset,
        linear_modules_to_merge: Dict[str, nn.Module],
    ):
        raise NotImplementedError
