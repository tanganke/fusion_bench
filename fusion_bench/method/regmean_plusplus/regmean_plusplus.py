import logging
import re
from collections import defaultdict
from typing import Dict, List, cast

import torch
from torch import Tensor, nn
from tqdm.autonotebook import tqdm

import fusion_bench.method.regmean.utils as regmean_utils
from fusion_bench import BaseAlgorithm, auto_register_config
from fusion_bench.mixins import SimpleProfilerMixin
from fusion_bench.modelpool import BaseModelPool

log = logging.getLogger(__name__)


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


def regmean_params_merge(
    param_weight_list: List[Tensor],
    param_regmean_list: List[Tensor],
    reduce_non_diagonal_ratio: float = 1.0,
    weight_transpose: bool = True,
    module_name: str = "",
    device="cpu",
):
    # two lists with length num_models_to_merge
    param_multiplied_results, module_regmean_weights_list = [], []
    for model_idx, module_regmean_weights in enumerate(param_regmean_list):
        # reduce non-diagonal elements
        module_regmean_weights = reduce_non_diagonal_elements(
            regmean_weights=module_regmean_weights,
            reduce_non_diagonal_ratio=reduce_non_diagonal_ratio,
        )
        module_regmean_weights_list.append(module_regmean_weights)

        model_to_merge_param = param_weight_list[model_idx]
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
    inv_sum_module_regmean_weights = torch.inverse(sum_module_regmean_weights)
    # merge parameters with regmean
    merged_param = torch.matmul(
        inv_sum_module_regmean_weights, sum_param_multiplied_results
    )
    # transpose to the original shape of "weight" in Linear module
    merged_param = merged_param.transpose(0, 1) if weight_transpose else merged_param

    return merged_param


def merging_with_regmean_weights(
    models_to_merge_param_dict: dict,
    models_to_merge_regmean_weights_list: list,
    reduce_non_diagonal_ratio: float = 1.0,
    weight_transpose: bool = True,
):
    """
    merge parameters of different models with computed regmean weights

    Asrgs:
        models_to_merge_param_dict: dict, dictionary of list, where key is the parameter name,
            value is a list of the corresponding parameters of all the models that need to be merged
        models_to_merge_regmean_weights_list: list, list of dictionaries with length len(models_to_merge),
            each dictionary records the regmean weights (matrix) of parameters for each model that needs to be merged, key is module name
        reduce_non_diagonal_ratio: float, reduce non-diagonal elements in regmean weights by multiplying this scalar

    Returns:
        dict: merged model parameters
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
                module_regmean_weights_list = []
                for model_idx, model_to_merge_regmean_weights in enumerate(
                    models_to_merge_regmean_weights_list
                ):
                    device = param_value_list[model_idx].device

                    # Tensor, shape (hidden_dim, hidden_dim)
                    module_regmean_weights = model_to_merge_regmean_weights[
                        module_name
                    ].to(device)
                    module_regmean_weights_list.append(module_regmean_weights)

                merged_params[param_name] = regmean_params_merge(
                    param_weight_list=param_value_list,
                    param_regmean_list=module_regmean_weights_list,
                    reduce_non_diagonal_ratio=reduce_non_diagonal_ratio,
                    weight_transpose=weight_transpose,
                    module_name=module_name,
                    device=device,
                )

                merged_by_regmean = True
        # use average merging for parameters whose names are not end with ".weight" or not in Linear module
        if not merged_by_regmean:
            merged_params[param_name] = torch.stack(param_value_list, dim=0).mean(dim=0)

    return merged_params


@auto_register_config
class RegMeanAlgorithmPlusPlus(
    SimpleProfilerMixin,
    BaseAlgorithm,
):
    _include_module_type = [nn.Linear]

    def __init__(
        self,
        *,
        num_regmean_examples: int,
        exclude_param_names_regex: list,
        reduce_non_diagonal_ratio: float,
        weight_transpose: bool,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_regmean_examples = num_regmean_examples
        self.exclude_param_names_regex = exclude_param_names_regex
        self.reduce_non_diagonal_ratio = reduce_non_diagonal_ratio
        self.weight_transpose = weight_transpose

    def run(self, modelpool: BaseModelPool, **kwargs):
        if not isinstance(modelpool, BaseModelPool):
            modelpool = BaseModelPool(modelpool)
        self.modelpool = modelpool
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        models_to_merge_dict = {
            name: model.to(device) for name, model in modelpool.named_models()
        }
        self.on_regmean_start()

        # initialize the merged models as the pretrained model
        merged_model = modelpool.load_pretrained_model().to(device)
        merged_params_dict = {}

        # 1. merge embedding layer
        merged_embedding_dict = self.merge_embedding_layer(
            models_to_merge_dict=models_to_merge_dict
        )
        merged_model.load_state_dict(merged_embedding_dict, strict=False)

        with torch.no_grad():
            # 1.1. compute input for the first layer
            with (
                self.profile("merging models"),
                self.profile("computing first layer input"),
            ):
                batches_input_dict = defaultdict(list)
                for name in tqdm(
                    models_to_merge_dict.keys(), desc="computing input for first layer"
                ):
                    dataset = modelpool.load_train_dataset(name)

                    batches_input_dict[name] = self.get_input_for_first_layer(
                        merged_model, dataset
                    )

            # 2. iteratively merge layer by layer with regmean algorithm
            backbone_layers = self.get_layers(merged_model)
            num_layers = len(backbone_layers)

            models_to_merge_layers_dict = defaultdict(list)
            for name, model in models_to_merge_dict.items():
                models_to_merge_layers_dict[name] = self.get_layers(model)

            param_names_to_merge = None
            for layer_idx, backbone_layer in tqdm(
                enumerate(backbone_layers), desc="merging layers", total=num_layers
            ):
                # dictionary of list, where key is the parameter name,
                # value is a list of the corresponding parameters of all the models that need to be merged
                models_to_merge_param_dict = defaultdict(list)

                # list of dictionaries with length len(models_to_merge),
                # each dictionary records the regmean weights (matrix) of parameters for each model that needs to be merged
                models_to_merge_regmean_weights_list = []

                for name, layers_to_merge in models_to_merge_layers_dict.items():
                    layer_to_merge = layers_to_merge[layer_idx]
                    param_dict = layer_to_merge.state_dict()

                    # exclude parameter whose name matches element in exclude_param_names_regex
                    if param_names_to_merge is None:
                        param_names_to_merge = regmean_utils.get_param_names_to_merge(
                            input_param_names=list(param_dict.keys()),
                            exclude_param_names_regex=self.config.get(
                                "exclude_param_names_regex", []
                            ),
                        )

                    for param_name in param_names_to_merge:
                        models_to_merge_param_dict[param_name].append(
                            param_dict[param_name]
                        )

                    linear_modules_to_merge = regmean_utils.get_modules_to_merge(
                        model=layer_to_merge,
                        include_module_types=self._include_module_type,
                    )
                    assert (
                        len(linear_modules_to_merge) > 0
                    ), "No linear modules to merge"

                    # 2.1. compute regmean weights for each model
                    with (
                        self.profile("merging models"),
                        self.profile("computing regmean weights"),
                    ):
                        regmean_weights = self.get_regmean_weights(
                            name,
                            layer_to_merge,
                            batches_input=batches_input_dict[name],
                            linear_modules_to_merge=linear_modules_to_merge,
                        )

                        module_subset = regmean_utils.get_param_names_to_merge(
                            input_param_names=list(param_dict.keys()),
                            exclude_param_names_regex=self.exclude_param_names_regex,
                        )
                        module_subset = [
                            name.replace(".weight", "").replace(".bias", "")
                            for name in module_subset
                        ]
                        module_subset = list(set(module_subset))
                        regmean_weights = {
                            module_name: regmean_weights[module_name]
                            for module_name in module_subset
                            if module_name in regmean_weights
                        }

                        models_to_merge_regmean_weights_list.append(regmean_weights)

                # 2.2. merge parameters with regmean weights
                with self.profile("merging models"):
                    # merging with regmean weights
                    merged_layer_params = merging_with_regmean_weights(
                        models_to_merge_param_dict=models_to_merge_param_dict,
                        models_to_merge_regmean_weights_list=models_to_merge_regmean_weights_list,
                        reduce_non_diagonal_ratio=self.reduce_non_diagonal_ratio,
                        weight_transpose=self.config.get("weight_transpose", True),
                    )

                    merged_params_dict = self.update_merged_params_dict(
                        merged_params_dict=merged_params_dict,
                        new_merged_params=merged_layer_params,
                        layer_idx=layer_idx,
                    )

                # 2.3. compute input for the next layer
                with (
                    self.profile("merging models"),
                    self.profile("forwarding next layer"),
                ):
                    if layer_idx < num_layers - 1:
                        backbone_layer.load_state_dict(
                            merged_layer_params, strict=False
                        )
                        batches_output_dict = defaultdict(list)
                        for name in models_to_merge_dict.keys():
                            batches_output_dict[name] = self.layer_batches_forward(
                                backbone_layer, batches_input_dict[name]
                            )
                        batches_input_dict = batches_output_dict

            # 3. load state dict to the merged model
            merged_model.load_state_dict(merged_params_dict, strict=False)

        self.print_profile_summary()
        return merged_model

    def merge_embedding_layer(self, models_to_merge_dict: Dict[str, nn.Module]):
        """
        Merge the embedding layer of the model with the merged model.
        This method should be implemented in subclasses if needed.
        """
        raise NotImplementedError()

    def get_input_for_first_layer(self, model: nn.Module, train_dataset):
        raise NotImplementedError

    def get_layers(self, model: nn.Module):
        raise NotImplementedError

    def update_merged_params_dict(
        self, merged_params_dict, new_merged_params, layer_idx
    ):
        raise NotImplementedError

    def layer_batches_forward(self, layer: nn.Module, batches_input: List[Tensor]):
        raise NotImplementedError

    def on_regmean_start(self):
        pass

    def get_regmean_weights(
        self,
        model_name: str,
        layer: nn.Module,
        batches_input: List[Tensor],
        linear_modules_to_merge: Dict[str, nn.Module],
    ):
        raise NotImplementedError
