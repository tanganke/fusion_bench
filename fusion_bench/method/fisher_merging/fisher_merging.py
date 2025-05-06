"""
This implementation is largely based on the implementation from  https://github.com/yule-BUAA/MergeLM/
"""

import logging
import re
from collections import defaultdict
from typing import Dict, List

import torch
from torch import Tensor, nn
from tqdm.autonotebook import tqdm

from fusion_bench.method import BaseAlgorithm
from fusion_bench.mixins import SimpleProfilerMixin
from fusion_bench.modelpool import BaseModelPool

log = logging.getLogger(__name__)


def get_param_names_to_merge(
    input_param_names: List[str], exclude_param_names_regex: list
) -> List[str]:
    """
    Get the names of parameters that need to be merged.

    Args:
        input_param_names (List[str]): List of input parameter names.
        exclude_param_names_regex (list): List of regular expressions for parameter names to be excluded.

    Returns:
        List[str]: List of parameter names to be merged.
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


def get_param_squared_gradients(
    model: nn.Module, param_names_to_merge: List[str]
) -> Dict[str, Tensor]:
    """
    Get the squared gradients of parameters.

    Args:
        model (nn.Module): The model.
        param_names_to_merge (List[str]): List of parameter names to be merged.

    Returns:
        Dict[str, Tensor]: Dictionary of parameter names and their squared gradients.
    """
    param_squared_gradients = {
        param_name: param_value.grad.detach() ** 2
        for param_name, param_value in model.state_dict(keep_vars=True).items()
        if param_name in param_names_to_merge
    }
    return param_squared_gradients


def get_models_fisher_norm(
    models_to_merge_param_dict: dict, models_to_merge_fisher_weights_list: list
) -> Tensor:
    """
    Get normalization of Fisher weights of all the models that need to be merged.

    Args:
        models_to_merge_param_dict (dict): Dictionary of list, where key is the parameter name,
                                           value is a list of the corresponding parameters of all the models that need to be merged.
        models_to_merge_fisher_weights_list (list): List of dictionaries with length len(models_to_merge),
                                                    each dictionary records the Fisher weights (matrix or vector) of parameters for each model that needs to be merged.

    Returns:
        Tensor: L2 norm over all the parameters of models that need to be merged.
    """
    # dict, key is parameter name, value is a Tensor with shape (num_models_to_merge, )
    models_fisher_norm_dict = {}
    # compute L2 norm over models for each parameter
    for param_name, _ in models_to_merge_param_dict.items():
        # Tensor, shape (num_models_to_merge, *fisher_weight_shape)
        models_fisher = torch.stack(
            [
                model_to_merge_fisher_weights[param_name]
                for model_to_merge_fisher_weights in models_to_merge_fisher_weights_list
            ],
            dim=0,
        )
        dims = [dim_idx for dim_idx in range(1, models_fisher.dim())]
        # Tensor, shape (num_models_to_merge, ), compute L2 norm for each parameter
        models_fisher_norm = torch.linalg.vector_norm(models_fisher, dim=dims)
        models_fisher_norm_dict[param_name] = models_fisher_norm

    # Tensor, shape (num_models_to_merge, num_parameters)
    models_fisher_norm = torch.stack(
        [models_fisher_norm for models_fisher_norm in models_fisher_norm_dict.values()],
        dim=1,
    )
    # Tensor, shape (num_models_to_merge, ), compute L2 norm over all the parameters
    models_fisher_norm = torch.norm(models_fisher_norm, dim=1)
    return models_fisher_norm


def merging_with_fisher_weights(
    models_to_merge_param_dict: Dict[str, List[Tensor]],
    models_to_merge_fisher_weights_list: list,
    fisher_scaling_coefficients: torch.Tensor,
    normalize_fisher_weight: bool = True,
    minimal_fisher_weight: float = 1e-6,
) -> Dict[str, Tensor]:
    """
    Merge parameters of different models with computed Fisher weights.

    Args:
        models_to_merge_param_dict (Dict[str, List[Tensor]]): Dictionary of list, where key is the parameter name,
                                                              value is a list of the corresponding parameters of all the models that need to be merged.
        models_to_merge_fisher_weights_list (list): List of dictionaries with length len(models_to_merge),
                                                    each dictionary records the Fisher weights (matrix or vector) of parameters for each model that needs to be merged.
        fisher_scaling_coefficients (torch.Tensor): Scaling coefficients to merge Fisher weights.
        normalize_fisher_weight (bool): Whether to normalize Fisher weights (L2 norm) or not.
        minimal_fisher_weight (float): The minimal value in Fisher weights, used for tackling the potential numerical issues.

    Returns:
        Dict[str, Tensor]: Dictionary of merged parameters.
    """
    # dict, dictionary of model parameters
    merged_params = {}

    if normalize_fisher_weight:
        # Tensor, shape (num_models_to_merge, ), L2 norm over all the parameters of models that need to be merged
        models_fisher_norm = get_models_fisher_norm(
            models_to_merge_param_dict=models_to_merge_param_dict,
            models_to_merge_fisher_weights_list=models_to_merge_fisher_weights_list,
        )

    for param_name, param_value_list in models_to_merge_param_dict.items():
        # shape (num_models_to_merge, *parameter_shape)
        param_values = torch.stack(param_value_list, dim=0)
        # Tensor, shape (num_models_to_merge, *fisher_weight_shape), use minimal_fisher_weight to solve the potential numerical issues
        models_to_merge_fisher_weights = (
            torch.stack(
                [
                    model_to_merge_fisher_weights[param_name]
                    for model_to_merge_fisher_weights in models_to_merge_fisher_weights_list
                ],
                dim=0,
            )
            + minimal_fisher_weight
        )

        # Tensor, shape (num_models_to_merge, 1, 1, ...)
        reshaped_scaling_coefficients = fisher_scaling_coefficients.reshape(
            -1, *[1 for _ in range(param_values.dim() - 1)]
        ).to(param_values.device)

        if normalize_fisher_weight:
            # Tensor, shape (num_models_to_merge, )
            _models_fisher_norm = 1.0 / (models_fisher_norm + minimal_fisher_weight)
            normalized_models_fisher_norm = (
                _models_fisher_norm / _models_fisher_norm.sum()
            )
            normalized_models_fisher_norm = normalized_models_fisher_norm.reshape(
                -1, *[1 for _ in range(param_values.dim() - 1)]
            )
            reshaped_scaling_coefficients = (
                reshaped_scaling_coefficients * normalized_models_fisher_norm
            )

        # shape (*parameter_shape)
        numerator = (
            reshaped_scaling_coefficients
            * models_to_merge_fisher_weights
            * param_values
        ).sum(dim=0)

        # shape (*parameter_shape)
        denominator = (
            reshaped_scaling_coefficients * models_to_merge_fisher_weights
        ).sum(dim=0)

        merged_param = numerator / denominator
        merged_params[param_name] = merged_param
    return merged_params


def fisher_merging(
    models_to_merge: List[nn.Module],
    trainers: list,
    exclude_param_names_regex: list,
    nums_fisher_examples: List[int],
    fisher_scaling_coefficients: list = None,
    normalize_fisher_weight: bool = True,
    minimal_fisher_weight: float = 1e-6,
) -> Dict[str, Tensor]:
    """
    Fisher merging method.

    Args:
        models_to_merge (List[nn.Module]): List of individual models that need to be merged.
        trainers (list): List of trainers of individual models.
        exclude_param_names_regex (list): List of regular expressions for parameter names to be excluded.
        nums_fisher_examples (List[int]): List of numbers of examples to compute Fisher weights.
        fisher_scaling_coefficients (list, optional): Scaling coefficients to merge Fisher weights. Defaults to None.
        normalize_fisher_weight (bool): Whether to normalize Fisher weights (L2 norm) or not. Defaults to True.
        minimal_fisher_weight (float): The minimal value in Fisher weights, used for tackling the potential numerical issues. Defaults to 1e-6.

    Returns:
        Dict[str, Tensor]: Dictionary of merged parameters.
    """
    # dictionary of list, where key is the parameter name,
    # value is a list of the corresponding parameters of all the models that need to be merged
    models_to_merge_param_dict = defaultdict(list)

    # list of dictionaries with length len(models_to_merge),
    # each dictionary records the fisher weights (matrix or vector) of parameters for each model that needs to be merged
    models_to_merge_fisher_weights_list = []

    assert (
        len(models_to_merge) == len(trainers) == len(nums_fisher_examples)
    ), "sizes of lists are not identical!"

    for model_idx, (model_to_merge, trainer, num_fisher_examples) in enumerate(
        zip(models_to_merge, trainers, nums_fisher_examples)
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

        # list of dictionaries with length (num_fisher_examples // batch_size) or (num_fisher_examples // batch_size) + 1,
        # each dictionary records the fisher weights of parameters for model_to_merge computed by examples in a batch
        batches_fisher_weights_list = []

        num_computed_examples = 0
        train_dataloader = trainer.get_train_dataloader()
        if num_fisher_examples % trainer._train_batch_size != 0:
            print(
                f"warning: the number of examples for computing fisher cannot be fully divided by the batch size for model {model_idx}, "
                "which may lead to a slightly different number of the actually used examples."
            )
        for step, inputs in tqdm(
            enumerate(train_dataloader),
            desc=f"computing fisher weights for model {model_idx}",
        ):
            if num_computed_examples >= num_fisher_examples:
                break
            inputs = trainer._prepare_inputs(inputs)
            outputs = model_to_merge(**inputs)
            # Tensor, shape (batch_size, num_label_classes)
            logits = outputs.logits
            # compute fisher weights for regression task
            if logits.shape[-1] == 1:
                # use the label information to compute loss and obtain gradients
                mse_loss = outputs.loss
                model_to_merge.zero_grad()
                mse_loss.backward()
                # dict, fisher weights of a batch
                batch_fisher_weights = get_param_squared_gradients(
                    model=model_to_merge, param_names_to_merge=param_names_to_merge
                )
            # compute fisher weights for classifxication task
            else:
                # use detach() to detach from the computation graph
                # Tensor, shape (batch_size, num_label_classes)
                labels_probabilities = torch.softmax(logits, dim=-1).detach()
                labels_log_probabilities = torch.log_softmax(logits, dim=-1)
                # sqrt labels_probabilities, since torch.sqrt(labels_probabilities) would be squared in the following squared gradients
                labels_expectations = (
                    torch.sqrt(labels_probabilities) * labels_log_probabilities
                )
                # sum over label classes and batch dimension
                sum_labels_expectations = labels_expectations.sum(dim=-1).sum(dim=0)
                model_to_merge.zero_grad()
                sum_labels_expectations.backward()
                # dict, fisher weights of a batch
                batch_fisher_weights = get_param_squared_gradients(
                    model=model_to_merge, param_names_to_merge=param_names_to_merge
                )

            batches_fisher_weights_list.append(batch_fisher_weights)
            num_computed_examples += trainer._train_batch_size

        model_to_merge_fisher_weights = {}
        for batch_fisher_weights in batches_fisher_weights_list:
            for key in batch_fisher_weights:
                if key not in model_to_merge_fisher_weights:
                    model_to_merge_fisher_weights[key] = batch_fisher_weights[key]
                else:
                    model_to_merge_fisher_weights[key] += batch_fisher_weights[key]

        # mean over batches
        for key in model_to_merge_fisher_weights:
            model_to_merge_fisher_weights[key] /= num_computed_examples
        models_to_merge_fisher_weights_list.append(model_to_merge_fisher_weights)

    # merging with fisher weights
    # if fisher_scaling_coefficients is None, then set the fisher weights of different models to contribute equally
    if fisher_scaling_coefficients is None:
        fisher_scaling_coefficients = torch.ones(len(models_to_merge)) / len(
            models_to_merge
        )
    else:
        assert isinstance(
            fisher_scaling_coefficients, list
        ), "wrong type of fisher_scaling_coefficients, should be list!"
        assert len(fisher_scaling_coefficients) == len(
            models_to_merge
        ), "mismatched length of fisher_scaling_coefficients!"
        fisher_scaling_coefficients = torch.Tensor(fisher_scaling_coefficients)
    # merging with fisher weights
    merged_params = merging_with_fisher_weights(
        models_to_merge_param_dict=models_to_merge_param_dict,
        models_to_merge_fisher_weights_list=models_to_merge_fisher_weights_list,
        fisher_scaling_coefficients=fisher_scaling_coefficients,
        normalize_fisher_weight=normalize_fisher_weight,
        minimal_fisher_weight=minimal_fisher_weight,
    )

    return merged_params


def filter_state_dict(
    state_dict: Dict[str, Tensor],
    param_names: List[str],
) -> Dict[str, Tensor]:
    """
    Filter the state dict with the param names.

    Args:
        state_dict (Dict[str, Tensor]): State dict of a model.
        param_names (List[str]): List of parameter names to be filtered.

    Returns:
        Dict[str, Tensor]: Filtered state dict.
    """
    filtered_state_dict = {}
    for key in param_names:
        filtered_state_dict[key] = state_dict[key]
    return filtered_state_dict


class FisherMergingAlgorithm(BaseAlgorithm, SimpleProfilerMixin):
    """
    Implements the Fisher Merging Algorithm.

    This class extends the BaseModelFusionAlgorithm to handle merging of models using Fisher weights.
    It supports excluding certain parameters, normalizing Fisher weights, and setting a minimal value for Fisher weights.

    Methods:
        run(modelpool: BaseModelPool) -> nn.Module:
            Executes the Fisher merging process on the model pool and returns the merged model.
    """

    _config_mapping = BaseAlgorithm._config_mapping | {
        "exclude_param_names_regex": "exclude_param_names_regex",
        "normalize_fisher_weight": "normalize_fisher_weight",
        "minimal_fisher_weight": "minimal_fisher_weight",
        "num_fisher_examples": "num_fisher_examples",
    }

    def __init__(
        self,
        *,
        exclude_param_names_regex: list,
        normalize_fisher_weight: bool,
        minimal_fisher_weight: float,
        num_fisher_examples: int,
    ):
        super().__init__()
        self.exclude_param_names_regex = exclude_param_names_regex
        self.normalize_fisher_weight = normalize_fisher_weight
        self.minimal_fisher_weight = minimal_fisher_weight
        self.num_fisher_examples = num_fisher_examples

    def run(self, modelpool: BaseModelPool) -> nn.Module:
        """
        Run the Fisher Merging Algorithm.

        This method constructs the wrapped model and performs test-time adaptation if necessary.

        Args:
            modelpool (BaseModelPool): The model pool containing the pretrained and fine-tuned models.

        Returns:
            nn.Module: The merged model after test-time adaptation.
        """
        log.info("Running Fisher Merging Algorithm")
        if isinstance(modelpool, (dict, list, tuple)):
            modelpool = BaseModelPool(modelpool)

        assert len(modelpool) > 0, "model pool is empty"
        assert (
            modelpool.has_pretrained
        ), "no pretrained model (base model) in the model pool"

        self.modelpool = modelpool
        self.on_fisher_merging_start()

        # dictionary of list, where key is the parameter name,
        # value is a list of the corresponding parameters of all the models that need to be merged
        models_to_merge_param_dict = defaultdict(list)

        # list of dictionaries with length len(models_to_merge),
        # each dictionary records the fisher weights (matrix or vector) of parameters for each model that needs to be merged
        models_to_merge_fisher_weights_list = []

        param_names_to_merge = None

        for name, model in modelpool.named_models():
            param_dict = model.state_dict()
            if param_names_to_merge is None:
                param_names_to_merge = get_param_names_to_merge(
                    input_param_names=list(param_dict.keys()),
                    exclude_param_names_regex=self.config.get(
                        "exclude_param_names_regex", []
                    ),
                )

            for param_name in param_names_to_merge:
                models_to_merge_param_dict[param_name].append(param_dict[param_name])

            with (
                self.profile("merging models"),
                self.profile("computing fisher weights"),
            ):
                model_to_merge_fisher_weights = self.get_fisher_weights(
                    model_name=name,
                    model=model,
                    train_dataset=modelpool.load_train_dataset(name),
                    param_names_to_merge=param_names_to_merge,
                )

                models_to_merge_fisher_weights_list.append(
                    model_to_merge_fisher_weights
                )

        with self.profile("merging models"):
            merged_params = merging_with_fisher_weights(
                models_to_merge_param_dict=models_to_merge_param_dict,
                models_to_merge_fisher_weights_list=models_to_merge_fisher_weights_list,
                fisher_scaling_coefficients=torch.ones(len(modelpool)) / len(modelpool),
                normalize_fisher_weight=self.config.get(
                    "normalize_fisher_weight", True
                ),
                minimal_fisher_weight=self.config.get("minimal_fisher_weight", 1e-6),
            )

            merged_model = modelpool.load_model("_pretrained_")
            merged_model.load_state_dict(merged_params, strict=False)

        self.print_profile_summary()
        return merged_model

    def get_fisher_weights(
        self,
        model_name: str,
        model: nn.Module,
        train_dataset,
        param_names_to_merge: List[str],
    ) -> Dict[str, Tensor]:
        """
        Compute the Fisher weights for the given model and training dataset.

        Args:
            model_name (str): The name of the model.
            model (nn.Module): The model module.
            train_dataset: The training dataset.
            param_names_to_merge (List[str]): List of parameter names to merge.

        Returns:
            Dict[str, Tensor]: The computed Fisher weights for each parameter.
        """
        # this function is used to compute fisher weights for a model
        # it should be implemented in the subclass
        raise NotImplementedError

    def on_fisher_merging_start(self):
        """
        Setup the zero-shot classification head before starting the Fisher merging process.
        """
        # this function is used to initialize some variables before running fisher merging
        pass
