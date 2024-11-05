import logging
import random
from typing import List, Mapping, Union  # noqa: F401

import torch
from torch import nn

from fusion_bench.method import BaseAlgorithm
from fusion_bench.modelpool import BaseModelPool

log = logging.getLogger(__name__)


def recombine_modellist(models: List[nn.ModuleList]):
    num_models = len(models)
    num_layers = len(models[0])

    new_models = [[] for _ in range(num_models)]
    for layer_idx in range(num_layers):
        shuffled_layers = [m[layer_idx] for m in models]
        random.shuffle(shuffled_layers)
        for model_idx in range(num_models):
            new_models[model_idx].append(shuffled_layers[model_idx])
    new_models = [nn.ModuleList(m) for m in new_models]
    return new_models


def recombine_modeldict(models: List[nn.ModuleDict]):
    num_models = len(models)

    new_models = [{} for _ in range(num_models)]
    for layer_name in models[0].keys():
        shuffled_layers = [m[layer_name] for m in models]
        random.shuffle(shuffled_layers)
        for model_idx in range(num_models):
            new_models[model_idx][layer_name] = shuffled_layers[model_idx]
    new_models = [nn.ModuleDict(m) for m in new_models]
    return new_models


def recombine_state_dict(models: List[nn.Module]):
    num_models = len(models)
    state_dicts = [model.state_dict() for model in models]
    new_state_dict = [{} for _ in range(num_models)]
    for key in state_dicts[0].keys():
        shuffled_layers = [state_dict[key] for state_dict in state_dicts]
        random.shuffle(shuffled_layers)
        for model_idx in range(num_models):
            new_state_dict[model_idx][key] = shuffled_layers[model_idx]
    for model_idx in range(num_models):
        models[model_idx].load_state_dict(new_state_dict[model_idx])
    return models


class ModelRecombinationAlgorithm(BaseAlgorithm):
    """
    Model recombination recombinates the layers of the given models, to create a new set of models.
    """

    _config_mapping = BaseAlgorithm._config_mapping | {
        "return_modelpool": "return_modelpool",
    }

    def __init__(self, return_modelpool: bool, **kwargs):
        self.return_modelpool = return_modelpool
        super().__init__(**kwargs)

    @torch.no_grad()
    def run(
        self,
        modelpool: BaseModelPool,
        return_modelpool: bool = True,
    ) -> Union[nn.Module, BaseModelPool]:
        """
        Executes the model recombination algorithm on a given model pool.

        This method loads models from the model pool, determines their type, and applies the appropriate recombination method.
        It then creates a new model pool with the recombined models. Depending on the `return_modelpool` flag, it either returns
        the entire new model pool or just the first model from it.

        - If the models in the model pool are of type `nn.ModuleList`, the recombination method `recombine_modellist` is used. Where each module in the list is shuffled across the models.
        - If the models are of type `nn.ModuleDict`, the recombination method `recombine_modeldict` is used. Where each module in the dictionary is shuffled across the models.
        - If the models are of type `nn.Module`, the recombination method `recombine_state_dict` is used. Where the state dictionaries of the models are shuffled across the models.

        Args:
            modelpool (BaseModelPool): The pool of models to recombine.
            return_modelpool (bool, optional): Flag indicating whether to return the entire model pool or just the first model. Defaults to True. If this algorithm is initialized with config, the value of `return_modelpool` in the config will be used and this argument passed to the method will be ignored.

        Returns:
            Union[nn.Module, BaseModelPool]: The recombined model pool or the first model from the recombined pool, depending on the `return_modelpool` flag.

        Raises:
            ValueError: If the models in the model pool are of an unsupported type.
        """
        # If the config has a return_modelpool flag, use that, otherwise use the argument
        if self.config.get("return_modelpool", None) is not None:
            return_modelpool = self.config.return_modelpool
        # check the modelpool type
        if not isinstance(modelpool, BaseModelPool):
            modelpool = BaseModelPool(modelpool)

        log.info(f"Running model recombination algorithm with {len(modelpool)} models")

        # TODO: optimize the `recombine_*` functions, if `return_modelpool` is False, we don't need to create the new modelpool, just the first model
        models = [modelpool.load_model(m) for m in modelpool.model_names]
        if isinstance(models[0], nn.ModuleList):
            new_models = recombine_modellist(models)
        elif isinstance(models[0], nn.ModuleDict):
            new_models = recombine_modeldict(models)
        elif isinstance(models[0], nn.Module):
            new_models = recombine_state_dict(models)
        else:
            raise ValueError(f"Unsupported model type {type(models[0])}")

        new_modelpool = BaseModelPool(
            {n: m for n, m in zip(modelpool.model_names, new_models)}
        )
        if return_modelpool:
            return new_modelpool
        else:
            return new_modelpool.load_model(new_modelpool.model_names[0])
