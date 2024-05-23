from omegaconf import DictConfig

from .adamerging.clip_layer_wise_adamerging import CLIPLayerWiseAdaMergingAlgorithm
from .adamerging.clip_task_wise_adamerging import CLIPTaskWiseAdaMergingAlgorithm
from .base_algorithm import ModelFusionAlgorithm
from .depth_upscaling import DepthUpscalingAlgorithm
from .dummy import DummyAlgorithm
from .ensemble import (
    EnsembleAlgorithm,
    MaxModelPredictorAlgorithm,
    WeightedEnsembleAlgorithm,
)
from .mixture_of_experts.mixtral_merging import (
    MixtralForCausalLMMergingAlgorithm,
    MixtralMoEMergingAlgorithm,
)
from .mixture_of_experts.mixtral_upcycling import (
    MixtralForCausalLMUpscalingAlgorithm,
    MixtralUpscalingAlgorithm,
)
from .model_recombination import ModelRecombinationAlgorithm
from .simple_average import SimpleAverageAlgorithm
from .task_arithmetic import TaskArithmeticAlgorithm
from .ties_merging.ties_merging import TiesMergingAlgorithm
from .we_moe.clip_we_moe import CLIPWeightEnsemblingMoEAlgorithm
from .weighted_average import WeightedAverageAlgorithm


def load_algorithm_from_config(method_config: DictConfig):
    """
    Loads an algorithm based on the provided configuration.

    The function checks the 'name' attribute of the configuration and returns an instance of the corresponding algorithm.
    If the 'name' attribute is not found or does not match any known algorithm names, a ValueError is raised.

    Args:
        method_config (DictConfig): The configuration for the algorithm. Must contain a 'name' attribute that specifies the type of the algorithm.

    Returns:
        An instance of the specified algorithm.

    Raises:
        ValueError: If 'name' attribute is not found in the configuration or does not match any known algorithm names.
    """
    if method_config.name == "dummy":
        return DummyAlgorithm(method_config)
    # model ensemble methods
    elif method_config.name == "simple_ensemble":
        return EnsembleAlgorithm(method_config)
    elif method_config.name == "weighted_ensemble":
        return WeightedEnsembleAlgorithm(method_config)
    elif method_config.name == "max_model_predictor":
        return MaxModelPredictorAlgorithm(method_config)
    # model merging methods
    elif method_config.name == "simple_average":
        return SimpleAverageAlgorithm(method_config)
    elif method_config.name == "weighted_average":
        return WeightedAverageAlgorithm(method_config)
    elif method_config.name == "task_arithmetic":
        return TaskArithmeticAlgorithm(method_config)
    elif method_config.name == "ties_merging":
        return TiesMergingAlgorithm(method_config)
    elif method_config.name == "clip_task_wise_adamerging":
        return CLIPTaskWiseAdaMergingAlgorithm(method_config)
    elif method_config.name == "clip_layer_wise_adamerging":
        return CLIPLayerWiseAdaMergingAlgorithm(method_config)
    # model mixing methods
    elif method_config.name == "depth_upscaling":
        return DepthUpscalingAlgorithm(method_config)
    elif method_config.name == "mixtral_moe_upscaling":
        return MixtralUpscalingAlgorithm(method_config)
    elif method_config.name == "mixtral_for_causal_lm_moe_upscaling":
        return MixtralForCausalLMUpscalingAlgorithm(method_config)
    elif method_config.name == "mixtral_moe_merging":
        return MixtralMoEMergingAlgorithm(method_config)
    elif method_config.name == "mixtral_for_causal_lm_merging":
        return MixtralForCausalLMMergingAlgorithm(method_config)
    elif method_config.name == "clip_weight_ensembling_moe":
        return CLIPWeightEnsemblingMoEAlgorithm(method_config)
    elif method_config.name == "model_recombination":
        return ModelRecombinationAlgorithm(method_config)
    else:
        raise ValueError(f"Unknown algorithm: {method_config.name}")
