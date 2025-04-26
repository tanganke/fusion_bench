import warnings

from omegaconf import DictConfig

from .base_algorithm import ModelFusionAlgorithm


class AlgorithmFactory:
    """
    Factory class to create and manage different model fusion algorithms.

    This class provides methods to create algorithms based on a given configuration,
    register new algorithms, and list available algorithms.
    """

    _aglorithms = {
        # single task learning (fine-tuning)
        "clip_finetune": ".classification.clip_finetune.ImageClassificationFineTuningForCLIP",
        # analysis
        # model merging methods
        "clip_task_wise_adamerging": ".adamerging.clip_task_wise_adamerging.CLIPTaskWiseAdaMergingAlgorithm",
        "clip_layer_wise_adamerging": ".adamerging.clip_layer_wise_adamerging.CLIPLayerWiseAdaMergingAlgorithm",
        "clip_layer_wise_adamerging_doge_ta": ".doge_ta.clip_layer_wise_adamerging.CLIPLayerWiseAdaMergingAlgorithm",
        "singular_projection_merging": "fusion_bench.method.smile_upscaling.singular_projection_merging.SingularProjectionMergingAlgorithm",
        "clip_layer_wise_adamerging_surgery": ".surgery.clip_layer_wise_adamerging_surgery.CLIPLayerWiseAdaMergingSurgeryAlgorithm",
        "clip_task_wise_gossip": ".gossip.clip_task_wise_gossip.CLIPTaskWiseGossipAlgorithm",
        "clip_layer_wise_gossip": ".gossip.clip_layer_wise_gossip.CLIPLayerWiseGossipAlgorithm",
        # plug-and-play model merging methods
        "clip_concrete_task_arithmetic": ".concrete_subspace.clip_concrete_task_arithmetic.ConcreteTaskArithmeticAlgorithmForCLIP",
        "clip_concrete_task_wise_adamerging": ".concrete_subspace.clip_concrete_adamerging.ConcreteTaskWiseAdaMergingForCLIP",
        "clip_concrete_layer_wise_adamerging": ".concrete_subspace.clip_concrete_adamerging.ConcreteLayerWiseAdaMergingForCLIP",
        "clip_post_defense_AWM": ".concrete_subspace.clip_post_defense.PostDefenseAWMAlgorithmForCLIP",
        "clip_post_defense_SAU": ".concrete_subspace.clip_post_defense.PostDefenseSAUAlgorithmForCLIP",
        "clip_safe_concrete_layer_wise_adamerging": ".concrete_subspace.clip_safe_concrete_adamerging.ConcreteSafeLayerWiseAdaMergingForCLIP",
        "clip_safe_concrete_task_wise_adamerging": ".concrete_subspace.clip_safe_concrete_adamerging.ConcreteSafeTaskWiseAdaMergingForCLIP",
        # model mixing methods
        "clip_weight_ensembling_moe": ".we_moe.clip_we_moe.CLIPWeightEnsemblingMoEAlgorithm",
        "sparse_clip_weight_ensembling_moe": "fusion_bench.method.SparseCLIPWeightEnsemblingMoEAlgorithm",
        "smile_mistral_upscaling": ".smile_upscaling.smile_mistral_upscaling.SmileMistralUpscalingAlgorithm",
        "rankone_moe": ".rankone_moe.clip_rankone_moe.CLIPRankOneMoEAlgorithm",
    }

    @staticmethod
    def create_algorithm(method_config: DictConfig) -> ModelFusionAlgorithm:
        """
        Create an instance of a model fusion algorithm based on the provided configuration.

        Args:
            method_config (DictConfig): The configuration for the algorithm. Must contain a 'name' attribute that specifies the type of the algorithm.

        Returns:
            ModelFusionAlgorithm: An instance of the specified algorithm.

        Raises:
            ValueError: If 'name' attribute is not found in the configuration or does not match any known algorithm names.
        """
        warnings.warn(
            "AlgorithmFactory.create_algorithm() is deprecated and will be removed in future versions. "
            "Please implement new model fusion algorithm using `fusion_bench.method.BaseModelFusionAlgorithm` instead.",
            DeprecationWarning,
        )

        from fusion_bench.utils import import_object

        algorithm_name = method_config.name
        if algorithm_name not in AlgorithmFactory._aglorithms:
            raise ValueError(
                f"Unknown algorithm: {algorithm_name}, available algorithms: {AlgorithmFactory._aglorithms.keys()}."
                "You can register a new algorithm using `AlgorithmFactory.register_algorithm()` method."
            )
        algorithm_cls = AlgorithmFactory._aglorithms[algorithm_name]
        if isinstance(algorithm_cls, str):
            if algorithm_cls.startswith("."):
                algorithm_cls = f"fusion_bench.method.{algorithm_cls[1:]}"
            algorithm_cls = import_object(algorithm_cls)
        return algorithm_cls(method_config)

    @staticmethod
    def register_algorithm(name: str, algorithm_cls):
        """
        Register a new algorithm with the factory.

        Args:
            name (str): The name of the algorithm.
            algorithm_cls: The class of the algorithm to register.
        """
        AlgorithmFactory._aglorithms[name] = algorithm_cls

    @classmethod
    def available_algorithms(cls):
        """
        Get a list of available algorithms.

        Returns:
            list: A list of available algorithm names.
        """
        return list(cls._aglorithms.keys())


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
    return AlgorithmFactory.create_algorithm(method_config)
