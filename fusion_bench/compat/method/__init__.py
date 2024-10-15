from omegaconf import DictConfig

from .base_algorithm import ModelFusionAlgorithm


class AlgorithmFactory:
    _aglorithms = {
        "dummy": ".dummy.DummyAlgorithm",
        # single task learning (fine-tuning)
        "clip_finetune": ".classification.clip_finetune.ImageClassificationFineTuningForCLIP",
        # analysis
        "TaskVectorCosSimilarity": ".analysis.task_vector_cos_similarity.TaskVectorCosSimilarity",
        # model ensemble methods
        "simple_ensemble": ".ensemble.EnsembleAlgorithm",
        "weighted_ensemble": ".ensemble.WeightedEnsembleAlgorithm",
        "max_model_predictor": ".ensemble.MaxModelPredictorAlgorithm",
        # model merging methods
        "simple_average": ".simple_average.SimpleAverageAlgorithm",
        "weighted_average": ".weighted_average.weighted_average.WeightedAverageAlgorithm",
        "weighted_average_for_llama": ".weighted_average.llama.WeightedAverageForLLama",
        "clip_fisher_merging": ".fisher_merging.clip_fisher_merging.FisherMergingAlgorithmForCLIP",
        "gpt2_fisher_merging": ".fisher_merging.gpt2_fisher_merging.FisherMergingAlgorithmForGPT2",
        "clip_regmean": ".regmean.clip_regmean.RegMeanAlgorithmForCLIP",
        "gpt2_regmean": ".regmean.gpt2_regmean.RegMeanAlgorithmForGPT2",
        "task_arithmetic": ".task_arithmetic.TaskArithmeticAlgorithm",
        "ties_merging": ".ties_merging.ties_merging.TiesMergingAlgorithm",
        "clip_task_wise_adamerging": ".adamerging.clip_task_wise_adamerging.CLIPTaskWiseAdaMergingAlgorithm",
        "clip_layer_wise_adamerging": ".adamerging.clip_layer_wise_adamerging.CLIPLayerWiseAdaMergingAlgorithm",
        "singular_projection_merging": ".smile_upscaling.singular_projection_merging.SingularProjectionMergingAlgorithm",
        "pwe_moe_ls_for_clip": ".pwe_moe.clip_pwe_moe.PWEMoELinearScalarizationForCLIP",
        "pwe_moe_epo_for_clip": ".pwe_moe.clip_pwe_moe.PWEMoExactParetoOptimalForCLIP",
        # plug-and-play model merging methods
        "clip_concrete_task_arithmetic": ".concrete_subspace.clip_concrete_task_arithmetic.ConcreteTaskArithmeticAlgorithmForCLIP",
        "clip_concrete_task_wise_adamerging": ".concrete_subspace.clip_concrete_adamerging.ConcreteTaskWiseAdaMergingForCLIP",
        "clip_concrete_layer_wise_adamerging": ".concrete_subspace.clip_concrete_adamerging.ConcreteLayerWiseAdaMergingForCLIP",
        # model mixing methods
        "depth_upscaling": ".depth_upscaling.DepthUpscalingAlgorithm",
        "mixtral_moe_upscaling": ".mixture_of_experts.mixtral_upcycling.MixtralUpscalingAlgorithm",
        "mixtral_for_causal_lm_moe_upscaling": ".mixture_of_experts.mixtral_upcycling.MixtralForCausalLMUpscalingAlgorithm",
        "mixtral_moe_merging": ".mixture_of_experts.mixtral_merging.MixtralMoEMergingAlgorithm",
        "mixtral_for_causal_lm_merging": ".mixture_of_experts.mixtral_merging.MixtralForCausalLMMergingAlgorithm",
        "clip_weight_ensembling_moe": ".we_moe.clip_we_moe.CLIPWeightEnsemblingMoEAlgorithm",
        "model_recombination": ".model_recombination.ModelRecombinationAlgorithm",
        "smile_upscaling": ".smile_upscaling.smile_upscaling.SmileUpscalingAlgorithm",
        "smile_mistral_upscaling": ".smile_upscaling.smile_mistral_upscaling.SmileMistralUpscalingAlgorithm",
        # pruning methods
        "magnitude_diff_pruning": ".pruning.MagnitudeDiffPruningAlgorithm",
        "magnitude_pruning_for_llama": ".pruning.llama_magnitude_prune.MagnitudePruningForLlama",
        "wanda_pruning_for_llama": ".pruning.llama_wanda_prune.WandaPruningForLlama",
    }

    @staticmethod
    def create_algorithm(method_config: DictConfig) -> ModelFusionAlgorithm:
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
                algorithm_cls = f"fusion_bench.compat.method.{algorithm_cls[1:]}"
            algorithm_cls = import_object(algorithm_cls)
        return algorithm_cls(method_config)

    @staticmethod
    def register_algorithm(name: str, algorithm_cls):
        AlgorithmFactory._aglorithms[name] = algorithm_cls

    @classmethod
    def available_algorithms(cls):
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
