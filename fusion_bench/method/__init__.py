# flake8: noqa F401
import sys
from typing import TYPE_CHECKING

from fusion_bench.utils import join_lists
from fusion_bench.utils.lazy_imports import LazyImporter

_import_structure = {
    # --------------
    "base_algorithm": ["BaseModelFusionAlgorithm", "BaseAlgorithm"],
    "dummy": ["DummyAlgorithm"],
    # single task learning (fine-tuning)
    "classification": [
        "ImageClassificationFineTuningForCLIP",
        "ContinualImageClassificationFineTuningForCLIP",
        "ImageClassificationFineTuning",
        "ImageClassificationFineTuning_Test",
    ],
    "lm_finetune": ["FullFinetuneSFT", "PeftFinetuneSFT", "BradleyTerryRewardModeling"],
    # analysis
    "analysis": ["TaskVectorCosSimilarity", "TaskVectorViolinPlot"],
    # model ensemble methods
    "ensemble": [
        "SimpleEnsembleAlgorithm",
        "WeightedEnsembleAlgorithm",
        "MaxModelPredictorAlgorithm",
    ],
    # model merging methods
    "linear": [
        "ExPOAlgorithm",
        "ExPOAlgorithmForLlama",
        "SimpleAverageForCausalLM",
        "SimpleAverageForLlama",
        "TaskArithmeticForCausalLM",
        "TaskArithmeticForLlama",
        "LinearInterpolationAlgorithm",
        "TiesMergingForCausalLM",
    ],
    "slerp": ["SlerpMergeAlgorithm", "SlerpForCausalLM"],
    "simple_average": ["SimpleAverageAlgorithm"],
    "weighted_average": ["WeightedAverageAlgorithm", "WeightedAverageForLLama"],
    "task_arithmetic": ["TaskArithmeticAlgorithm"],
    "ties_merging": ["TiesMergingAlgorithm"],
    "dare": ["DareSimpleAverage", "DareTaskArithmetic", "DareTiesMerging"],
    "fisher_merging": [
        "FisherMergingAlgorithm",
        "FisherMergingForCLIPVisionModel",
        "FisherMergingAlgorithmForGPT2",
    ],
    "regmean": ["RegMeanAlgorithmForCLIP", "RegMeanAlgorithmForGPT2"],
    "regmean_plusplus": ["RegMeanAlgorithmPlusPlus", "RegMeanAlgorithmForCLIPPlusPlus"],
    "adamerging": [
        "CLIPTaskWiseAdaMergingAlgorithm",
        "CLIPLayerWiseAdaMergingAlgorithm",
        "GPT2LayerWiseAdaMergingAlgorithm",
        "LayerWiseAdaMergingForLlamaSFT",
        "FlanT5LayerWiseAdaMergingAlgorithm",
        "ResNetLayerWiseAdamerging",
        "ResNetTaskWiseAdamerging",
    ],
    "pwe_moe": [
        "PWEMoELinearScalarizationForCLIP",
        "PWEMoExactParetoOptimalForCLIP",
    ],
    "ada_svd": ["AdaSVDMergingForCLIPVisionModel"],
    "doge_ta": ["DOGE_TA_Algorithm"],
    "task_singular_vector": ["TaskSingularVectorMerging"],
    "isotropic_merging": [
        "ISO_C_Merge",  # alias
        "ISO_CTS_Merge",  # alias
        "IsotropicMergingInCommonAndTaskSubspace",
        "IsotropicMergingInCommonSubspace",
    ],
    "opcm": ["OPCMForCLIP"],
    "dop": ["ContinualDOPForCLIP"],
    "gossip": [
        "CLIPLayerWiseGossipAlgorithm",
        "CLIPTaskWiseGossipAlgorithm",
        "FlanT5LayerWiseGossipAlgorithm",
    ],
    "fw_merging": ["FrankWolfeHardAlgorithm", "FrankWolfeSoftAlgorithm"],
    "tall_mask": ["TallMaskTaskArithmeticAlgorithm"],
    "model_stock": ["ModelStock"],
    "wudi": ["wudi_merging", "WUDIMerging"],
    # plug-and-play model merging methods
    "concrete_subspace": [
        "ConcreteTaskArithmeticAlgorithmForCLIP",
        "ConcreteTaskWiseAdaMergingForCLIP",
        "ConcreteLayerWiseAdaMergingForCLIP",
        "ConcreteSafeLayerWiseAdaMergingForCLIP",
        "ConcreteSafeTaskWiseAdaMergingForCLIP",
        "PostDefenseAWMAlgorithmForCLIP",
        "PostDefenseSAUAlgorithmForCLIP",
    ],
    # model mixing methods
    "depth_upscaling": ["DepthUpscalingAlgorithm", "DepthUpscalingForLlama"],
    "mixture_of_experts": [
        "MixtralUpscalingAlgorithm",
        "MixtralForCausalLMUpscalingAlgorithm",
        "MixtralMoEMergingAlgorithm",
        "MixtralForCausalLMMergingAlgorithm",
    ],
    "dawe": ["DataAdaptiveWeightEnsemblingForCLIP"],
    "we_moe": [
        "CLIPWeightEnsemblingMoEAlgorithm",
        "FlanT5WeightEnsemblingMoEAlgorithm",
    ],
    "rankone_moe": ["CLIPRankOneMoEAlgorithm", "RankOneMoEAlgorithm"],
    "sparse_we_moe": [
        "SparseWeightEnsemblingMoEAlgorithm",
        "SparseCLIPWeightEnsemblingMoEAlgorithm",
    ],
    "model_recombination": ["ModelRecombinationAlgorithm"],
    "smile_upscaling": [
        "SmileUpscalingAlgorithm",
        "SingularProjectionMergingAlgorithm",
    ],
    # task vector compression methods
    "bitdelta": ["BitDeltaAlgorithm"],
    # pruning methods
    "pruning": [
        "MagnitudeDiffPruningAlgorithm",
        "RandomPruningForLlama",
        "MagnitudePruningForLlama",
        "WandaPruningForLlama",
        "SparseGPTPruningForLlama",
    ],
    "sparselo": [
        "IterativeSparseLoForLlama",
        "SparseLoForLlama",
        "PCPSparseLoForLlama",
    ],
    # MoE expert pruning
    "expert_sparsity": [
        "DynamicSkippingPruningForMixtral",
        "LayerWisePruningForMixtral",
        "ProgressivePruningForMixtral",
    ],
}
_available_algorithms = join_lists(list(_import_structure.values()))
_extra_objects = {
    "_available_algorithms": _available_algorithms,
}

if TYPE_CHECKING:
    from .ada_svd import AdaSVDMergingForCLIPVisionModel
    from .adamerging import *
    from .analysis import TaskVectorCosSimilarity, TaskVectorViolinPlot
    from .base_algorithm import BaseAlgorithm, BaseModelFusionAlgorithm
    from .bitdelta import BitDeltaAlgorithm
    from .classification import (
        ContinualImageClassificationFineTuningForCLIP,
        ImageClassificationFineTuning,
        ImageClassificationFineTuning_Test,
        ImageClassificationFineTuningForCLIP,
    )
    from .concrete_subspace import (
        ConcreteLayerWiseAdaMergingForCLIP,
        ConcreteSafeLayerWiseAdaMergingForCLIP,
        ConcreteSafeTaskWiseAdaMergingForCLIP,
        ConcreteTaskArithmeticAlgorithmForCLIP,
        ConcreteTaskWiseAdaMergingForCLIP,
        PostDefenseAWMAlgorithmForCLIP,
        PostDefenseSAUAlgorithmForCLIP,
    )
    from .dare import DareSimpleAverage, DareTaskArithmetic, DareTiesMerging
    from .dawe import DataAdaptiveWeightEnsemblingForCLIP
    from .depth_upscaling import DepthUpscalingAlgorithm, DepthUpscalingForLlama
    from .doge_ta import DOGE_TA_Algorithm
    from .dop import ContinualDOPForCLIP
    from .dummy import DummyAlgorithm
    from .ensemble import (
        MaxModelPredictorAlgorithm,
        SimpleEnsembleAlgorithm,
        WeightedEnsembleAlgorithm,
    )
    from .expert_sparsity import (
        DynamicSkippingPruningForMixtral,
        LayerWisePruningForMixtral,
        ProgressivePruningForMixtral,
    )
    from .fisher_merging import (
        FisherMergingAlgorithm,
        FisherMergingAlgorithmForGPT2,
        FisherMergingForCLIPVisionModel,
    )
    from .fw_merging import FrankWolfeHardAlgorithm, FrankWolfeSoftAlgorithm
    from .gossip import (
        CLIPLayerWiseGossipAlgorithm,
        CLIPTaskWiseGossipAlgorithm,
        FlanT5LayerWiseGossipAlgorithm,
    )
    from .isotropic_merging import (
        ISO_C_Merge,
        ISO_CTS_Merge,
        IsotropicMergingInCommonAndTaskSubspace,
        IsotropicMergingInCommonSubspace,
    )
    from .linear import (
        ExPOAlgorithm,
        ExPOAlgorithmForLlama,
        LinearInterpolationAlgorithm,
        SimpleAverageForCausalLM,
        SimpleAverageForLlama,
        TaskArithmeticForCausalLM,
        TaskArithmeticForLlama,
        TiesMergingForCausalLM,
    )
    from .lm_finetune import *
    from .mixture_of_experts import (
        MixtralForCausalLMMergingAlgorithm,
        MixtralForCausalLMUpscalingAlgorithm,
        MixtralMoEMergingAlgorithm,
        MixtralUpscalingAlgorithm,
    )
    from .model_recombination import ModelRecombinationAlgorithm
    from .model_stock import ModelStock
    from .opcm import OPCMForCLIP
    from .pruning import (
        MagnitudeDiffPruningAlgorithm,
        MagnitudePruningForLlama,
        RandomPruningForLlama,
        SparseGPTPruningForLlama,
        WandaPruningForLlama,
    )
    from .pwe_moe import (
        PWEMoELinearScalarizationForCLIP,
        PWEMoExactParetoOptimalForCLIP,
    )
    from .rankone_moe import CLIPRankOneMoEAlgorithm, RankOneMoEAlgorithm
    from .regmean import RegMeanAlgorithmForCLIP, RegMeanAlgorithmForGPT2
    from .regmean_plusplus import (
        RegMeanAlgorithmForCLIPPlusPlus,
        RegMeanAlgorithmPlusPlus,
    )
    from .simple_average import SimpleAverageAlgorithm
    from .slerp import SlerpForCausalLM, SlerpMergeAlgorithm
    from .smile_upscaling import (
        SingularProjectionMergingAlgorithm,
        SmileUpscalingAlgorithm,
    )
    from .sparse_we_moe import (
        SparseCLIPWeightEnsemblingMoEAlgorithm,
        SparseWeightEnsemblingMoEAlgorithm,
    )
    from .sparselo import (
        IterativeSparseLoForLlama,
        PCPSparseLoForLlama,
        SparseLoForLlama,
    )
    from .tall_mask import TallMaskTaskArithmeticAlgorithm
    from .task_arithmetic import TaskArithmeticAlgorithm
    from .task_singular_vector import TaskSingularVectorMerging
    from .ties_merging import TiesMergingAlgorithm
    from .we_moe import (
        CLIPWeightEnsemblingMoEAlgorithm,
        FlanT5WeightEnsemblingMoEAlgorithm,
    )
    from .weighted_average import WeightedAverageAlgorithm, WeightedAverageForLLama
    from .wudi import WUDIMerging, wudi_merging

else:
    sys.modules[__name__] = LazyImporter(
        __name__,
        globals()["__file__"],
        _import_structure,
        extra_objects=_extra_objects,
    )
