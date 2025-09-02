import logging
import os
from typing import Dict, List, Mapping, Optional, TypeVar, Union  # noqa: F401

from typing_extensions import override

from fusion_bench import auto_register_config, timeit_context
from fusion_bench.method import TiesMergingAlgorithm
from fusion_bench.mixins.simple_profiler import SimpleProfilerMixin
from fusion_bench.modelpool import CausalLMBackbonePool, CausalLMPool
from fusion_bench.models.hf_utils import create_default_model_card

log = logging.getLogger(__name__)


@auto_register_config
class TiesMergingForCausalLM(
    TiesMergingAlgorithm,
):
    R"""
    TIES merging algorithm for CausalLM models.

    This class extends the TiesMergingAlgorithm to work specifically with CausalLM models,
    providing model saving capabilities and backbone merging support.
    """

    _config_mapping = TiesMergingAlgorithm._config_mapping | {
        "merge_backbone": "merge_backbone",
    }

    def __init__(
        self,
        scaling_factor: float,
        threshold: float,
        remove_keys: List[str] = None,
        merge_func: str = "sum",
        merge_backbone: bool = False,
        model_save_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            scaling_factor=scaling_factor,
            threshold=threshold,
            remove_keys=remove_keys,
            merge_func=merge_func,
            **kwargs,
        )

    @override
    def run(self, modelpool: CausalLMPool):
        if self.merge_backbone:
            assert modelpool.has_pretrained
            backbone_modelpool = CausalLMBackbonePool(**modelpool.config)
            model = modelpool.load_model("_pretrained_")
            backbone_model = super().run(backbone_modelpool)
            model.model.layers = backbone_model
        else:
            model = super().run(modelpool)

        if self.model_save_path is not None:
            with timeit_context(f"Saving the model to {self.model_save_path}"):
                description = f"Merged model using TIES merging with scaling factor {self.scaling_factor} and threshold {self.threshold}."
                modelpool.save_model(
                    model=model,
                    path=self.model_save_path,
                    save_tokenizer=True,
                    algorithm_config=self.config,
                    description=description,
                )
        return model
