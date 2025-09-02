import os
from copy import deepcopy
from typing import TYPE_CHECKING, Optional

from omegaconf import flag_override
from typing_extensions import override

from fusion_bench import timeit_context
from fusion_bench.method.base_algorithm import BaseAlgorithm
from fusion_bench.method.simple_average import SimpleAverageAlgorithm
from fusion_bench.mixins import auto_register_config
from fusion_bench.modelpool import CausalLMBackbonePool, CausalLMPool
from fusion_bench.models.hf_utils import create_default_model_card
from fusion_bench.utils import instantiate
from fusion_bench.utils.pylogger import get_rankzero_logger

log = get_rankzero_logger(__name__)


@auto_register_config
class SimpleAverageForCausalLM(BaseAlgorithm):
    R"""
    A simple averaging algorithm for LLama models. If `merge_backbone` is set to `True`, the backbone of the model will be averaged and the rest of the model will be loaded from the pre-trained model.

    Examples:
        The following example demonstrates how to use the `SimpleAverageForCausalLM` algorithm to merge Mistral models.

        ```bash
        fusion_bench \
            method=linear/simple_average_for_causallm \
            method.model_save_path=outputs/simle_mixtral_exp_v4/simple_average \
            modelpool=CausalLMPool/simle_mixtral_exp_v4.yaml
        ```
    """

    def __init__(
        self,
        merge_backbone: bool = False,
        model_save_path: Optional[str] = None,
        show_pbar: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

    @override
    def run(self, modelpool: CausalLMPool):
        if self.model_save_path:
            tokenizer = modelpool.load_tokenizer()

        if self.merge_backbone:
            assert modelpool.has_pretrained
            log.info(
                "Merging backbone of the model pool, use CausalLMBackbonePool instead of CausalLMPool."
            )
            modelpool_config = deepcopy(modelpool.config)
            with flag_override(modelpool_config, "allow_objects", True):
                modelpool_config._target_ = (
                    "fusion_bench.modelpool.causal_lm.CausalLMBackbonePool"
                )
            backbone_modelpool = instantiate(modelpool_config)
            model = modelpool.load_model("_pretrained_")
            backbone_model = SimpleAverageAlgorithm(show_pbar=self.show_pbar).run(
                backbone_modelpool
            )
            model.model.layers = backbone_model
        else:
            model = SimpleAverageAlgorithm(show_pbar=self.show_pbar).run(
                modelpool=modelpool
            )

        if self.model_save_path is not None:
            with timeit_context(f"Saving the model to {self.model_save_path}"):
                tokenizer.save_pretrained(self.model_save_path)
                model.save_pretrained(self.model_save_path)
                model_card_str = create_default_model_card(
                    models=[modelpool.get_model_path(m) for m in modelpool.model_names],
                    description="Merged model using simple averaging.",
                    algorithm_config=self.config,
                    modelpool_config=modelpool.config,
                )
                with open(os.path.join(self.model_save_path, "README.md"), "w") as f:
                    f.write(model_card_str)
        return model


SimpleAverageForLlama = SimpleAverageForCausalLM
"""Alias for SimpleAverageForCausalLM"""
