"""
A dummy method that does nothing, but returns the `_pretrained` model.
"""

import logging

from omegaconf import DictConfig

from fusion_bench.modelpool import BaseModelPool

from .base_algorithm import BaseModelFusionAlgorithm

log = logging.getLogger(__name__)


class DummyAlgorithm(BaseModelFusionAlgorithm):
    def run(self, modelpool: BaseModelPool):
        """
        This method returns the pretrained model from the model pool.
        If the pretrained model is not available, it returns the first model from the model pool.

        Raises:
            AssertionError: If the model is not found in the model pool.
        """
        model = modelpool.load_pretrained_or_first_model()

        assert model is not None, "Model is not found in the model pool."
        return model
