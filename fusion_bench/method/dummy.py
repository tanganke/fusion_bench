"""
A dummy method that does nothing, but returns the `_pretrained` model.
"""

import logging

from omegaconf import DictConfig

from ..modelpool import ModelPool, to_modelpool
from .base_algorithm import ModelFusionAlgorithm

log = logging.getLogger(__name__)


class DummyAlgorithm(ModelFusionAlgorithm):
    def __init__(self, algorithm_config: DictConfig):
        super().__init__(algorithm_config)

    def run(self, modelpool: ModelPool):
        """
        This method returns the pretrained model from the model pool.
        If the pretrained model is not available, it returns the first model from the model pool.

        Raises:
            AssertionError: If the model is not found in the model pool.
        """
        modelpool = to_modelpool(modelpool)
        if "_pretrained_" in modelpool._model_names:
            model = modelpool.load_model("_pretrained_")
        else:
            log.warning(
                "No pretrained model found in the model pool. Returning the first model."
            )
            model = modelpool.load_model(modelpool.model_names[0])

        assert model is not None, "Model is not found in the model pool."
        return model
