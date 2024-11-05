"""
A dummy method that does nothing, but returns the `_pretrained_` model.
"""

import logging

from torch import nn

from fusion_bench.method import BaseAlgorithm
from fusion_bench.modelpool import BaseModelPool

log = logging.getLogger(__name__)


class DummyAlgorithm(BaseAlgorithm):
    def run(self, modelpool: BaseModelPool):
        """
        This method returns the pretrained model from the model pool.
        If the pretrained model is not available, it returns the first model from the model pool.

        Args:
            modelpool (BaseModelPool): The pool of models to fuse.

        Raises:
            AssertionError: If the model is not found in the model pool.
        """
        if isinstance(modelpool, nn.Module):
            return modelpool
        elif not isinstance(modelpool, BaseModelPool):
            modelpool = BaseModelPool(modelpool)

        model = modelpool.load_pretrained_or_first_model()

        assert model is not None, "Model is not found in the model pool."
        return model
