from typing import Any

from fusion_bench import BaseModelPool, auto_register_config
from fusion_bench.mixins import LightningFabricMixin
from fusion_bench.utils.instantiate_utils import instantiate

from .svc import SingularValueCalibration


@auto_register_config
class SingularValueCalibrationWithBaseMethod(
    LightningFabricMixin, SingularValueCalibration
):
    """
    Generic SVC wrapper that first runs a base merging method and then calibrates
    the resulting merged model.
    """

    def __init__(self, base_method: Any, alpha: float, accelerator=None, **kwargs):
        super().__init__(alpha=alpha, accelerator=accelerator, **kwargs)
        self.base_method = base_method

    def _instantiate_base_method(self):
        if isinstance(self.base_method, dict) or hasattr(self.base_method, "keys"):
            base_method = instantiate(self.base_method, _recursive_=False)
        else:
            base_method = self.base_method

        if hasattr(base_method, "_fabric_instance"):
            base_method._fabric_instance = self.fabric
        if hasattr(base_method, "_fabric"):
            base_method._fabric = self.fabric
        if hasattr(self, "_program") and hasattr(base_method, "_program"):
            base_method._program = self._program

        return base_method

    def run(self, modelpool: BaseModelPool):
        if not isinstance(modelpool, BaseModelPool):
            modelpool = BaseModelPool(modelpool)

        base_method = self._instantiate_base_method()
        merged_model = base_method.run(modelpool)
        modelpool.add_model("_merged_", merged_model)

        return super().run(modelpool)
