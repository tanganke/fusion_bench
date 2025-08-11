from fusion_bench import BaseModelPool
from fusion_bench.utils import instantiate
from fusion_bench.utils.lazy_state_dict import LazyStateDict


class LazyStateDictPool(BaseModelPool):
    def load_model(self, model_name_or_config: str, *args, **kwargs) -> LazyStateDict:
        if model_name_or_config in self._models:
            checkpoint_config = self._models[model_name_or_config]
        else:
            checkpoint_config = model_name_or_config
        if isinstance(checkpoint_config, str):
            return LazyStateDict(checkpoint_config, *args, **kwargs)
        else:
            return instantiate(checkpoint_config, *args, **kwargs)
