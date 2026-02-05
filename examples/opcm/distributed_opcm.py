from transformers import CLIPVisionModel

from fusion_bench import BaseModelPool
from fusion_bench.constants.paths import DEFAULT_CONFIG_PATH
from fusion_bench.method.opcm.opcm_general import OPCM
from fusion_bench.utils import timeit_context

config_file = DEFAULT_CONFIG_PATH / "method/opcm/opcm_general.yaml"


with timeit_context("loading models"):
    models = {
        "_pretrained_": CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32"),
        "sun397": CLIPVisionModel.from_pretrained(
            "tanganke/clip-vit-base-patch32_sun397"
        ),
        "stanford-cars": CLIPVisionModel.from_pretrained(
            "tanganke/clip-vit-base-patch32_stanford-cars"
        ),
    }

algo: OPCM = OPCM.from_yaml(config_file)
algo.num_ray_actors = 2 # set the number of ray actors to use for parallel merging
algo.run(BaseModelPool(models))
