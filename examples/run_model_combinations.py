"""
This script is an example of how to run multiple experiments with different combinations of models.
"""

import itertools
import multiprocessing

from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf

from fusion_bench.scripts.cli import main
from fusion_bench.utils.rich_utils import setup_colorlogging

setup_colorlogging()

MODEL_NAMES = {
    "model_1": "tanganke/clip-vit-base-patch32_sun397",
    "model_2": "tanganke/clip-vit-base-patch32_stanford-cars",
}
CONFIG_PATH = "config"
CONFIG_NAME = "fabric_model_fusion"
CONFIG_OVERRIDES = [
    "method=simple_average",
    "modelpool=CLIPVisionModelPool/_template",
    "dry_run=false",  # print the config without running the program, remove this after testing
]


if __name__ == "__main__":
    with initialize(version_base=None, config_path="config", job_name="test_app"):
        cfg = compose(
            config_name=CONFIG_NAME,
            overrides=CONFIG_OVERRIDES,
            return_hydra_config=False,
        )

        for num_models in range(1, len(MODEL_NAMES) + 1):
            for selected_models in itertools.combinations(MODEL_NAMES, num_models):
                models = {
                    "_pretrained_": "openai/clip-vit-base-patch32",
                }
                for model_name in selected_models:
                    models[model_name] = MODEL_NAMES[model_name]
                cfg.modelpool.models = DictConfig(models)

                print(cfg)

                mp_ctx = multiprocessing.get_context("spawn")
                p = mp_ctx.Process(
                    target=main,
                    args=(cfg,),
                )
                p.start()
                p.join()
