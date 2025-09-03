import os

import lightning as L
from hydra import compose, initialize

from fusion_bench import instantiate
from fusion_bench.method import SimpleEnsembleAlgorithm
from fusion_bench.modelpool import CLIPVisionModelPool
from fusion_bench.scripts.cli import _get_default_config_path
from fusion_bench.taskpool import CLIPVisionModelTaskPool
from fusion_bench.utils.rich_utils import setup_colorlogging

setup_colorlogging()

fabric = L.Fabric(accelerator="auto", devices=1)

# Load configuration using Hydra
with initialize(
    version_base=None,
    config_path=os.path.relpath(
        _get_default_config_path(), start=os.path.dirname(__file__)
    ),
):
    cfg = compose(
        config_name="fabric_model_fusion",
        overrides=[
            "modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8",
            "taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8",
        ],
    )
    modelpool: CLIPVisionModelPool = instantiate(cfg.modelpool)
    taskpool: CLIPVisionModelTaskPool = instantiate(cfg.taskpool, move_to_device=False)
    taskpool.fabric = fabric

# Hard-coded device map and algorithm instantiation for 8 models
device_map = {
    0: "cuda:0",
    1: "cuda:0",
    2: "cuda:0",
    3: "cuda:0",
    4: "cuda:1",
    5: "cuda:1",
    6: "cuda:1",
    7: "cuda:1",
}
algorithm = SimpleEnsembleAlgorithm(device_map=device_map)
ensemble = algorithm.run(modelpool)

report = taskpool.evaluate(ensemble)
print(report)
