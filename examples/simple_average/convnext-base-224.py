"""
Example of merging ConvNeXt models using simple averaging.
"""

import lightning as L

from fusion_bench.method import SimpleAverageAlgorithm
from fusion_bench.modelpool import ConvNextForImageClassificationPool, BaseModelPool
from fusion_bench.models.wrappers.switch import SwitchModule, set_active_option
from fusion_bench.taskpool.image_classification import ImageClassificationTaskPool
from fusion_bench.utils import initialize_hydra_config, instantiate

fabric = L.Fabric(accelerator="auto", devices=1)
fabric.launch()

config = initialize_hydra_config(
    config_name="fabric_model_fusion",
    overrides=[
        "method=simple_average",
        "modelpool=ConvNextForImageClassification/convnext-base-224_8-tasks",
        "taskpool=ImageClassificationTaskPool/convnext-base-224_8-tasks.yaml",
    ],
)

algorithm: SimpleAverageAlgorithm = instantiate(config.method)
modelpool: ConvNextForImageClassificationPool = instantiate(config.modelpool)
taskpool: ImageClassificationTaskPool = instantiate(config.taskpool)
taskpool.fabric = fabric

models = {
    model_name: modelpool.load_model(model_name) for model_name in modelpool.model_names
}

# Wrap classification heads in a SwitchModule
heads = {model_name: m.classifier for model_name, m in models.items()}
head = SwitchModule(heads)

merged_model = algorithm.run(modelpool=BaseModelPool(models))
merged_model.classifier = head
report = taskpool.evaluate(merged_model)
print(report)
