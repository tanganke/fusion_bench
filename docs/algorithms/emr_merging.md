# EMR Merging

**EMR-MERGING** (Elect, Mask & Rescale-Merging) is a novel model merging method that combines a unified model with lightweight task-specific modulators. 
Unlike traditional methods that merge models into a single unified model, EMR-Merging creates:

1. A **unified task vector** elected from all model weights
2. **Task-specific masks** for direction alignment
3. **Task-specific rescalers** for magnitude alignment

The key advantage is that applying task-specific modulators to the unified model better approximates each task-specific model, significantly improving performance while requiring **no data, tuning, or additional training**.

## Usage

### Basic Example

```python
import lightning as L
from fusion_bench import (
    CLIPVisionModelPool,
    CLIPVisionModelTaskPool,
    instantiate,
    initialize_hydra_config,
)
from fusion_bench.models.hf_clip import HFCLIPClassifier
from fusion_bench.tasks.clip_classification import (
    get_classnames_and_templates,
)

# Initialize Fabric
fabric = L.Fabric(accelerator="auto", devices=1)
fabric.launch()

# Load configuration
config = initialize_hydra_config(
    config_name="fabric_model_fusion",
    overrides=[
        "method=emr_merging/emr_merging",
        "modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8",
        "taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8",
    ],
)

# Instantiate components
algorithm = instantiate(config.method)
modelpool = instantiate(config.modelpool)
taskpool = instantiate(config.taskpool)
taskpool.fabric = fabric

# Run EMR merging
emr_model = algorithm.run(modelpool)

# Evaluate on each task
if not taskpool._is_setup:
    taskpool.setup()

classifier = HFCLIPClassifier(
    taskpool.clip_model,
    processor=taskpool.processor,
)
classifier.clip_model.vision_model = emr_model
classifier = fabric.to_device(classifier)

results = {}
for task_name in taskpool._test_datasets:
    # Set task-specific modulator
    emr_model.set_task(task_name)
    
    # Set classification task
    classnames, templates = get_classnames_and_templates(task_name)
    classifier.set_classification_task(
        classnames=classnames,
        templates=templates,
    )
    
    # Evaluate
    result = taskpool._evaluate(
        classifier,
        test_loader=taskpool.test_dataloaders[task_name],
        task_name=task_name,
    )
    results[task_name] = result

print("Final results:", results)
```
