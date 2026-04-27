# Comparing Multiple Fusion Methods Side-by-Side

When merging models, one of the first questions to answer is: which fusion method works best for your use case? FusionBench makes it straightforward to run several algorithms against the same set of models and compare their results in a single experiment or across multiple runs.

This guide shows how to compare `simple_average`, `task_arithmetic`, and `adamerging` on the same model pool and evaluation tasks.

## Why Compare Fusion Methods?

Different fusion methods make different assumptions about how models should be combined:

- **Simple Average** (`fusion_bench.method.SimpleAverageAlgorithm`): Equally weights all model parameters. This is the baseline that is surprisingly effective and serves as a strong reference point. Configuration requires no hyperparameters.
- **Task Arithmetic** (`fusion_bench.method.TaskArithmeticAlgorithm`): Computes task vectors (deltas from a pretrained base) then combines them with a configurable `scaling_factor`. This method explicitly leverages the pretrained model as an anchor.
- **AdaMerge** (`fusion_bench.method.AdamergingAlgorithm`): An adaptive method that determines per-parameter merge ratios based on the direction of parameter changes, often outperforming naive averaging when models were fine-tuned on complementary tasks.

## Prerequisites

Ensure FusionBench is installed and you have access to the models in your model pool configuration.

```bash
pip install fusion-bench
```

## Comparing via CLI

### Running Each Method Separately

The simplest approach is to run each fusion method individually, saving both the merged model and the evaluation report:

```bash
# 1. Simple Average
fusion_bench \
  method=simple_average \
  modelpool=ConvNextForImageClassification/convnext-base-224_8-tasks \
  taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8_val \
  merged_model_save_path=outputs/simple_average/merged_model.pt \
  report_save_path=outputs/simple_average/report.json
```

```bash
# 2. Task Arithmetic (default scaling_factor=0.3)
fusion_bench \
  method=task_arithmetic \
  modelpool=ConvNextForImageClassification/convnext-base-224_8-tasks \
  taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8_val \
  method.scaling_factor=0.3 \
  merged_model_save_path=outputs/task_arithmetic/merged_model.pt \
  report_save_path=outputs/task_arithmetic/report.json
```

```bash
# 3. Task Arithmetic with different scaling factor
fusion_bench \
  method=task_arithmetic \
  modelpool=ConvNextForImageClassification/convnext-base-224_8-tasks \
  taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8_val \
  method.scaling_factor=1.0 \
  merged_model_save_path=outputs/task_arithmetic_sf1/merged_model.pt \
  report_save_path=outputs/task_arithmetic_sf1/report.json
```

```bash
# 4. AdaMerge
fusion_bench \
  method=adamerging/clip \
  modelpool=ConvNextForImageClassification/convnext-base-224_8-tasks \
  taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8_val \
  merged_model_save_path=outputs/adamerging/merged_model.pt \
  report_save_path=outputs/adamerging/report.json
```

### Using Hydra Multi-Run for Sweeps

Hydra's multi-run feature (`-multirun` or `-m`) lets you sweep over hyperparameters in a single command. This is ideal for comparing methods or tuning hyperparameters like `scaling_factor`:

```bash
# Sweep scaling_factor across multiple values
fusion_bench -m \
  method=task_arithmetic \
  modelpool=ConvNextForImageClassification/convnext-base-224_8-tasks \
  taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8_val \
  method.scaling_factor=0.1,0.3,0.5,1.0 \
  report_save_path=outputs/ta_sweep/\${now:%Y%m%d_%H%M%S}/report.json
```

Each combination spawns a separate job. You can also sweep over methods:

```bash
# Sweep across multiple methods
fusion_bench -m \
  method=task_arithmetic,adamerging/clip \
  modelpool=ConvNextForImageClassification/convnext-base-224_8-tasks \
  taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8_val \
  report_save_path=outputs/method_sweep/\${now:%Y%m%d_%H%M%S}/report.json
```

### Dry Run to Inspect Configurations

Before running the full comparison, validate each configuration:

```bash
fusion_bench --cfg job \
  method=simple_average \
  modelpool=ConvNextForImageClassification/convnext-base-224_8-tasks \
  taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8_val
```

## Comparing via Python API

For more programmatic control, you can use the FusionBench API directly:

```python
from fusion_bench import BaseAlgorithm, BaseModelPool
from fusion_bench.utils import instantiate
from omegaconf import OmegaConf
import json
import os

# Define the model pool (shared across all methods)
modelpool_cfg = OmegaConf.load(
    "config/modelpool/ConvNextForImageClassification/convnext-base-224_8-tasks.yaml"
)
modelpool = BaseModelPool.from_config(modelpool_cfg)

# Define the methods to compare
method_configs = {
    "simple_average": "config/method/simple_average.yaml",
    "task_arithmetic_0.3": "config/method/task_arithmetic.yaml",  # default scaling_factor=0.3
}

# Override task_arithmetic scaling factor programmatically
ta_cfg = OmegaConf.load("config/method/task_arithmetic.yaml")
ta_cfg.scaling_factor = 0.5
method_configs["task_arithmetic_0.5"] = ta_cfg

results = {}

for method_name, method_cfg in method_configs.items():
    print(f"\n{'=' * 60}")
    print(f"Running method: {method_name}")
    print(f"{'=' * 60}")

    # Instantiate the algorithm
    algorithm = BaseAlgorithm.from_config(method_cfg)

    # Make a fresh copy of the model pool for each method
    fresh_modelpool = BaseModelPool.from_config(modelpool_cfg)

    # Run the fusion
    merged_model = algorithm.run(fresh_modelpool)

    # Evaluate (using a task pool)
    taskpool_cfg = OmegaConf.load(
        "config/taskpool/CLIPVisionModelTaskPool/clip-vit-classification_TA8_val.yaml"
    )
    taskpool = taskpool_cfg._target_.from_config(taskpool_cfg)
    report = taskpool.evaluate(merged_model)

    results[method_name] = report

    # Save the merged model
    save_dir = f"outputs/programmatic/{method_name}"
    os.makedirs(save_dir, exist_ok=True)
    modelpool.save_model(merged_model, f"{save_dir}/merged_model.pt")

    # Save the report
    with open(f"{save_dir}/report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"Report saved to {save_dir}/report.json")

# Print comparison summary
print("\n" + "=" * 60)
print("COMPARISON SUMMARY")
print("=" * 60)
for method_name, report in results.items():
    avg_acc = report.get("average", {}).get("accuracy", "N/A")
    print(f"  {method_name}: average accuracy = {avg_acc}")
```

## Interpreting Results

Each report JSON file contains per-task metrics plus an aggregated `average` entry. Typical structure:

```json
{
  "model_info": {
    "trainable_params": 87654321,
    "all_params": 87654321,
    "trainable_percentage": 1.0
  },
  "sun397": {
    "accuracy": 0.8234,
    "loss": 0.612
  },
  "stanford-cars": {
    "accuracy": 0.9102,
    "loss": 0.312
  },
  "average": {
    "accuracy": 0.8567,
    "loss": 0.478
  }
}
```

Key metrics to compare across methods:

| Metric | What it tells you |
|--------|-------------------|
| `average.accuracy` | Overall performance across all tasks |
| Per-task `accuracy` | Which method excels on specific domains |
| `average.loss` | Calibration and confidence of predictions |

## Tips for Effective Comparisons

1. **Fix the model pool**: Always use the same set of models across methods. Different model pools introduce confounding variables.

2. **Include the pretrained model**: Methods like Task Arithmetic require a `_pretrained_` entry in the model pool config. Ensure your YAML includes this:

   ```yaml
   _target_: fusion_bench.modelpool.ConvNextForImageClassificationPool
   models:
     _pretrained_: facebook/convnext-base-224
     sun397: tanganke/convnext-base-224_sun397
     stanford-cars: tanganke/convnext-base-224_stanford-cars
   ```

3. **Use `--fast_dev_run` for quick sanity checks**:

   ```bash
   fusion_bench --fast_dev_run method=task_arithmetic ...
   ```

   This evaluates on a single batch instead of the full dataset, letting you verify the pipeline works before committing to a full run.

4. **Control randomness**: Set a fixed seed for reproducibility:

   ```bash
   fusion_bench seed=42 method=adamerging/clip ...
   ```

5. **Use Hydra overrides to tune hyperparameters** without editing YAML files. For Task Arithmetic, the key parameter is `scaling_factor`:

   ```bash
   fusion_bench method=task_arithmetic method.scaling_factor=0.7 ...
   ```

## Next Steps

- Explore individual method documentation in `docs/algorithms/` for deeper understanding of each algorithm.
- See the [Save and Load Merged Models](save_and_load_merged_model.md) guide for persisting and reusing merged models.
- Check out [Hyperparameter Optimization with Optuna](../hyperparameter_optimization_with_optuna.md) for automated search across method hyperparameters.
