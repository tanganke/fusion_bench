# SimpleProfilerMixin

The `SimpleProfilerMixin` provides lightweight, wall-clock timing for code blocks within your FusionBench algorithms. Built on Lightning's `SimpleProfiler`, it measures execution time for named actions, making it easy to identify performance bottlenecks without the overhead of statistical profilers.

## Overview

Unlike statistical profilers (cProfile, pyinstrument) that sample or trace every function call, `SimpleProfilerMixin` focuses on timing specific, named blocks of code. This makes it ideal for:

- Measuring the duration of distinct pipeline stages (data loading, model fusion, evaluation)
- Comparing execution times across algorithm variants
- Quick performance diagnostics during development

## Basic Usage

### Context Manager (Recommended)

The cleanest way to profile code is using the `with` context manager:

```python
from fusion_bench.mixins import SimpleProfilerMixin
from fusion_bench.method import BaseAlgorithm

class MyAlgorithm(SimpleProfilerMixin, BaseAlgorithm):
    def run(self, modelpool):
        with self.profile("model_loading"):
            models = [modelpool.load_model(name) for name in modelpool.model_names]

        with self.profile("weight_averaging"):
            merged = average_models(models)

        with self.profile("device_transfer"):
            merged = self.to_device(merged)

        # Print the profiling summary
        self.print_profile_summary("Fusion Pipeline Timing")
```

### Manual Start/Stop

For cases where context managers don't fit your code structure:

```python
def run(self, modelpool):
    self.start_profile("data_preparation")
    # ... data loading and preprocessing ...
    self.stop_profile("data_preparation")

    self.start_profile("fusion_step")
    # ... model fusion logic ...
    self.stop_profile("fusion_step")
```

## Key Methods

### `profile(action_name)` — Context Manager

Starts profiling on entry and stops on exit (even if an exception occurs). The action name appears in the summary output.

```python
with self.profile("training_epoch"):
    for batch in dataloader:
        loss = train_step(batch)
```

### `start_profile(action_name)` and `stop_profile(action_name)`

Manual control for more flexible profiling:

```python
self.start_profile("inference")
results = model.predict(test_data)
self.stop_profile("inference")
```

### `print_profile_summary(title=None)`

Outputs a formatted table showing all profiled actions with their execution times. Decorated with `@rank_zero_only`, so it only prints on the main process in distributed settings.

```python
self.print_profile_summary("Algorithm Performance Breakdown")
```

### `profiler` Property

Access the underlying `SimpleProfiler` instance directly for advanced usage. Lazy-initialized on first access.

## Integration with Other Mixins

The mixin works seamlessly with `LightningFabricMixin`:

```python
class MyAlgorithm(SimpleProfilerMixin, LightningFabricMixin, BaseAlgorithm):
    def run(self, modelpool):
        with self.profile("model_setup"):
            model = modelpool.load_model("_pretrained_")
            model = self.to_device(model)
            model = self.fabric.setup(model)

        with self.profile("training"):
            for epoch in range(self.num_epochs):
                self._train_epoch(model, epoch)

        self.print_profile_summary("Training Pipeline")
```

## Practical Examples

### Profiling a Model Merging Algorithm

```python
class MergingAlgorithm(SimpleProfilerMixin, BaseAlgorithm):
    def run(self, modelpool):
        # Profile each stage of the merging pipeline
        with self.profile("load_models"):
            pretrained = modelpool.load_model("_pretrained_")
            finetuned_models = [
                modelpool.load_model(name) for name in modelpool.model_names
            ]

        with self.profile("compute_task_vectors"):
            task_vectors = [
                compute_task_vector(ft, pretrained)
                for ft in finetuned_models
            ]

        with self.profile("apply_masking"):
            masked_vectors = [self.apply_mask(tv) for tv in task_vectors]

        with self.profile("merge_vectors"):
            merged_vector = sum(masked_vectors)

        with self.profile("apply_to_pretrained"):
            merged_model = apply_vector(pretrained, merged_vector)

        self.print_profile_summary("Merging Algorithm Profile")
        return merged_model
```

### Profiling Ensemble Evaluation

```python
class EnsembleAlgorithm(SimpleProfilerMixin, BaseAlgorithm):
    def run(self, modelpool):
        with self.profile("load_all_models"):
            models = self._load_models(modelpool)

        with self.profile("forward_pass"):
            predictions = self._forward(models)

        with self.profile("ensemble_aggregation"):
            final_pred = self._aggregate(predictions)

        with self.profile("metric_computation"):
            metrics = self._compute_metrics(final_pred)

        self.print_profile_summary("Ensemble Evaluation Profile")
        return metrics
```

## Best Practices

1. **Use descriptive names**: Choose action names that clearly describe the operation (e.g., "load_pretrained_model" not "step1").
2. **Profile at the right granularity**: Profile coarse-grained blocks (entire phases) rather than individual lines. For fine-grained profiling, use `PyinstrumentProfilerMixin`.
3. **Always print the summary**: Call `print_profile_summary()` after your profiling blocks to see the results.
4. **Distributed awareness**: The summary only prints on rank zero. In multi-GPU runs, timing reflects the synchronized wall-clock time.
5. **Don't profile in production**: Remove or disable profiling calls before benchmarking final performance, as the act of profiling adds small overhead.

## SimpleProfiler vs PyinstrumentProfiler

| Aspect | SimpleProfilerMixin | PyinstrumentProfilerMixin |
|---|---|---|
| Granularity | Block-level (named sections) | Function-level (full call tree) |
| Overhead | Minimal | Moderate |
| Use case | Pipeline timing | Bottleneck diagnosis |
| Output | Duration table | Call tree with percentages |

## Reference

::: fusion_bench.mixins.simple_profiler.SimpleProfilerMixin
