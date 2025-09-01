# Image Classification Tasks for CLIP Models

## CLIPVisionModelTaskPool

The `CLIPVisionModelTaskPool` class is used to define image classification tasks for CLIP models. It provides methods to evaluate the performance of a given model on multiple datasets.

### Attributes

- `test_datasets`: A dictionary containing the test datasets.
- `processor`: The processor used for preprocessing the input data. This is used to set up the classifier.
- `data_processor`: The data processor used for processing the input data.
- `clip_model`: The CLIP model used for evaluation.
- `dataloader_kwargs`: Keyword arguments for the data loader.
- `layer_wise_feature_save_path`: Path to save the layer-wise features.
- `layer_wise_feature_first_token_only`: Boolean indicating whether to save only the first token of the features.
- `layer_wise_feature_max_num`: Maximum number of features to save.
- `fast_dev_run`: Boolean indicating whether to run in fast development mode.

### Methods

- `setup()`: Sets up the processor, data processor, CLIP model, test datasets, and data loaders.
- `evaluate(model)`: Evaluates the given model on the image classification task.
- `on_task_evaluation_begin(classifier, task_name)`: Called at the beginning of task evaluation to set up hooks for saving layer-wise features.
- `on_task_evaluation_end()`: Called at the end of task evaluation to save features and remove hooks.

### Configuration

The `CLIPVisionModelTaskPool` class can be configured using a YAML file. Here is an example configuration:

```yaml
test_datasets:
  dataset1: ...
  dataset2: ...
processor:
  _target_: transformers.CLIPProcessor.from_pretrained
  pretrained_model_name_or_path: openai/clip-vit-base-patch32
data_processor:
  _target_: transformers.CLIPProcessor.from_pretrained
  pretrained_model_name_or_path: openai/clip-vit-base-patch32
clip_model:
  _target_: transformers.CLIPModel.from_pretrained
  pretrained_model_name_or_path: openai/clip-vit-base-patch32
dataloader_kwargs:
  batch_size: 32
  num_workers: 4
layer_wise_feature_save_path: path/to/save/features
layer_wise_feature_first_token_only: true
layer_wise_feature_max_num: 1000
fast_dev_run: false
```

### References

For detailed API documentation, see [fusion_bench.taskpool.CLIPVisionModelTaskPool][fusion_bench.taskpool.CLIPVisionModelTaskPool] in the API reference.
