# OpenCLIPVisionModelPool

This is a model pool for OpenCLIP Vision models.

## Usage

By default, the model checkpoints are placed in the `.cache/task_vectors_checkpoints` directory.

```
.cache/
├── task_vectors_checkpoints/
│   ├── ViT-B-16
│   │   ├── Cars/finetuned.pt
│   │   ├── DTD/finetuned.pt
│   │   ├── ...
│   ├── ViT-B-32
│   │   ├── Cars/finetuned.pt
│   │   ├── DTD/finetuned.pt
│   │   ├── ...
│   ├── ...
│   ├── head_Cars.pt
│   ├── head_DTD.pt
│   ├── ...
|   └── zeroshot.pt
└── ...
```

## Model Configuration

The model pool supports several formats for model configuration:

1. **Direct Path (String)**: 
   - A string path to a model checkpoint in pickle format
   - Example: `"path/to/model.pt"`

2. **Pickle Path Configuration**:
   ```yaml
   model_name: "ViT-B-16"  # Name of the model
   pickle_path: "path/to/model.pt"  # Path to pickle file
   ```

3. **State Dict Configuration**:
   ```yaml
   model_name: "ViT-B-16"  # Name of the model
   state_dict_path: "path/to/state_dict.pt"  # Path to state dict file
   ```

4. **Hydra Configuration**:
   - Any configuration that can be instantiated using Hydra's `instantiate`

## Classification Head Configuration

The classification heads can be configured in two ways:

1. **Direct Path (String)**:
   - A string path to a classification head checkpoint in pickle format
   - Example: `"path/to/head.pt"`

2. **Hydra Configuration**:
   - Any configuration that can be instantiated using Hydra's `instantiate`

## Dataset Configuration

The model pool supports loading datasets in two ways:

1. **Direct Dataset Name (String)**:
   - A string identifier that can be loaded using `datasets.load_dataset`
   - Example: `"cifar10"`

2. **Custom Configuration**:
   - Any custom dataset configuration that can be handled by the parent class

## Example Configuration

Here's an example of a complete configuration:

```yaml
models:
  vit_b16:
    model_name: "ViT-B-16"
    pickle_path: ".cache/task_vectors_checkpoints/ViT-B-16/Cars/finetuned.pt"
  vit_b32:
    model_name: "ViT-B-32"
    state_dict_path: ".cache/task_vectors_checkpoints/ViT-B-32/DTD/finetuned.pt"

classification_heads:
  cars_head: ".cache/task_vectors_checkpoints/head_Cars.pt"
  dtd_head: ".cache/task_vectors_checkpoints/head_DTD.pt"
```

