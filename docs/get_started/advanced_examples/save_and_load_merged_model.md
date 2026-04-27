# Saving and Loading Merged Models

After running a fusion algorithm, you typically want to save the merged model for later inference, deployment, or further experimentation. FusionBench provides built-in support for saving merged models via the `merged_model_save_path` CLI flag, and each ModelPool class defines how models are serialized.

## Saving Merged Models via CLI

The primary mechanism for saving merged models is the `merged_model_save_path` top-level configuration option. Set it to a file path or directory, and FusionBench calls `modelpool.save_model()` automatically after fusion completes.

```bash
fusion_bench \
  method=simple_average \
  modelpool=ConvNextForImageClassification/convnext-base-224_8-tasks \
  taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8_val \
  merged_model_save_path=outputs/merged_model
```

### How `save_model` Works

The behavior of `save_model` depends on the ModelPool implementation:

- **BaseModelPool** (default): Saves the model's state dict using `torch.save(model.state_dict(), path)`. The output is a `.pt` file.
- **CLIPVisionModelPool**: Calls `model.save_pretrained(path)` which saves a directory containing model weights, config, and tokenizer files.
- **ResNetForImageClassificationPool**: For torchvision models, saves the state dict. For transformers models, calls `model.save_pretrained(path)` and also saves the processor.
- **CausalLMPool (LLM)**: Calls `model.save_pretrained(path)` and optionally saves the tokenizer and pushes to HuggingFace Hub.

### Additional Save Options

Use `merged_model_save_kwargs` to pass extra keyword arguments to `modelpool.save_model()`. This is particularly useful for Hugging Face model pools:

```bash
fusion_bench \
  method=adamerging/clip \
  modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8 \
  taskpool=dummy \
  merged_model_save_path=outputs/merged_clip \
  merged_model_save_kwargs='{safe_serialization: true}'
```

For CausalLMPool, you can control tokenizer saving and hub upload:

```bash
fusion_bench -c job \
  method=linear/weighted_average_for_llama \
  modelpool=CausalLMPool/smile_mistral_exp_v4 \
  taskpool=dummy \
  merged_model_save_path=outputs/merged_llm \
  merged_model_save_kwargs='{push_to_hub: false, save_tokenizer: true}'
```

## Saving via Python API

When using the programmatic API, call `save_model` on your ModelPool instance:

```python
from fusion_bench import BaseAlgorithm, BaseModelPool
from omegaconf import OmegaConf

# Load configs
modelpool_cfg = OmegaConf.load(
    "config/modelpool/ConvNextForImageClassification/convnext-base-224_8-tasks.yaml"
)
method_cfg = OmegaConf.load("config/method/task_arithmetic.yaml")

# Instantiate
modelpool = BaseModelPool.from_config(modelpool_cfg)
algorithm = BaseAlgorithm.from_config(method_cfg)

# Run fusion
merged_model = algorithm.run(modelpool)

# Save the merged model
modelpool.save_model(merged_model, "outputs/my_merged_model.pt")
```

For HuggingFace-based pools that support `save_pretrained`:

```python
from fusion_bench.modelpool.clip_vision.modelpool import CLIPVisionModelPool

modelpool = CLIPVisionModelPool.from_config(modelpool_cfg)
merged_model = algorithm.run(modelpool)

# Save as a directory with config and weights
modelpool.save_model(merged_model, "outputs/merged_clip_vision/")
```

## Loading Merged Models for Inference

### Loading State Dict Models (PyTorch)

When the model was saved as a state dict (`.pt` file), load it into a fresh model instance:

```python
import torch
from transformers import ConvNextForImageClassification

# Create a model instance (use the same architecture as the base model)
model = ConvNextForImageClassification.from_pretrained("facebook/convnext-base-224")

# Load the merged state dict
state_dict = torch.load("outputs/my_merged_model.pt", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

# Run inference
# ...
```

### Loading HuggingFace Saved Models

When `save_pretrained` was used, the entire model is self-contained in a directory:

```python
from transformers import CLIPVisionModel, CLIPProcessor

# Load from the saved directory
model = CLIPVisionModel.from_pretrained("outputs/merged_clip_vision/")
processor = CLIPProcessor.from_pretrained("outputs/merged_clip_vision/")

# Run inference
# inputs = processor(images=my_image, return_tensors="pt")
# outputs = model(**inputs)
```

### Loading via ModelPool

For the most seamless experience, use the ModelPool itself to load models:

```python
from fusion_bench.modelpool.clip_vision.modelpool import CLIPVisionModelPool
from omegaconf import OmegaConf

# Reference the saved model in a new model pool config
config = {
    "_target_": "fusion_bench.modelpool.clip_vision.modelpool.CLIPVisionModelPool",
    "models": {
        "my_merged": "outputs/merged_clip_vision/"
    }
}

modelpool = CLIPVisionModelPool.from_config(OmegaConf.create(config))
model = modelpool.load_model("my_merged")
```

## Common Patterns

### Save Multiple Methods for A/B Testing

```bash
for method in simple_average task_arithmetic adamerging/clip; do
    fusion_bench \
      method=$method \
      modelpool=ConvNextForImageClassification/convnext-base-224_8-tasks \
      taskpool=dummy \
      merged_model_save_path="outputs/models/$method_merged" \
      report_save_path=false
done
```

### Save and Load in a Single Script

```python
import torch
import os
from fusion_bench import BaseAlgorithm, BaseModelPool
from omegaconf import OmegaConf

SAVE_PATH = "outputs/experiment_001/merged.pt"

def train_and_save():
    """Run fusion and save the model."""
    modelpool_cfg = OmegaConf.load(
        "config/modelpool/ConvNextForImageClassification/convnext-base-224_8-tasks.yaml"
    )
    method_cfg = OmegaConf.load("config/method/task_arithmetic.yaml")

    modelpool = BaseModelPool.from_config(modelpool_cfg)
    algorithm = BaseAlgorithm.from_config(method_cfg)

    merged_model = algorithm.run(modelpool)
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    modelpool.save_model(merged_model, SAVE_PATH)
    print(f"Model saved to {SAVE_PATH}")
    return SAVE_PATH

def load_and_inference():
    """Load a saved model and run inference."""
    state_dict = torch.load(SAVE_PATH, map_location="cpu")

    from transformers import ConvNextForImageClassification
    model = ConvNextForImageClassification.from_pretrained(
        "facebook/convnext-base-224"
    )
    model.load_state_dict(state_dict)
    model.eval()

    # Your inference logic here
    return model

# Usage
model_path = train_and_save()
inference_model = load_and_inference()
```

## Troubleshooting

- **File not found on load**: Verify the path matches exactly what was passed to `merged_model_save_path`. For HuggingFace saves, the path is a directory, not a file.
- **Shape mismatch on state dict load**: Ensure the model architecture you are loading into matches the fused model. Use the same base model (e.g., `_pretrained_` entry) for loading.
- **Missing processor/tokenizer**: If you need the processor or tokenizer, use a ModelPool that supports `save_pretrained` rather than the base `save_model` which only saves state dicts.

## Related

- [Comparing Multiple Fusion Methods](ensemble_comparison.md) — Use saved models from different methods for side-by-side comparison.
- [CLI Reference](../../cli/fusion_bench.md) — Full documentation of `merged_model_save_path` and `merged_model_save_kwargs` flags.
