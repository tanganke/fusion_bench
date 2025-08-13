# Model Pool Module

A modelpool is a collection of models that are utilized in the process of model fusion.
In the context of straightforward model fusion techniques, like averaging, only models with the same architecture are used.
While for more complex methods, such as AdaMerging [^1], each model is paired with a unique set of unlabeled test data. This data is used during the test-time adaptation phase.

## Configuration Structure

Starting from version 0.2, modelpools use Hydra-based configuration with the `_target_` field to specify the class to instantiate. A modelpool configuration file typically contains the following fields:

### Core Fields

- `_target_`: The fully qualified class name of the modelpool (e.g., [`fusion_bench.modelpool.CLIPVisionModelPool`][])
- `models`: A dictionary of model configurations where each key is the model name and the value is the model configuration:
    - Special model names: `_pretrained_` refers to the base/pretrained model
    - Each model configuration should contain `_target_` field specifying how to load the model
    - Additional parameters can be passed to the model loading function

### Dataset Fields (Optional)

For model fusion techniques that require datasets:

- `train_datasets`: Dictionary of training dataset configurations
- `val_datasets`: Dictionary of validation dataset configurations  
- `test_datasets`: Dictionary of testing dataset configurations

Each dataset configuration should contain:

- `_target_`: The loading function (e.g., `datasets.load_dataset`)
- Additional parameters for the dataset loading function

### Additional Model-Specific Fields

Different modelpool types may include additional configuration fields:

- `processor`: For vision models, configuration for image preprocessors or tokenizers
- `tokenizer`: For language models, tokenizer configuration
- `model_kwargs`: Additional arguments passed to model loading functions
- `base_model`: Base model identifier used as a reference for other models

## Configuration Examples

### Basic CLIP Vision Model Pool

```yaml
_target_: fusion_bench.modelpool.CLIPVisionModelPool
base_model: openai/clip-vit-base-patch32
models:
  _pretrained_:
    _target_: transformers.CLIPVisionModel.from_pretrained
    pretrained_model_name_or_path: ${...base_model}
  finetuned_model:
    _target_: transformers.CLIPVisionModel.from_pretrained
    pretrained_model_name_or_path: path/to/finetuned/model
processor:
  _target_: transformers.CLIPProcessor.from_pretrained
  pretrained_model_name_or_path: ${..base_model}
```

### Causal Language Model Pool

```yaml
_target_: fusion_bench.modelpool.CausalLMPool
base_model: decapoda-research/llama-7b-hf
models:
  _pretrained_:
    _target_: transformers.LlamaForCausalLM.from_pretrained
    pretrained_model_name_or_path: ${...base_model}
  math_model:
    _target_: transformers.LlamaForCausalLM.from_pretrained
    pretrained_model_name_or_path: path/to/math/model
model_kwargs:
  torch_dtype: bfloat16
tokenizer:
  _target_: transformers.AutoTokenizer.from_pretrained
  pretrained_model_name_or_path: ${..base_model}
```

### Model Pool with Datasets

```yaml
_target_: fusion_bench.modelpool.CLIPVisionModelPool
base_model: openai/clip-vit-base-patch32
models:
  _pretrained_:
    _target_: transformers.CLIPVisionModel.from_pretrained
    pretrained_model_name_or_path: ${...base_model}
train_datasets:
  eurosat:
    _target_: datasets.load_dataset
    path: tanganke/eurosat
    split: train
  cars:
    _target_: datasets.load_dataset
    path: tanganke/stanford_cars
    split: train
processor:
  _target_: transformers.CLIPProcessor.from_pretrained
  pretrained_model_name_or_path: ${..base_model}
```

## Usage

### Creating a ModelPool

Starting from v0.2, modelpools can be created directly or through Hydra configuration:

```python
# Create from configuration file
from fusion_bench.utils import instantiate
from omegaconf import OmegaConf

config = OmegaConf.load("path/to/modelpool/config.yaml")
modelpool = instantiate(config)

# Create directly
from fusion_bench.modelpool import CLIPVisionModelPool
modelpool = CLIPVisionModelPool(
    models={
        "_pretrained_": {
            "_target_": "transformers.CLIPVisionModel.from_pretrained", 
            "pretrained_model_name_or_path": "openai/clip-vit-base-patch32"
        }
    }
)
```

### Loading Models

Models are loaded on-demand when requested:

```python
# Load a specific model
model = modelpool.load_model('_pretrained_')

# Load pretrained model (if available)
model = modelpool.load_pretrained_model()

# Load pretrained model or first available model
model = modelpool.load_pretrained_or_first_model()

# Iterate over all models
for model_name, model in modelpool.named_models():
    print(f"Processing {model_name}")
```

### Model Pool Properties

```python
# Check if pretrained model exists
if modelpool.has_pretrained:
    print("Pretrained model available")

# Get model names (excluding special models like _pretrained_)
model_names = modelpool.model_names

# Get all model names (including special models)  
all_names = modelpool.all_model_names

# Get number of models
num_models = len(modelpool)
```

### Working with Datasets

If datasets are configured, you can access them similarly:

```python
# Load datasets
train_dataset = modelpool.load_train_dataset('eurosat')
val_dataset = modelpool.load_val_dataset('eurosat')
test_dataset = modelpool.load_test_dataset('eurosat')

# Get dataset names
train_names = modelpool.train_dataset_names
val_names = modelpool.val_dataset_names
test_names = modelpool.test_dataset_names
```


## Implementation Details

- [fusion_bench.modelpool.BaseModelPool][]

[^1]: AdaMerging: Adaptive Model Merging for Multi-Task Learning. http://arxiv.org/abs/2310.02575
