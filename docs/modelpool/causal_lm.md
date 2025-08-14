# Large Language Models (Causal LMs)

The [`CausalLMPool`][fusion_bench.modelpool.CausalLMPool] class provides a unified interface for managing and loading causal language models from the Hugging Face Transformers library with flexible configuration options.

## Configuration

The [`CausalLMPool`][fusion_bench.modelpool.CausalLMPool] can be configured using YAML files. Here are the main configuration options:

### Basic Configuration

```{.yaml .annotate}
_target_: fusion_bench.modelpool.CausalLMPool # (1)
models:
  _pretrained_: path_to_pretrained_model # (2)
  model_a: path_to_model_a
  model_b: path_to_model_b
model_kwargs: # (3)
  torch_dtype: bfloat16  # or float16, float32, etc.
tokenizer: path_to_tokenizer # (4)
```

1. `_target_` indicates the modelpool class to be instantiated.
2. `_pretrained_`, `model_a`, and `model_b` indicates the name of the model to be loaded, if a plain string is given as the value, it will be passed to `AutoModelForCausalLM.from_pretrained` to load the model.
3. `model_kwargs` is a dictionary of keyword arguments to be passed to `AutoModelForCausalLM.from_pretrained`, can be overridden by passing kwargs to `modelpool.load_model` function.
4. `tokenizer` indicates the tokenizer to be loaded, if a plain string, it will be passed to `AutoTokenizer.from_pretrained`.

!!! note "Special Model Names in FusionBench"
    Names starting and ending with "_" are reserved for special purposes in FusionBench.
    For example, `_pretrained_` is a special model name in FusionBench, it is used to specify the pre-trained model to be loaded and pre-trained model can be loaded by calling `modelpool.load_pretrained_model()` or `modelpool.load_model("_pretrained_")`.

### Basic Usage

#### Information about the Model Pool

Get all the model names in the model pool except the special model names:

```python
>>> modelpool.model_names
['model_a', 'model_b']
```

Check if a pre-trained model is in the model pool:

```python
>>> modelpool.has_pretrained
True
```

Get all the model names in the model pool, including the special model names:

```python
>>> modelpool.all_model_names
['_pretrained_', 'model_a', 'model_b']
```

#### Loading and Saving Models and Tokenizers

Load a model from the model pool by model name:

```python
>>> model_a = modelpool.load_model("model_a")
```

Load a model from the model pool and pass/override additional arguments to the model constructor:

```python
>>> model_a_fp32 = modelpool.load_model("model_a", torch_dtype="float32")
```

Load the pre-trained model from the model pool:

```python
>>> pretrained_model = modelpool.load_pretrained_model()
# or equivalently
>>> pretrained_model = modelpool.load_model("_pretrained_")
```

Load the pre-trained model or the first model in the model pool:

```python
# if there is a pre-trained model in the model pool, then it will be loaded
# otherwise, the first model in the model pool will be loaded
>>> model = modelpool.load_pretrained_or_first_model()
```

Load the tokenizer from the model pool:

```python
>>> tokenizer = modelpool.load_tokenizer()
```

Save a model with tokenizer:

```python
# Save model with tokenizer
>>> modelpool.save_model(
    model=model,
    path="path/to/save",
    save_tokenizer=True,
    push_to_hub=False
)
```

### Advanced Configuration

You can also use more detailed configuration with explicit model and tokenizer settings:

```{.yaml .annotate}
_target_: fusion_bench.modelpool.CausalLMPool
models:
  _pretrained_:
    _target_: transformers.AutoModelForCausalLM # (1)
    pretrained_model_name_or_path: path_to_pretrained_model
  model_a:
    _target_: transformers.AutoModelForCausalLM
    pretrained_model_name_or_path: path_to_model_a
tokenizer:
  _target_: transformers.AutoTokenizer # (2)
  pretrained_model_name_or_path: path_to_tokenizer
model_kwargs:
  torch_dtype: bfloat16
```

1. `_target_` indicates the model class to be loaded, if a plain string is given as the value, it will be passed to `AutoModelForCausalLM.from_pretrained` to load the model.
    By setting `_target_`, you can use a custom model class or function to load the model.
    For example, you can use [`load_peft_causal_lm`][fusion_bench.modelpool.causal_lm.load_peft_causal_lm] to load a PEFT model.
2. `_target_` indicates the tokenizer class to be loaded, if a plain string is given as the value, it will be passed to `AutoTokenizer.from_pretrained` to load the tokenizer.
    By setting `_target_`, you can use a custom tokenizer class or function to load the tokenizer.

### Working with PEFT Models

```python
from fusion_bench.modelpool.causal_lm import load_peft_causal_lm

# Load a PEFT model
model = load_peft_causal_lm(
    base_model_path="path/to/base/model",
    peft_model_path="path/to/peft/model",
    torch_dtype="bfloat16",
    is_trainable=True,
    merge_and_unload=False
)
```

## Configuration Examples

### Single Model Configuration

```yaml title="config/modelpool/CausalLMPool/single_llama_model.yaml"
--8<-- "config/modelpool/CausalLMPool/single_llama_model.yaml"
```

### Multiple Models Configuration

Here we use models from [MergeBench](https://huggingface.co/MergeBench) as an example.

=== "gemma-2-2b"

    ```yaml title="config/modelpool/CausalLMPool/mergebench/gemma-2-2b.yaml"
    --8<-- "config/modelpool/CausalLMPool/mergebench/gemma-2-2b.yaml"
    ```

=== "gemma-2-2b-it"

    ```yaml title="config/modelpool/CausalLMPool/mergebench/gemma-2-2b-it.yaml"
    --8<-- "config/modelpool/CausalLMPool/mergebench/gemma-2-2b-it.yaml"
    ```

=== "gemma-2-9b"

    ```yaml title="config/modelpool/CausalLMPool/mergebench/gemma-2-9b.yaml"
    --8<-- "config/modelpool/CausalLMPool/mergebench/gemma-2-9b.yaml"
    ```

=== "gemma-2-9b-it"

    ```yaml title="config/modelpool/CausalLMPool/mergebench/gemma-2-9b-it.yaml"
    --8<-- "config/modelpool/CausalLMPool/mergebench/gemma-2-9b-it.yaml"
    ```

=== "Llama-3.1-8B"

    ```yaml title="config/modelpool/CausalLMPool/mergebench/Llama-3.1-8B.yaml"
    --8<-- "config/modelpool/CausalLMPool/mergebench/Llama-3.1-8B.yaml"
    ```

=== "Llama-3.1-8B-Instruct"

    ```yaml title="config/modelpool/CausalLMPool/mergebench/Llama-3.1-8B-Instruct.yaml"
    --8<-- "config/modelpool/CausalLMPool/mergebench/Llama-3.1-8B-Instruct.yaml"
    ```

=== "Llama-3.2-3B"

    ```yaml title="config/modelpool/CausalLMPool/mergebench/Llama-3.2-3B.yaml"
    --8<-- "config/modelpool/CausalLMPool/mergebench/Llama-3.2-3B.yaml"
    ```

=== "Llama-3.2-3B-Instruct"

    ```yaml title="config/modelpool/CausalLMPool/mergebench/Llama-3.2-3B-Instruct.yaml"
    --8<-- "config/modelpool/CausalLMPool/mergebench/Llama-3.2-3B-Instruct.yaml"
    ```

#### Merge Large Language Models with FusionBench

Merge gemma-2b models with simple average:

```bash
fusion_bench method=simple_average modelpool=CausalLMPool/mergebench/gemma-2-2b
```

Merge gemma-2b models with Task Arithmetic:

```bash
fusion_bench method=task_arithmetic modelpool=CausalLMPool/mergebench/gemma-2-2b
```

Merge Llama-3.1-8B models with Ties-Merging:

```bash
fusion_bench method=ties_merging modelpool=CausalLMPool/mergebench/Llama-3.1-8B
```

Merge Llama-3.1-8B-Instruct models with Dare-Ties, with 70% sparsity:

```bash
fusion_bench method=dare/ties_merging method.sparsity_ratio=0.7 modelpool=CausalLMPool/mergebench/Llama-3.1-8B-Instruct
```

## Special Features

### CausalLMBackbonePool

The [`CausalLMBackbonePool`][fusion_bench.modelpool.CausalLMBackbonePool] is a specialized version of `CausalLMPool` that returns only the transformer layers of the model. This is useful when you need to work with the model's backbone architecture directly.

```python
from fusion_bench.modelpool import CausalLMBackbonePool

backbone_pool = CausalLMBackbonePool.from_config(config)
layers = backbone_pool.load_model("model_a")  # Returns model.layers
```

## Implementation Details

- [fusion_bench.modelpool.CausalLMPool][]
- [fusion_bench.modelpool.CausalLMBackbonePool][]
- [fusion_bench.modelpool.causal_lm.load_peft_causal_lm][]
