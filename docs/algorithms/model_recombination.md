# Model Recombination

<figure markdown="span">
    ![alt text](images/fedmr_model_recombination.jpg){ width="900"}
    <figcaption>Credit to [FedMR](http://arxiv.org/abs/2305.10730) </figcaption>
</figure>

## Usage

`ModelRecombinationAlgorithm` is a class used to recombine models in a model pool. Here's how to use it:

First, import the necessary modules:

```python
from fusion_bench.method import ModelRecombinationAlgorithm
from fusion_bench.modelpool import ModelPool, to_modelpool
from torch import nn
```

Create an instance of `ModelRecombinationAlgorithm`:

```python
model_recombination = ModelRecombinationAlgorithm()
```

Create a model pool using the `to_modelpool` function. This function takes a list of models or a dict of models and converts it into a `ModelPool`:

```python
models = [nn.Linear(10, 10) for _ in range(3)]
modelpool = to_modelpool(models)
```

Use the `run` method of the `ModelRecombinationAlgorithm` instance to recombine the models in the model pool:

```python
new_modelpool = model_recombination.run(modelpool, return_modelpool=True)
```

The `run` method takes two arguments:

- `modelpool`: The model pool to recombine.
- `return_modelpool` (optional): A boolean indicating whether to return the entire model pool or just the first model. Defaults to `True`.

If `return_modelpool` is `True`, the `run` method returns a new `ModelPool` with the recombined models. If `False`, it returns the first model from the new model pool.

```python
new_model = model_recombination.run(modelpool, return_modelpool=False)
```

You can check the type of the returned value to ensure that the `run` method worked correctly:

```python
assert isinstance(new_modelpool, ModelPool)
assert isinstance(new_model, nn.Module)
```

## Code Integration

Configuration template for the model recombination algorithm:

```yaml title="config/method/model_recombination.yaml"
name: model_recombination
# if `return_model_pool` is not null, the argument `return_modelpool` passed to the `run` method will be ignored.
return_modelpool: null
```

Construct a model recombination using our CLI tool `fusion_bench`:

```bash
fusion_bench \
    method=model_recombination \
        method.return_modelpool=false \
    modelpool=... \
    taskpool=...
```


## Implementation Details

- [fusion_bench.method.ModelRecombinationAlgorithm][]
- [fusion_bench.method.model_recombination.recombine_modellist][]
- [fusion_bench.method.model_recombination.recombine_modeldict][]
- [fusion_bench.method.model_recombination.recombine_state_dict][]

