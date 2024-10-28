# Introduction to Model Pool Module

A modelpool is a collection of models that are utilized in the process of model fusion.
In the context of straightforward model fusion techniques, like averaging, only models with the same architecture are used.
While for more complex methods, such as AdaMerging [^1], each model is paired with a unique set of unlabeled test data. This data is used during the test-time adaptation phase.

## Yaml Configuration

A modelpool is specified by a `yaml` configuration file, which often contains the following fields:

- `type`: The name of the modelpool.
- `models`: A list of models, each model is dict with the following fields:
    - `name`: The name of the model. There are some special names that are reserved for specific purposes, such as `_pretrained_` for the pretrained model.
    - `path`: The path to the model file.
    - `type`: The type of the model. If this field is not specified, the type is inferred from the `model_type`.
  
For more complex model fusion techniques that requires data, the modelpool configuration file may also contain the following fields:

- `dataset_type`: The type of the dataset used for training the models in the modelpool.
- `datasets`: A list of datasets, each dataset is dict with the following fields:
    - `name`: The name of the dataset, which is used to pair the dataset with the corresponding model. The name of the dataset should match the name of the model.
    - `path`: The path to the dataset file.
    - `type`: The type of the dataset. If this field is not specified, the type is inferred from the `dataset_type`.

We provide a list of modelpools that contain models trained on different datasets and with different architectures.
Each modelpool is described in a separate document.

## Basic Usage

The model is not loaded by default when you initialize a modelpool, you can load a model from a modelpool by calling the `load_model` method:

```python
model = modelpool.load_model('model_name')
```


## References

::: fusion_bench.modelpool.BaseModelPool

[^1]: AdaMerging: Adaptive Model Merging for Multi-Task Learning. http://arxiv.org/abs/2310.02575
