# Task Arithmetic

In the rapidly advancing field of machine learning, multi-task learning has emerged as a powerful paradigm, allowing models to leverage information from multiple tasks to improve performance and generalization. One intriguing method in this domain is Task Arithmetic, which involves the combination of task-specific vectors derived from model parameters. 

<figure markdown="span">
  ![Image title](images/Task Arithmetic.png){ width="450" }
  <figcaption>Task Arithmetic. This figure credited to <sup id="fnref:2"><a class="footnote-ref" href="#fn:2">2</a></sup></figcaption>
</figure>

**Task Vector**. A task vector is used to encapsulate the adjustments needed by a model to specialize in a specific task. 
It is derived from the differences between a pre-trained model's parameters and those fine-tuned for a particular task. 
Formally, if $\theta_i$ represents the model parameters fine-tuned for the i-th task and $\theta_0$ denotes the parameters of the pre-trained model, the task vector for the i-th task is defined as:

$$\tau_i = \theta_i - \theta_0$$

This representation is crucial for methods like Task Arithmetic, where multiple task vectors are aggregated and scaled to form a comprehensive multi-task model.

**Task Arithmetic**[^1] begins by computing a task vector $\tau_i$ for each individual task, using the set of model parameters $\theta_0 \cup \{\theta_i\}_i$ where $\theta_0$ is the pre-trained model and $\theta_i$ are the fine-tuned parameters for i-th task.
These task vectors are then aggregated to form a multi-task vector.
Subsequently, the multi-task vector is combined with the pre-trained model parameters to obtain the final multi-task model.
This process involves scaling the combined vector element-wise by a scaling coefficient (denoted as $\lambda$), before adding it to the initial pre-trained model parameters. 
The resulting formulation for obtaining a multi-task model is expressed as 

$$ \theta = \theta_0 + \lambda \sum_{i} \tau_i. $$

The choice of the scaling coefficient $\lambda$ plays a crucial role in the final model performance. Typically, $\lambda$ is chosen based on validation set performance. 

## Examples

To use the Task Arithmetic algorithm, you can use the `TaskArithmeticAlgorithm` class from the `fusion_bench.method` module.

```python
from fusion_bench.method import TaskArithmeticAlgorithm
from omegaconf import DictConfig

# Instantiate the TaskArithmeticAlgorithm
method_config = {'name': 'task_arithmetic', 'scaling_factor': 0.5}
algorithm = TaskArithmeticAlgorithm(DictConfig(method_config))

# Assume we have a dict of PyTorch models (nn.Module instances) that we want to merge.
# The models should all have the same architecture.
# the dict must contain the pre-trained model with the key '_pretrained_', and arbitrary number of fine-tuned models.
models = {'_pretrained_': nn.Linear(10,10), 'model_1': nn.Linear(10,10), 'model_2': nn.Linear(10,10)}

# Run the algorithm on the models.
# This will return a new model that is the result of task arithmetic on the input models.
merged_model = algorithm.run(models)
```


## Code Integration

Configuration template for the Task Arithmetic algorithm:

```yaml title="config/method/task_arithmetic.yaml"
name: task_arithmetic
scaling_factor: 0.5 # Scaling factor for task vectors
```

Use the following command to run the Task Arithmetic algorithm:

```bash
fusion_bench method=task_arithmetic ...
```

For example, to run the Task Arithmetic algorithm on two models with scaling factor 0.5:

```bash
fusion_bench method=task_arithmetic \
    method.scaling_factor=0.5 \
  modelpool=clip-vit-base-patch32_svhn_and_mnist \
  taskpool=clip-vit-base-patch32_svhn_and_mnist
```

where the configuration for the model pool is:

```yaml title="config/modelpool/clip-vit-base-patch32_svhn_and_mnist.yaml"
type: huggingface_clip_vision
# the modelpool must contain the pre-trained model with the name '_pretrained_', 
# and arbitrary number of fine-tuned models.
models:
  - name: _pretrained_
    path: google/flan-t5-base
  - name: _pretrained_
    path: openai/clip-vit-base-patch32
  - name: svhn
    path: tanganke/clip-vit-base-patch32_svhn
  - name: mnist
    path: tanganke/clip-vit-base-patch32_mnist
```

and the configuration for the task pool:

```yaml title="config/taskpool/clip-vit-base-patch32_svhn_and_mnist.yaml"
type: clip_vit_classification

dataset_type: huggingface_image_classification
tasks:
  - name: svhn
    dataset:
      type: instantiate
      name: svhn
      object: 
        _target_: datasets.load_dataset
        _args_:
          - svhn
          - cropped_digits
        split: test
  - name: mnist
    dataset:
      name: mnist
      split: test

...
```


## References

::: fusion_bench.method.TaskArithmeticAlgorithm
    options:
        members: true


[^1]: (ICLR 2023) Editing Models with Task Arithmetic. http://arxiv.org/abs/2212.04089
[^2]: (ICLR 2024) AdaMerging: Adaptive Model Merging for Multi-Task Learning. http://arxiv.org/abs/2310.02575
