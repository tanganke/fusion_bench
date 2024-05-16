# Task Arithmetic

In the rapidly advancing field of machine learning, multi-task learning has emerged as a powerful paradigm, allowing models to leverage information from multiple tasks to improve performance and generalization. One intriguing method in this domain is Task Arithmetic, which involves the combination of task-specific vectors derived from model parameters. 

<figure markdown="span">
  ![Image title](images/Task Arithmetic.png){ width="450" }
  <figcaption>Task Arithmetic. Credit to <sup id="fnref:2"><a class="footnote-ref" href="#fn:2">2</a></sup></figcaption>
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

::: fusion_bench.method.TaskArithmeticAlgorithm
    options:
        members: true


[^1]: (ICLR 2023) Editing Models with Task Arithmetic. http://arxiv.org/abs/2212.04089
[^2]: (ICLR 2024) AdaMerging: Adaptive Model Merging for Multi-Task Learning. http://arxiv.org/abs/2310.02575
