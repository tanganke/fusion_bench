# AdaMerging

<figure markdown="span">
    ![alt text](images/adamerging.png){ width="750" }
    <figcaption>Task Vector, Task Arithmetic, and AdaMerging. Credit to <sup id="fnref:1"><a class="footnote-ref" href="#fn:1">1</a></sup></figcaption>
</figure>

In the complex landscape of multi-task learning, AdaMerging has emerged as a potent method for adaptively merging model parameters to optimize performance across tasks. Unlike traditional fixed-coefficient methods, AdaMerging autonomously learns merging coefficients, offering a more refined and responsive approach[^1]. 

The cornerstone of AdaMerging lies in its adaptive nature, where it learns the coefficients for merging either on a task-wise or layer-wise basis. This adaptability is driven by an entropy minimization strategy applied to unlabeled test samples as a surrogate objective function, which serves to refine the merging coefficients for optimal performance.

Task-wise AdaMerging is formulated as:

$$
\theta = \theta_0 + \sum_{i=1}^{n} \lambda_i \tau_i
$$

where $\lambda_i$ represents the merging coefficient for the \(i\)-th task, and $\tau_i$ denotes the task vector for the \(i\)-th task.

On the other hand, Layer-wise AdaMerging is articulated as:

$$
\theta^l = \theta_0^l + \sum_{i=1}^{n} \lambda^{l}_{i} \tau^{l}_{i}
$$

where the merging coefficient $\lambda^{l}_{i}$ and task vector $\tau^{l}_{i}$ are specific to each layer \(l\) of the model.

By leveraging this adaptive learning approach, AdaMerging significantly enhances the model's ability to generalize across tasks and layers, resulting in a more robust and finely-tuned performance profile. The method’s reliance on entropy minimization ensures that the merging process continually seeks the most informative and stable configuration, adapting to the specific needs of the dataset and tasks at hand.

## Code Integration

Merge CLIP-ViT-B/32 models from eight downstream image classification tasks:

```bash
fusion_bench \
    method=adamerging \
        method.name=clip_layer_wise_adamerging \
        method.save_merging_weights=merging_weights.pt \
    modelpool=clip-vit-base-patch32_TA8 \
    taskpool=clip-vit-classification_TA8 \
    fabric_logger.root_dir=outputs/logs/ViT-B-32 \
    fabric_logger.name=clip_layer_wise_adamerging_adam
```

Part of the output:

```
Profiler Report

----------------------------------------------------------------------------------------------------------------------------------
|  Action                       |  Mean duration (s)    |  Num calls            |  Total time (s)       |  Percentage %         |
----------------------------------------------------------------------------------------------------------------------------------
|  Total                        |  -                    |  26001                |  724.65               |  100 %                |
----------------------------------------------------------------------------------------------------------------------------------
|  backward pass                |  0.060172             |  8000                 |  481.38               |  66.429               |
|  forward pass                 |  0.016124             |  8000                 |  128.99               |  17.801               |
|  data loading                 |  0.0063443            |  8000                 |  50.754               |  7.004                |
|  merging weights              |  0.050735             |  1000                 |  50.735               |  7.0013               |
|  construct the wrapped model  |  7.2558               |  1                    |  7.2558               |  1.0013               |
|  optimizer step               |  0.00098186           |  1000                 |  0.98186              |  0.13549              |
----------------------------------------------------------------------------------------------------------------------------------
```

[^1]: (ICLR 2024) AdaMerging: Adaptive Model Merging for Multi-Task Learning. https://openreview.net/pdf?id=nZP6NgD3QY
