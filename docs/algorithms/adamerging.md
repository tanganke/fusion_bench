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

## AdaMerging Analysis

**Task-wise Coefficients.** 
The below Figure shows the changes during the iteration process of merging coefficient optimization of each task vector in Task-wise AdaMerging and AdaMerging++, which is shown every ten steps. We consistently observe that the merging coefficients of each task vector are inconsistent. When the number of tasks is relatively large, it is obviously undesirable to grid search the coefficients of each task, but our AdaMerging avoids this manual search process.

<figure markdown="span">
![alt text](images/adamerging_model_merging_coefficients.png){ width="900px" }
<figcaption style="max-width:90%" markdown="span">
Model merging coefficients $\{λ_k\}_{k=1}^K$ change with respect to training steps on ViT-B/32:  
(a) Task-wise AdaMerging; (b) Task-wise AdaMerging++. Each line represents the change process of the coefficient $λ_k$ of a task vector $T_k (k \in \{1, 2, . . . , K\})$.
</figcaption>
</figure>

**Layer-wise Coefficients.**
The following Figure shows the merging coefficients learned by Layer-wise AdaMerging and AdaMerging++ on ViT-B/32 respectively. We observed that:  

1. The coefficients learned by each layer of each task vector are different, which shows that the importance of each layer in the model merging process is different. 
2. The coefficients learned by shallow layers are generally smaller than those of deep layers, which indicates that shallow layers rely more on the weights of the pre-trained model rather than the weights provided by task vectors, while the deep layers rely more on the weights provided by the task vectors. This may be since the shallow layer learns general features, which are cross-task, while the deep layer learns task-specific features [^2]. This finding is also consistent with routing analysis in [^3].

<figure markdown="span">
![alt text](images/adamerging_layerwise_coefficients.png){ width="900px" }
<figcaption style="max-width:90%" markdown="span">
Learned model merging coefficients $\{λ_l^k\}^{K,L}_{k=1,l=1}$ of Layer-wise AdaMerging (Above) and AdaMerging++ (Below) on ViT-B/32. 
The $k$-th row represents the $k$-th task vector, the $l$-th column represents the $l$-th layer, and the intersection point represents the coefficient $λ^l_k$.
</figcaption>
</figure>

## Code Integration

Merge CLIP-ViT-B/32 models from eight downstream image classification tasks:

```bash
fusion_bench \
    method=adamerging \
        method.name=clip_layer_wise_adamerging \
        method.save_merging_weights=merging_weights.pt \
    modelpool=clip-vit-base-patch32_TA8 \
    taskpool=clip-vit-classification_TA8 \
    fabric.loggers.root_dir=outputs/logs/ViT-B-32 \
    fabric.loggers.name=clip_layer_wise_adamerging_adamerging
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

## Reference

### Task-Wise AdaMerging

::: fusion_bench.method.adamerging.task_wise_adamerging
::: fusion_bench.method.adamerging.clip_task_wise_adamerging

### Layer-Wise AdaMerging

::: fusion_bench.method.adamerging.layer_wise_adamerging
::: fusion_bench.method.adamerging.clip_layer_wise_adamerging

[^1]: (ICLR 2024) AdaMerging: Adaptive Model Merging for Multi-Task Learning. https://openreview.net/pdf?id=nZP6NgD3QY
[^2]: Jason Yosinski, Jeff Clune, Yoshua Bengio, and Hod Lipson. How transferable are features in deep neural networks? Advances in neural information processing systems, 27, 2014.
[^3]: A. Tang, L. Shen, Y. Luo, N. Yin, L. Zhang, and D. Tao, “Merging Multi-Task Models via Weight-Ensembling Mixture of Experts,” ICML 2024. doi: 10.48550/arXiv.2402.00433.
