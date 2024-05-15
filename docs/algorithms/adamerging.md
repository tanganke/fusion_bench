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

[^1]: (ICLR 2024) AdaMerging: Adaptive Model Merging for Multi-Task Learning. http://arxiv.org/abs/2310.02575