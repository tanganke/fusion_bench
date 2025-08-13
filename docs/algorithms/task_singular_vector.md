# Task Singular Vector

Task Singular Vector Merging (TSVM) is a model merging technique that uses Singular Value Decomposition (SVD) to combine multiple task-specific fine-tuned models into a single merged model. This method is particularly effective for merging models that have been fine-tuned on different tasks from a common pre-trained base model.

## Mathematical Foundation

### Problem Setup

Let $\theta_0$ be the parameters of a pre-trained base model, and $\{\theta_1, \theta_2, \ldots, \theta_n\}$ be the parameters of $n$ models fine-tuned on different tasks from the same base model $\theta_0$.

### Task Vector Computation

For each fine-tuned model $i$, we first compute the **task vector** $\tau_i$, which represents the parameter changes from the base model:

$$\tau_i = \theta_i - \theta_0$$

### SVD-Based Merging Algorithm

The core innovation of TSVM lies in applying SVD to the task vectors and then merging them in the singular vector space.

#### Step 1: SVD Decomposition

For each parameter matrix $W^{(i)}_k$ in task vector $\tau_i$ (where $k$ indexes the layer/parameter group), if the matrix is 2-dimensional, we compute its SVD:

$$W^{(i)}_k = U^{(i)}_k S^{(i)}_k (V^{(i)}_k)^T$$

where:

- $U^{(i)}_k \in \mathbb{R}^{m \times r}$ contains the left singular vectors
- $S^{(i)}_k \in \mathbb{R}^{r \times r}$ is a diagonal matrix of singular values  
- $V^{(i)}_k \in \mathbb{R}^{n \times r}$ contains the right singular vectors
- $r = \min(m, n)$ is the rank

#### Step 2: Dimension Reduction and Concatenation

To reduce memory usage and computational complexity, we apply a reduction factor:

$$\text{reduction_factor} = \frac{1}{T}$$

where $T$ is the number of tasks.


For each task $i$, we select only the top $\lfloor r \cdot \text{reduction_factor} \rfloor$ singular components and place them in task-specific positions within larger matrices:

$$U = [U^{(1)}_k[:, :d], U^{(2)}_k[:, :d], \ldots, U^{(T)}_k[:, :d]]$$

$$S = \text{diag}(S^{(1)}_k[:d], S^{(2)}_k[:d], \ldots, S^{(T)}_k[:d])$$

$$V = [V^{(1)}_k[:, :d], V^{(2)}_k[:, :d], \ldots, V^{(T)}_k[:, :d]]$$

where $d = \lfloor r \cdot \text{reduction_factor} \rfloor$.

#### Step 3: Second-Level SVD

We then compute the SVD of the concatenated matrices:

$$U = \hat{U} \hat{S} (\hat{V})^T$$

$$V = \hat{U} \hat{S} (\hat{V})^T$$

#### Step 4: Final Reconstruction

The merged task vector for parameter $k$ is reconstructed as:

$$\tau_{\text{TSVM}} = \hat{U} \cdot (\hat{V})^T \cdot \text{diag}(S) \cdot \hat{U} \cdot (\hat{V})^T$$

### Handling Non-2D Parameters

For parameters that are not 2-dimensional (e.g., bias vectors, normalization parameters), TSVM simply computes the arithmetic mean:

$$\tau_{\text{TSVM,non-2D}} = \frac{1}{T} \sum_{i=1}^{T} \tau^{(i)}_k$$

### Final Model Construction

The final merged model parameters are obtained by adding the merged task vector to the base model:

$$\theta_{\text{TSVM}} = \theta_0 + \alpha \tau_{\text{TSVM}}$$

where $\alpha$ is an optional global scaling factor.


## Implementation Details

- [fusion_bench.method.TaskSingularVectorMerging][]
