from typing import List

import torch


def compute_task_singular_interference(weight_differences: List[torch.Tensor]) -> float:
    R"""
    Compute the singular interference of a list of weight differences $\{W_i - W_0\}_{i=1}^T$,
    where $W_0$ is the pre-trained model weight, $W_i$ is the weight of the i-th fine-tuned model
    and $T$ is the number of fine-tuned models.

    Args:
        weight_differences (List[torch.Tensor]): A list of weight differences $\{W_i - W_0\}_{i=1}^T$.

    Returns:
        float: The singular interference of the list of weight differences.
    """
    device = weight_differences[0].device
    dtype = weight_differences[0].dtype

    U = []
    S = []
    V = []
    for delta_w in weight_differences:
        u, s, vh = torch.linalg.svd(delta_w, full_matrices=False)
        U.append(u)
        S.append(s)
        V.append(vh.t())
    U = torch.cat(U, dim=0)
    S = torch.cat(S, dim=0)
    V = torch.cat(V, dim=0)

    singular_task_interference = torch.linalg.multi_dot(
        (
            U.t() @ U - torch.eye(U.shape[1], device=device, dtype=dtype),
            torch.diag(S),
            V.t() @ V - torch.eye(V.shape[1], device=device, dtype=dtype),
        )
    )
    singular_task_interference = torch.linalg.norm(singular_task_interference, ord="1")
    return singular_task_interference
