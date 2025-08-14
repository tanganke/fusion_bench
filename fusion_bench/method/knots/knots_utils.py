import torch


def subspace_alignment(
    delta_weights: list[torch.Tensor],
    svd_dtype: torch.dtype | None = torch.float64,
    eps: float = 1e-4,
):
    """
    Reference: Model merging with SVD to tie the Knots. http://arxiv.org/abs/2410.19735
    """
    if svd_dtype is None:
        svd_dtype = delta_weights[0].dtype
    original_dtype = delta_weights[0].dtype
    output_dim, input_dim = delta_weights[0].size()
    concat_task_vector = torch.cat(delta_weights, dim=1)
    U, S, Vh = torch.linalg.svd(concat_task_vector.to(svd_dtype), full_matrices=False)
    # Keep only supported basis components
    U = U[:, S > eps].to(original_dtype)
    Vh = Vh[S > eps].to(original_dtype)
    S = S[S > eps].to(original_dtype)
    Vhs = torch.split(Vh, input_dim, dim=1)
    return U, S, Vhs
