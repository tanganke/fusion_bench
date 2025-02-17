import torch


def compute_svd_and_compress(key, matrix, sv_reduction):
    """
    Computes the Singular Value Decomposition (SVD) of a given matrix and compresses it by reducing the number of singular values.

    Args:
        key (Any): An identifier for the matrix.
        matrix (torch.Tensor): The input matrix to decompose.
        sv_reduction (float): The fraction of singular values to retain (0 < sv_reduction <= 1).

    Returns:
        tuple: A tuple containing:
            - key (Any): The original identifier for the matrix.
            - u (torch.Tensor): The left singular vectors of the reduced SVD.
            - s (torch.Tensor): The reduced singular values.
            - v (torch.Tensor): The right singular vectors of the reduced SVD.
    """
    u, s, v = torch.linalg.svd(matrix, full_matrices=False)
    reduced_index_s = int(s.shape[0] * sv_reduction)
    return key, u[:, :reduced_index_s], s[:reduced_index_s], v[:reduced_index_s, :]


def compress_tv(task_vectors, sv_reduction):
    """
    Compress task vectors using Singular Value Decomposition (SVD).

    Args:
        task_vectors (dict): A dictionary where keys are dataset names and values are task vectors.
            Each task vector is expected to have a 'vector' attribute which is a dictionary
            with keys as layer names and values as layer matrices.
        sv_reduction (int): The fraction of singular values to keep for compression.

    Returns:
        dict: A dictionary with the same structure as `task_vectors`, but with each layer matrix
            replaced by its compressed SVD components (u, s, v) if the layer is 2-dimensional.
            If the layer is not 2-dimensional, it is stored as is under the key "dim1".
    """
    with torch.no_grad():
        svd_dict = {}
        for dataset, task_vector in task_vectors.items():
            svd_dict[dataset] = {}
            for key, layer in task_vector.vector.items():
                if len(layer.shape) == 2:  # and "text_projection" not in key:
                    _, u, s, v = compute_svd_and_compress(key, layer, sv_reduction)
                    svd_dict[dataset][key] = {"u": u, "s": s, "v": v}
                else:
                    svd_dict[dataset][key] = {"dim1": layer}
        return svd_dict
