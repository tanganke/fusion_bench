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
    # 保存原始数据类型
    original_dtype = matrix.dtype
    
    # 如果是BFloat16类型，转换为float32
    if matrix.dtype == torch.bfloat16:
        matrix = matrix.to(torch.float32)
    
    # 执行SVD操作
    u, s, v = torch.linalg.svd(matrix, full_matrices=False)
    
    # 计算压缩后的索引
    reduced_index_s = int(s.shape[0] * sv_reduction)
    
    # 获取压缩后的结果
    u_reduced = u[:, :reduced_index_s]
    s_reduced = s[:reduced_index_s]
    v_reduced = v.T[:, :reduced_index_s]
    
    # 如果原始数据是BFloat16，将结果转换回BFloat16
    if original_dtype == torch.bfloat16:
        u_reduced = u_reduced.to(torch.bfloat16)
        s_reduced = s_reduced.to(torch.bfloat16)
        v_reduced = v_reduced.to(torch.bfloat16)
        # 注意：保持原始的u, s, v为float32类型，因为这些可能用于后续计算
    
    # 返回结果
    return (
        key,
        u_reduced,
        s_reduced,
        v_reduced,
        u,
        s,
        v.T,
    )


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
