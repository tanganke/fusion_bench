import torch
import math


def compute_svd_dict(task_vectors, config):
    """
    Computes the Singular Value Decomposition (SVD) for each task vector in the provided datasets and stores the results in a dictionary.

    Args:
        task_vectors (list): A list of task vector objects, where each task vector contains a dictionary of matrices to be decomposed.
        config (object): Configuration object containing the list of datasets under the attribute `DATASETS`.

    Returns:
        dict: A dictionary where each key is a dataset name and the value is another dictionary containing the SVD components ('u', 's', 'v') for each matrix in the task vector.
        If a matrix is not 2-dimensional or contains 'text_projection' in its key, it is stored under the key 'dim1' without decomposition.
    """
    sv_reduction = 1 / len(config.DATASETS)
    with torch.no_grad():
        svd_dict = {}
        for i, (task_vector, dataset) in enumerate(zip(task_vectors, config.DATASETS)):
            svd_dict[dataset] = {}
            print(f"Computing SVD for {dataset}...")
            for key in task_vector.vector:
                svd_dict[dataset][key] = {}
                if (
                    len(task_vector.vector[key].shape) == 2
                    and "text_projection" not in key
                ):
                    u, s, v = torch.linalg.svd(
                        task_vector.vector[key], full_matrices=False
                    )
                    reduced_index_s = int(s.shape[0] * sv_reduction)

                    temp_u = torch.zeros_like(u)
                    # select only the first reduced_index_s columns of u and place them
                    temp_u[:, i * reduced_index_s : (i + 1) * reduced_index_s] = u[
                        :, :reduced_index_s
                    ]
                    svd_dict[dataset][key]["u"] = temp_u

                    temp_s = torch.zeros_like(s)
                    temp_s[i * reduced_index_s : (i + 1) * reduced_index_s] = s[
                        :reduced_index_s
                    ]

                    svd_dict[dataset][key]["s"] = temp_s  # s_reduced

                    # select only the first reduced_index_s rows of v and place them
                    temp_v = torch.zeros_like(v)
                    temp_v[i * reduced_index_s : (i + 1) * reduced_index_s, :] = v[
                        :reduced_index_s, :
                    ]

                    svd_dict[dataset][key]["v"] = temp_v

                    # temp_mat = temp_u @ torch.diag_embed(temp_s) @ temp_v
                else:
                    svd_dict[dataset][key]["dim1"] = task_vector.vector[key]
    return svd_dict


def sum_svd_dict(svd_dict, config):
    """
    Sums the Singular Value Decomposition (SVD) components from multiple datasets and computes a new vector.

    Args:
        svd_dict (dict): A dictionary containing SVD components for multiple datasets. The structure of the dictionary is expected to be:
                         {
                             dataset_name: {
                                 key: {
                                     "u": tensor,
                                     "s": tensor,
                                     "v": tensor,
                                     "dim1": tensor (optional)
                                 }
                             }
                         }
        config (object): A configuration object that contains a list of dataset names under the attribute `DATASETS`.

    Returns:
        dict: A dictionary containing the merged SVD components or averaged "dim1" values for each key.
    """
    print("Summing SVD...")
    new_vector = {}
    for key in svd_dict[config.DATASETS[0]]:
        if "u" in svd_dict[config.DATASETS[0]][key].keys():
            sum_u = sum([svd_dict[dataset][key]["u"] for dataset in config.DATASETS])
            sum_s = sum([svd_dict[dataset][key]["s"] for dataset in config.DATASETS])
            sum_v = sum([svd_dict[dataset][key]["v"] for dataset in config.DATASETS])

            u_u, s_u, v_u = torch.linalg.svd(sum_u, full_matrices=False)
            u_v, s_v, v_v = torch.linalg.svd(sum_v, full_matrices=False)
            new_vector[key] = torch.linalg.multi_dot(
                (
                    u_u,
                    v_u,
                    torch.diag(sum_s),
                    u_v,
                    v_v,
                )
            )
        else:
            for i, dataset in enumerate(config.DATASETS, start=1):
                if i == 1:
                    new_vector[key] = svd_dict[dataset][key]["dim1"]
                else:
                    new_vector[key] += (
                        svd_dict[dataset][key]["dim1"] - new_vector[key]
                    ) / i
    return new_vector


###############
##### LOSSLESS Orthogonalization
def compute_and_sum_svd_mem_reduction_lossless(task_vectors, config):
    """
    Computes the Singular Value Decomposition (SVD) for each task vector and merge the results.

    This function performs the following steps:
    1. Iterates over each layer in the task vectors.
    2. For each layer, it computes the SVD of the corresponding matrix if it is a 2D tensor excluding "text_projection".
    3. Concatenate the U_i, S_i, and V_i matrices from the SVD across all tasks.
    4. If the vector is not a 2D tensor or is "text_projection", it computes the mean of the vectors.
    5. After concatenating the SVD components, recomputes the SVD of the summed U and V matrices and constructs the merged layer.

    Args:
        task_vectors (list): A list of task vectors, where each task vector is a dictionary containing the vectors for each task.
        config (object): A configuration object containing the device and dataset information.

    Returns:
        dict: A dictionary containing the new vectors after summing the SVD components.
    """
    # becareful wit vit-l on 20 task it does not fit in GPU or in 64 GB RAM (try without last layer)
    device = config.device
    print("Computing SVD...")
    with torch.no_grad():
        new_vector = {}
        for key in task_vectors[0].vector:
            new_vector[key] = {}
            for i, (task_vector, dataset) in enumerate(
                zip(task_vectors, config.DATASETS)
            ):
                vec = task_vector.vector[key].to(device)

                if (
                    len(task_vector.vector[key].shape) == 2
                    and "text_projection" not in key
                ):

                    u, s, v = torch.linalg.svd(vec, full_matrices=False)

                    if i == 0:
                        print(f"Computed SVD for {key}...")
                        sum_u = torch.zeros(
                            u.shape[0], u.shape[1] * config.num_tasks, device=device
                        )
                        sum_s = torch.zeros(
                            s.shape[0] * config.num_tasks, device=device
                        )
                        sum_v = torch.zeros(
                            v.shape[0] * config.num_tasks, v.shape[1], device=device
                        )
                    reduced_index_s = s.shape[0]

                    # select only the first reduced_index_s columns of u and place them
                    sum_u[:, i * reduced_index_s : (i + 1) * reduced_index_s] = u[
                        :, :reduced_index_s
                    ]
                    sum_s[i * reduced_index_s : (i + 1) * reduced_index_s] = s[
                        :reduced_index_s
                    ]
                    # select only the first reduced_index_s rows of v and place them
                    sum_v[i * reduced_index_s : (i + 1) * reduced_index_s, :] = v[
                        :reduced_index_s, :
                    ]

                else:
                    if i == 0:
                        new_vector[key] = vec.clone()
                    else:
                        new_vector[key] += (vec - new_vector[key]) / (i + 1)

            if len(task_vector.vector[key].shape) == 2 and "text_projection" not in key:
                u_u, s_u, v_u = torch.linalg.svd(sum_u, full_matrices=False)
                u_v, s_v, v_v = torch.linalg.svd(sum_v, full_matrices=False)

                new_vector[key] = torch.linalg.multi_dot(
                    (
                        u_u,
                        v_u,
                        torch.diag(sum_s),
                        u_v,
                        v_v,
                    )
                )

    return new_vector


###############
##### LOSSLESS EIGENDECOMP
def compute_and_sum_svd_mem_reduction_lossless_eigen(task_vectors, config):
    """
    Computes the Singular Value Decomposition (SVD) for each task vector and merge the results.

    This function performs the following steps:
    1. Iterates over each layer in the task vectors.
    2. For each layer, it computes the SVD of the corresponding matrix if it is a 2D tensor excluding "text_projection".
    3. Concatenate the U_i, S_i, and V_i matrices from the SVD across all tasks.
    4. If the vector is not a 2D tensor or is "text_projection", it computes the mean of the vectors.
    5. After concatenating the SVD components, recomputes the eigendecomposition of the summed U and V matrices and constructs the merged layer.

    Args:
        task_vectors (list): A list of task vectors, where each task vector is a dictionary containing the vectors for each task.
        config (object): A configuration object containing the device and dataset information.

    Returns:
        dict: A dictionary containing the new vectors after merging the SVD components.
    """
    # becareful wit vit-l on 20 task it does not fit in GPU or in 64 GB RAM (try without last layer)
    device = config.device
    print("Computing SVD...")
    with torch.no_grad():
        new_vector = {}
        for key in task_vectors[0].vector:
            new_vector[key] = {}
            for i, (task_vector, dataset) in enumerate(
                zip(task_vectors, config.DATASETS)
            ):
                vec = task_vector.vector[key].to(device)

                if (
                    len(task_vector.vector[key].shape) == 2
                    and "text_projection" not in key
                ):

                    u, s, v = torch.linalg.svd(vec, full_matrices=False)

                    if i == 0:
                        print(f"Computed SVD for {key}...")
                        sum_u = torch.zeros(
                            u.shape[0], u.shape[1] * config.num_tasks, device=device
                        )
                        sum_s = torch.zeros(
                            s.shape[0] * config.num_tasks, device=device
                        )
                        sum_v = torch.zeros(
                            v.shape[0] * config.num_tasks, v.shape[1], device=device
                        )
                    reduced_index_s = s.shape[0]

                    # select only the first reduced_index_s columns of u and place them
                    sum_u[:, i * reduced_index_s : (i + 1) * reduced_index_s] = u[
                        :, :reduced_index_s
                    ]
                    sum_s[i * reduced_index_s : (i + 1) * reduced_index_s] = s[
                        :reduced_index_s
                    ]
                    # select only the first reduced_index_s rows of v and place them
                    sum_v[i * reduced_index_s : (i + 1) * reduced_index_s, :] = v[
                        :reduced_index_s, :
                    ]

                else:
                    if i == 0:
                        new_vector[key] = vec.clone()
                    else:
                        new_vector[key] += (vec - new_vector[key]) / (i + 1)

            if len(task_vector.vector[key].shape) == 2 and "text_projection" not in key:
                sum_s, indices = torch.sort(sum_s, stable=True)

                sum_u = torch.index_select(sum_u, 1, indices)
                l_u, q_u = torch.linalg.eigh(sum_u.mT @ sum_u)
                u_orth = (
                    q_u
                    @ torch.diag(1.0 / (torch.sqrt(torch.abs(l_u)) + 1e-12))
                    @ q_u.mT
                )

                sum_v = torch.index_select(sum_v, 0, indices)

                l_v, q_v = torch.linalg.eigh(sum_v @ sum_v.mT)
                v_orth = (
                    q_v
                    @ torch.diag(1.0 / (torch.sqrt(torch.abs(l_v)) + 1e-12))
                    @ q_v.mT
                )

                new_vector[key] = torch.linalg.multi_dot(  # bool_mask *
                    (
                        u_orth,
                        torch.diag(sum_s),
                        v_orth,
                    )
                )

    return new_vector


###############
#### TSV Merge Orthogonalization
def compute_and_sum_svd_mem_reduction(task_vectors, config):
    """
    Computes the Singular Value Decomposition (SVD) for each vector in the task_vectors,
    reduces the dimensionality of the vectors based on the sv_reduction factor, and concatenate
    the low-rank matrices. If the vector is not a 2D tensor or is "text_projection", it computes the mean of the vectors.
    Computation of the SVD is performed also for the second operation.

    Args:
        task_vectors (list): A list of task vector objects, where each object contains a
                             dictionary of vectors.
        config (object): Configuration object containing the following attributes:
                         - DATASETS (list): List of datasets.
                         - device (torch.device): The device to perform computations on.

    Returns:
        dict: A dictionary containing the new vectors after SVD computation and merging.
    """
    sv_reduction = 1 / len(config.DATASETS)
    device = config.device
    print("Computing SVD...")
    with torch.no_grad():
        new_vector = {}
        for key in task_vectors[0].vector:
            new_vector[key] = {}
            for i, (task_vector, dataset) in enumerate(
                zip(task_vectors, config.DATASETS)
            ):
                vec = task_vector.vector[key].to(device)

                if (
                    len(task_vector.vector[key].shape) == 2
                    and "text_projection" not in key
                ):
                    u, s, v = torch.linalg.svd(vec, full_matrices=False)

                    if i == 0:
                        print(f"Computed SVD for {key}...")
                        sum_u = torch.zeros_like(u, device=device)
                        sum_s = torch.zeros_like(s, device=device)
                        sum_v = torch.zeros_like(v, device=device)
                    reduced_index_s = int(s.shape[0] * sv_reduction)

                    # select only the first reduced_index_s columns of u and place them
                    sum_u[:, i * reduced_index_s : (i + 1) * reduced_index_s] = u[
                        :, :reduced_index_s
                    ]
                    sum_s[i * reduced_index_s : (i + 1) * reduced_index_s] = s[
                        :reduced_index_s
                    ]
                    # select only the first reduced_index_s rows of v and place them
                    sum_v[i * reduced_index_s : (i + 1) * reduced_index_s, :] = v[
                        :reduced_index_s, :
                    ]

                else:
                    if i == 0:
                        new_vector[key] = vec.clone()
                    else:
                        new_vector[key] += (vec - new_vector[key]) / (i + 1)

            if len(task_vector.vector[key].shape) == 2 and "text_projection" not in key:
                u_u, s_u, v_u = torch.linalg.svd(sum_u, full_matrices=False)
                u_v, s_v, v_v = torch.linalg.svd(sum_v, full_matrices=False)

                new_vector[key] = torch.linalg.multi_dot(
                    (
                        u_u,
                        v_u,
                        torch.diag(sum_s),
                        u_v,
                        v_v,
                    )
                )

    return new_vector


###############
#### TSV Merge Eigendecomp
def compute_and_sum_svd_mem_reduction_2(task_vectors, config):
    """
    Computes the Singular Value Decomposition (SVD) for each vector in the task_vectors,
    reduces the dimensionality of the vectors based on the sv_reduction factor, and concatenate
    the low-rank matrices. If the vector is not a 2D tensor or is "text_projection", it computes the mean of the vectors.
    Computation of the eigendecomposition is performed instead of the SVD for the second operation.

    Args:
        task_vectors (list): A list of task vector objects, where each object contains a
                             dictionary of vectors.
        config (object): Configuration object containing the following attributes:
                         - DATASETS (list): List of datasets.
                         - device (torch.device): The device to perform computations on.

    Returns:
        dict: A dictionary containing the new vectors after SVD computation and merging.
    """
    sv_reduction = 1 / len(config.DATASETS)
    device = config.device
    print("Computing SVD...")
    with torch.no_grad():
        new_vector = {}
        for key in task_vectors[0].vector:
            new_vector[key] = {}
            for i, (task_vector, dataset) in enumerate(
                zip(task_vectors, config.DATASETS)
            ):
                vec = task_vector.vector[key].to(device)

                if (
                    len(task_vector.vector[key].shape) == 2
                    and "text_projection" not in key
                ):
                    u, s, v = torch.linalg.svd(vec, full_matrices=False)

                    if i == 0:
                        print(f"Computed SVD for {key}...")
                        sum_u = torch.zeros_like(u, device=device)
                        sum_s = torch.zeros_like(s, device=device)
                        sum_v = torch.zeros_like(v, device=device)
                    reduced_index_s = int(s.shape[0] * sv_reduction)

                    # select only the first reduced_index_s columns of u and place them
                    sum_u[:, i * reduced_index_s : (i + 1) * reduced_index_s] = u[
                        :, :reduced_index_s
                    ]
                    sum_s[i * reduced_index_s : (i + 1) * reduced_index_s] = s[
                        :reduced_index_s
                    ]
                    # select only the first reduced_index_s rows of v and place them
                    sum_v[i * reduced_index_s : (i + 1) * reduced_index_s, :] = v[
                        :reduced_index_s, :
                    ]

                else:
                    if i == 0:
                        new_vector[key] = vec.clone()
                    else:
                        new_vector[key] += (vec - new_vector[key]) / (i + 1)

            if len(task_vector.vector[key].shape) == 2 and "text_projection" not in key:
                sum_s, indices = torch.sort(sum_s, stable=True)

                sum_u = torch.index_select(sum_u, 1, indices)
                l_u, q_u = torch.linalg.eigh(sum_u.mT @ sum_u)
                u_orth = (
                    q_u
                    @ torch.diag(1.0 / (torch.sqrt(torch.abs(l_u)) + 1e-12))
                    @ q_u.mT
                )

                sum_v = torch.index_select(sum_v, 0, indices)

                l_v, q_v = torch.linalg.eigh(sum_v @ sum_v.mT)
                v_orth = (
                    q_v
                    @ torch.diag(1.0 / (torch.sqrt(torch.abs(l_v)) + 1e-12))
                    @ q_v.mT
                )

                new_vector[key] = torch.linalg.multi_dot(  # bool_mask *
                    (
                        sum_u,
                        u_orth,
                        torch.diag(sum_s),
                        v_orth,
                        sum_v,
                    )
                )

    return new_vector


###############
#### Rank Reduction TV
def compute_and_sum_svd_mem_reduction_rank_reduction(task_vectors, config):
    """
    Compute and sum the Singular Value Decomposition (SVD) of task vectors with rank reduction.

    This function performs SVD on the vectors in `task_vectors` and reduces their rank based on the
    number of tasks specified in `config.DATASETS`. The reduced vectors are then summed together.

    Args:
        task_vectors (list): A list of task vector objects. Each object should have a `vector` attribute
                             which is a dictionary where keys are vector names and values are tensors.
        config (object): Configuration object containing the following attributes:
                         - DATASETS (list): List of datasets.
                         - device (torch.device): The device to perform computations on.

    Returns:
        dict: A dictionary containing the new vectors after SVD computation and summation.
    """
    sv_reduction = 1 / len(config.DATASETS)
    device = config.device
    print("Computing SVD...")
    with torch.no_grad():
        new_vector = {}
        for key in task_vectors[0].vector:
            new_vector[key] = {}
            for i, (task_vector, dataset) in enumerate(
                zip(task_vectors, config.DATASETS)
            ):
                vec = task_vector.vector[key].to(device)

                if (
                    len(task_vector.vector[key].shape) == 2
                    and "text_projection" not in key
                ):
                    u, s, v = torch.linalg.svd(vec, full_matrices=False)

                    if i == 0:
                        print(f"Computed SVD for {key}...")
                        sum_u = torch.zeros_like(u, device=device)
                        sum_s = torch.zeros_like(s, device=device)
                        sum_v = torch.zeros_like(v, device=device)
                    reduced_index_s = int(s.shape[0] * sv_reduction)

                    # select only the first reduced_index_s columns of u and place them
                    sum_u[:, i * reduced_index_s : (i + 1) * reduced_index_s] = u[
                        :, :reduced_index_s
                    ]
                    sum_s[i * reduced_index_s : (i + 1) * reduced_index_s] = s[
                        :reduced_index_s
                    ]
                    # select only the first reduced_index_s rows of v and place them
                    sum_v[i * reduced_index_s : (i + 1) * reduced_index_s, :] = v[
                        :reduced_index_s, :
                    ]

                else:
                    if i == 0:
                        new_vector[key] = vec.clone()
                    else:
                        new_vector[key] += (vec - new_vector[key]) / (i + 1)

            if len(task_vector.vector[key].shape) == 2 and "text_projection" not in key:
                new_vector[key] = torch.linalg.multi_dot(
                    (
                        sum_u,
                        torch.diag(sum_s),
                        sum_v,
                    )
                )
    return new_vector


def compute_and_sum_svd_mem_reduction_dummy(task_vectors, config):
    """To perform dummy operations."""
    sv_reduction = 1 / 8
    print("Computing SVD...")
    with torch.no_grad():
        new_vector = {}
        for key in task_vectors[0].vector:
            new_vector[key] = {}
            for i in range(0, 8):
                if (
                    len(task_vectors[0].vector[key].shape) == 2
                    and "text_projection" not in key
                ):
                    if i == 0:
                        u, s, v = torch.linalg.svd(
                            task_vectors[0].vector[key], full_matrices=False
                        )
                        reduced_index_s = int(s.shape[0] * sv_reduction)

                        print(f"Computed SVD for {key}...")
                        sum_u = torch.zeros_like(u)
                        sum_s = torch.zeros_like(s)
                        sum_v = torch.zeros_like(v)

                        # select only the first reduced_index_s columns of u and place them
                        sum_u[:, i * reduced_index_s : (i + 1) * reduced_index_s] = u[
                            :, :reduced_index_s
                        ]
                        sum_s[i * reduced_index_s : (i + 1) * reduced_index_s] = s[
                            :reduced_index_s
                        ]
                        # select only the first reduced_index_s rows of v and place them
                        sum_v[i * reduced_index_s : (i + 1) * reduced_index_s, :] = v[
                            :reduced_index_s, :
                        ]
                    else:
                        # generate u vectors orthogonal to the previous ones
                        # generate v vectors orthogonal to the previous ones
                        print("dummy")
                        u = torch.nn.functional.normalize(
                            torch.randn_like(sum_u), p=2, dim=-2
                        )
                        v = torch.nn.functional.normalize(
                            torch.randn_like(sum_v), p=2, dim=-1
                        )

                        # select only the first reduced_index_s columns of u and place them
                        sum_u[:, i * reduced_index_s : (i + 1) * reduced_index_s] = u[
                            :, :reduced_index_s
                        ]
                        sum_s[i * reduced_index_s : (i + 1) * reduced_index_s] = s[
                            :reduced_index_s
                        ]
                        # select only the first reduced_index_s rows of v and place them
                        sum_v[i * reduced_index_s : (i + 1) * reduced_index_s, :] = v[
                            :reduced_index_s, :
                        ]

                else:
                    if i == 0:
                        new_vector[key] = task_vectors[0].vector[key]
                    # else:
                    #     new_vector[key] += (
                    #         task_vector.vector[key] - new_vector[key]
                    #     ) / (i + 1)

            if (
                len(task_vectors[0].vector[key].shape) == 2
                and "text_projection" not in key
            ):

                new_vector[key] = torch.linalg.multi_dot(
                    (
                        sum_u,
                        torch.diag(sum_s),
                        sum_v,
                    )
                )

    return new_vector
