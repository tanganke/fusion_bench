import math
from typing import List

import torch

from fusion_bench.utils import timeit_context
from fusion_bench.utils.type import StateDictType


def iso_c(
    task_vectors: List[StateDictType],
    accelerator="cuda",
    exclude_keys: List[str] = None,
) -> StateDictType:
    exclude_keys = [] if exclude_keys is None else exclude_keys

    with torch.no_grad(), timeit_context("ISO-C Merging"):
        new_vector = {}
        for key in task_vectors[0]:
            print(f"Merging {key}...")
            original_device = task_vectors[0][key].device
            tvs = [
                task_vector[key].to(device=accelerator, non_blocking=True)
                for task_vector in task_vectors
            ]
            num_tvs = len(tvs)
            new_vector[key] = sum(tvs) / num_tvs
            del tvs  # free memory

            if len(task_vectors[0][key].shape) == 2 and key not in exclude_keys:
                # if the key is a 2D matrix, we need to merge the task vectors in the common space
                new_vector[key] *= num_tvs
                U, S, V = torch.linalg.svd(new_vector[key], full_matrices=False)
                S_mean = torch.ones_like(S) * S.mean()

                new_vector[key] = torch.linalg.multi_dot(
                    (
                        U,
                        torch.diag(S_mean),
                        V,
                    )
                )
            new_vector[key] = new_vector[key].to(
                device=original_device, non_blocking=True
            )
    return new_vector


@torch.no_grad()
def iso_cts(
    task_vectors: List[StateDictType],
    common_space_fraction: float,
    accelerator: str = "cuda",
    exclude_keys: List[str] = None,
):
    exclude_keys = [] if exclude_keys is None else exclude_keys
    new_vector = {}

    print("ISO-CTS Merging")
    for key in task_vectors[0]:
        shape_ = task_vectors[0][key].shape
        original_device = task_vectors[0][key].device
        is_2d_matrix = (len(shape_) == 2) and (key not in exclude_keys)
        if not is_2d_matrix:
            print(f"Combining by avg {key}...")
            for i, task_vector in enumerate(task_vectors):
                vec = task_vector[key].to(device=accelerator, non_blocking=True)
                if i == 0:
                    new_vector[key] = vec.clone()
                else:
                    new_vector[key] += (vec - new_vector[key]) / (i + 1)

            # move the new vector to the original device
            new_vector[key] = new_vector[key].to(
                device=original_device, non_blocking=True
            )
            continue

        print(f"Computing common space using sum for {key}...")
        combined_w = sum(
            [
                task_vector[key].to(device=accelerator, non_blocking=True)
                for task_vector in task_vectors
            ]
        )

        ### Calculate the common space size (making sure that task specific space is equally divisible) ###
        common_space_index_s = int(min(shape_) * common_space_fraction)
        _task_specific_total_space_index_s = round(
            (min(shape_) - common_space_index_s) / len(task_vectors)
        ) * len(task_vectors)
        common_space_index_s = min(shape_) - _task_specific_total_space_index_s

        u, s, v = torch.linalg.svd(combined_w, full_matrices=False)
        common_space_u = u[:, :common_space_index_s]
        common_space_s = s[:common_space_index_s]
        common_space_v = v[:common_space_index_s, :]
        ###################################################################

        ### Calculate task specific space ###
        n_dims_per_task = int((min(shape_) - common_space_index_s) / len(task_vectors))
        for i, task_vector in enumerate(task_vectors):
            w = task_vector[key].to(device=accelerator)

            # calculate the projection onto task specific space to remove the common space
            w_ts = w - common_space_u @ common_space_u.T @ w
            u_ts, s_ts, v_ts = torch.linalg.svd(w_ts, full_matrices=False)

            if i == 0:
                combined_space_u = torch.zeros_like(u_ts, device=accelerator)
                combined_space_s = torch.zeros_like(s_ts, device=accelerator)
                combined_space_v = torch.zeros_like(v_ts, device=accelerator)

            combined_space_u[:, i * n_dims_per_task : (i + 1) * n_dims_per_task] = u_ts[
                :, :n_dims_per_task
            ]
            combined_space_s[i * n_dims_per_task : (i + 1) * n_dims_per_task] = s_ts[
                :n_dims_per_task
            ]
            combined_space_v[i * n_dims_per_task : (i + 1) * n_dims_per_task, :] = v_ts[
                :n_dims_per_task, :
            ]
        ###################################################################

        combined_space_u[
            :,
            len(task_vectors) * n_dims_per_task : len(task_vectors) * n_dims_per_task
            + common_space_index_s,
        ] = common_space_u
        combined_space_s[
            len(task_vectors) * n_dims_per_task : len(task_vectors) * n_dims_per_task
            + common_space_index_s
        ] = common_space_s
        combined_space_v[
            len(task_vectors) * n_dims_per_task : len(task_vectors) * n_dims_per_task
            + common_space_index_s,
            :,
        ] = common_space_v

        ### Orthogonalize combined_space_u and combined_space_v ###
        u_combined_space_u, s_combined_space_u, v_combined_space_u = torch.linalg.svd(
            combined_space_u, full_matrices=False
        )
        u_combined_space_v, s_combined_space_v, v_combined_space_v = torch.linalg.svd(
            combined_space_v, full_matrices=False
        )
        combined_space_u = u_combined_space_u @ v_combined_space_u
        combined_space_v = u_combined_space_v @ v_combined_space_v
        ###################################################################

        combined_space_s = torch.ones_like(combined_space_s) * combined_space_s.mean()

        new_vector[key] = torch.linalg.multi_dot(
            (
                combined_space_u,
                torch.diag(combined_space_s),
                combined_space_v,
            )
        )
        new_vector[key] = new_vector[key].to(device=original_device, non_blocking=True)

    return new_vector


def check_parameterNamesMatch(checkpoints):
    parameter_names = set(checkpoints[0].keys())

    if len(checkpoints) >= 2:
        # raise ValueError("Number of models is less than 2.")
        for checkpoint in checkpoints[1:]:
            current_parameterNames = set(checkpoint.keys())
            if current_parameterNames != parameter_names:
                raise ValueError(
                    "Differing parameter names in models. "
                    f"The different parameters are {parameter_names.symmetric_difference(current_parameterNames)}"
                )
