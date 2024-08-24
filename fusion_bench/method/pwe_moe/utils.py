import itertools

import numpy as np


def generate_simplex_grid(n: int, m: int):
    """
    Generate a uniform grid of points on the n-dimensional simplex.

    Examples:

        >>> generate_simplex_grid(3,2)
        array([[0., 0., 1.],
              [0., 1., 0.],
              [1., 0., 0.]], dtype=float32)

        >>> generate_simplex_grid(2,3)
        array([[0. , 1. ],
              [0.5, 0.5],
              [1. , 0. ]], dtype=float32)

    Args:
        n (int): The dimension of the simplex.
        m (int): The number of grid points along each dimension.

    Returns:
        list: A list of n-dimensional vectors representing the grid points.
    """
    m = m - 1
    # Generate all combinations of indices summing up to m
    indices = list(itertools.combinations_with_replacement(range(m + 1), n - 1))
    # Initialize an empty list to store the grid points
    grid_points = []

    # Iterate over each combination of indices
    for idx in indices:
        # Append 0 and m to the indices
        extended_idx = [0] + list(idx) + [m]
        # Compute the vector components by taking the differences between consecutive indices and dividing by m
        point = [(extended_idx[i + 1] - extended_idx[i]) / m for i in range(n)]
        grid_points.append(point)

    return np.array(grid_points, dtype=np.float32)
