import copy
from collections import OrderedDict

from torch import nn

# Model conversion utils


def state_dict_to_vector(state_dict, remove_keys=[]):
    """
    Convert a state dictionary to a vector.

    Args:
        state_dict (dict): The state dictionary to convert.
        remove_keys (list, optional): List of keys to remove from the state dictionary. Defaults to [].

    Returns:
        torch.Tensor: The converted vector.
    """
    shared_state_dict = copy.deepcopy(state_dict)
    for key in remove_keys:
        if key in shared_state_dict:
            del shared_state_dict[key]
    sorted_shared_state_dict = OrderedDict(sorted(shared_state_dict.items()))
    return nn.utils.parameters_to_vector(
        [value.reshape(-1) for key, value in sorted_shared_state_dict.items()]
    )


def vector_to_state_dict(vector, state_dict, remove_keys=[]):
    """
    Convert a vector to a state dictionary.

    Args:
        vector (torch.Tensor): The vector to convert.
        state_dict (dict): The reference state dictionary to define the order of the vector.
        remove_keys (list, optional): List of keys to remove from the reference state dictionary. Defaults to [].

    Returns:
        dict: The converted state dictionary.
    """
    # create a reference dict to define the order of the vector
    reference_dict = copy.deepcopy(state_dict)
    for key in remove_keys:
        if key in reference_dict:
            del reference_dict[key]
    sorted_reference_dict = OrderedDict(sorted(reference_dict.items()))

    # create a shared state dict using the reference dict
    nn.utils.vector_to_parameters(vector, sorted_reference_dict.values())

    # add back the encoder and decoder embedding weights.
    if "transformer.shared.weight" in sorted_reference_dict:
        for key in remove_keys:
            sorted_reference_dict[key] = sorted_reference_dict[
                "transformer.shared.weight"
            ]
    return sorted_reference_dict
