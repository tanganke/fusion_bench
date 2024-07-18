from typing import List, Mapping

import torch
from torch import nn

__all__ = "ParamterDictModel"


def set_attr(obj, names: List[str], val, check_parent: bool = False):
    """
    Sets an attribute of an object recursively.

    Args:
        obj (object): Object to set attribute of.
        names (list): List of attribute names to set recursively.
        val (object): Value to set the attribute to.
        check_parent (bool): If True, checks if the parent attribute exists; otherwise, creates it if it does not exist.
    """
    if len(names) == 1:
        setattr(obj, names[0], val)
    else:
        if check_parent and not hasattr(obj, names[0]):
            setattr(obj, names[0], nn.Module())
        set_attr(getattr(obj, names[0]), names[1:], val, check_parent=check_parent)


def has_attr(obj, names: List[str]):
    """
    Checks if an attribute exists in an object recursively.

    Args:
        obj (object): Object to check attribute of.
        names (list): List of attribute names to check recursively.

    Returns:
        bool: True if the attribute exists; otherwise, False.
    """
    if len(names) == 1:
        return hasattr(obj, names[0])
    else:
        return has_attr(getattr(obj, names[0]), names[1:])


class ParameterDictModel(nn.Module):
    """
    This model is used to create a model with parameters from a dictionary.
    It behaves like a normal `nn.ParameterDict`, but support keys with dots.
    """

    def __init__(
        self,
        parameters: Mapping[str, nn.Parameter],
    ):
        super().__init__()
        for name, param in parameters.items():
            assert isinstance(param, nn.Parameter), f"{name} is not a nn.Parameter"
            set_attr(
                self,
                name.split("."),
                param,
                check_parent=True,
            )

    def __repr__(self):
        """
        Generate a string representation of the model's parameters.

        Returns:
            str: A string representation of the model's parameters.
        """
        param_reprs = []
        for name, param in self.named_parameters():
            param_repr = f"{name}: {param.size()}"
            param_reprs.append(param_repr)
        return f"{self.__class__.__name__}({', '.join(param_reprs)})"
