from typing import List, Mapping, Optional, Tuple

import torch
from torch import nn

__all__ = "ParamterDictModel"


def _set_attr(
    obj,
    names: List[str],
    val,
    check_parent: bool = False,
    parent_builder=nn.Module,
):
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
            setattr(obj, names[0], parent_builder())
        _set_attr(
            getattr(obj, names[0]),
            names[1:],
            val,
            check_parent=check_parent,
            parent_builder=parent_builder,
        )


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
        parameters: Optional[Mapping[str, nn.Parameter]] = None,
    ):
        super().__init__()
        if parameters is not None:
            for name, param in parameters.items():
                assert isinstance(param, nn.Parameter), f"{name} is not a nn.Parameter"
                _set_attr(
                    self,
                    name.split("."),
                    param,
                    check_parent=True,
                    parent_builder=self.__class__,
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

    def __getitem__(self, key: str):
        if not has_attr(self, key.split(".")):
            raise KeyError(f"Key {key} not found in {self}")
        key = key.split(".")
        obj = self
        for k in key:
            obj = getattr(obj, k)
        return obj

    def __setitem__(self, key: str, value: nn.Parameter):
        if not has_attr(self, key.split(".")):
            _set_attr(self, key.split("."), value, check_parent=True)
        else:
            _set_attr(self, key.split("."), value, check_parent=False)

    def __contains__(self, key: str):
        return has_attr(self, key.split("."))

    def keys(self):
        return [name for name, _ in self.named_parameters()]

    def items(self) -> List[Tuple[str, nn.Parameter]]:
        return [(name, self[name]) for name in self.keys()]

    def values(self) -> List[nn.Parameter]:
        return [self[name] for name in self.keys()]
