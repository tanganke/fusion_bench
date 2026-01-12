from typing import Iterator, List, Mapping, Optional, Tuple, Union

import torch
from torch import nn

__all__ = ["ParameterDictModel"]


def set_nested_attr(
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
        set_nested_attr(
            getattr(obj, names[0]),
            names[1:],
            val,
            check_parent=check_parent,
            parent_builder=parent_builder,
        )


def has_nested_attr(obj, names: List[str]):
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
        if not hasattr(obj, names[0]):
            return False
        return has_nested_attr(getattr(obj, names[0]), names[1:])


class ParameterDictModel(nn.Module):
    """
    A module that stores parameters in a nested dictionary structure.

    This model behaves similarly to `nn.ParameterDict`, but supports hierarchical keys
    with dots (e.g., "layer1.weight"). Parameters are stored as nested attributes,
    allowing for structured parameter access and manipulation.

    Example:
        >>> params = {
        ...     "encoder.weight": nn.Parameter(torch.randn(10, 5)),
        ...     "decoder.bias": nn.Parameter(torch.randn(5)),
        ... }
        >>> model = ParameterDictModel(params)
        >>> model["encoder.weight"].shape
        torch.Size([10, 5])
        >>> "encoder.weight" in model
        True
    """

    def __init__(
        self,
        parameters: Optional[Mapping[str, Union[nn.Parameter, torch.Tensor]]] = None,
    ) -> None:
        """
        Args:
            parameters: Optional mapping of parameter names to parameter tensors.
                Keys can contain dots to create nested structures.
                Values must be `nn.Parameter` or `nn.Buffer` instances.
        """

        super().__init__()
        if parameters is not None:
            for name, param in parameters.items():
                assert isinstance(
                    param, (nn.Parameter, nn.Buffer)
                ), f"{name} is not a nn.Parameter or nn.Buffer"
                set_nested_attr(
                    self,
                    name.split("."),
                    param,
                    check_parent=True,
                    parent_builder=__class__,
                )

    def __repr__(self) -> str:
        """
        Generate a string representation of the model's parameters.

        Returns:
            A string representation of the model's parameters in the format:
            "ParameterDictModel(name1: shape1, name2: shape2, ...)"
        """
        param_reprs = []
        for name, param in self.named_parameters():
            param_repr = f"{name}: {param.size()}"
            param_reprs.append(param_repr)
        return f"{self.__class__.__name__}({', '.join(param_reprs)})"

    def __iter__(self) -> Iterator[str]:
        """
        Iterate over the model's parameters.

        Yields:
            Tuples of (parameter name, parameter tensor).
        """
        yield from self.keys()

    def __getitem__(
        self, key: str
    ) -> Union[nn.Parameter, torch.Tensor, "ParameterDictModel"]:
        """
        Retrieve a parameter or nested submodule by key.

        Args:
            key: Parameter name, which can contain dots for nested access.

        Returns:
            The parameter, tensor, or nested ParameterDictModel at the specified key.

        Raises:
            KeyError: If the key is not found in the model.
        """
        assert isinstance(
            key, str
        ), f"Key must be a string, but got {type(key)}: {key}."
        if not has_nested_attr(self, key.split(".")):
            raise KeyError(f"Key {key} not found in {self}")
        key_parts = key.split(".")
        obj = self
        for k in key_parts:
            obj = getattr(obj, k)
        return obj

    def __setitem__(self, key: str, value: Union[nn.Parameter, torch.Tensor]) -> None:
        """
        Set a parameter at the specified key, creating nested structure if needed.

        Args:
            key: Parameter name, which can contain dots for nested assignment.
            value: Parameter or tensor to assign.
        """
        if not has_nested_attr(self, key.split(".")):
            set_nested_attr(self, key.split("."), value, check_parent=True)
        else:
            set_nested_attr(self, key.split("."), value, check_parent=False)

    def __contains__(self, key: str) -> bool:
        """
        Check if a parameter key exists in the model.

        Args:
            key: Parameter name, which can contain dots for nested checking.

        Returns:
            True if the key exists, False otherwise.
        """
        return has_nested_attr(self, key.split("."))

    def keys(self):
        """
        Return a list of all parameter names in the model.

        Returns:
            List of parameter names (including nested names with dots).
        """
        return self.state_dict().keys()

    def items(self):
        """
        Return a list of (name, parameter) tuples.

        Returns:
            List of tuples containing parameter names and their corresponding tensors.
        """
        yield from self.state_dict().items()

    def values(self):
        """
        Return a list of all parameter values in the model.

        Returns:
            List of parameter tensors.
        """
        yield from self.state_dict().values()

    def __len__(self) -> int:
        """
        Return the number of parameters in the model.

        Returns:
            The total number of parameters.
        """
        return len(self.keys())
