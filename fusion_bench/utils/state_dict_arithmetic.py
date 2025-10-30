from collections import OrderedDict
from numbers import Number
from typing import Callable, Dict, List, Literal, Optional, Union, cast

import torch
from torch import Tensor
from tqdm.auto import tqdm

from fusion_bench.utils.type import TorchModelType

from .type import BoolStateDictType, StateDictType

__all__ = [
    "ArithmeticStateDict",
    "load_state_dict_with_prefix",
    "state_dicts_check_keys",
    "state_dict_to_device",
    "num_params_of_state_dict",
    "state_dict_flatten",
    "state_dict_avg",
    "state_dict_sub",
    "state_dict_add",
    "state_dict_add_scalar",
    "state_dict_mul",
    "state_dict_div",
    "state_dict_power",
    "state_dict_interpolation",
    "state_dict_sum",
    "state_dict_weighted_sum",
    "state_dict_diff_abs",
    "state_dict_binary_mask",
    "state_dict_hadamard_product",
]


class ArithmeticStateDict(OrderedDict):
    """
    An OrderedDict subclass that supports arithmetic operations on state dictionaries.

    This class provides convenient operator overloading for common state dict operations
    like addition, subtraction, multiplication, and division, while maintaining all
    the functionality of OrderedDict.

    Examples:
        >>> sd1 = ArithmeticStateDict({'weight': torch.tensor([1.0, 2.0]), 'bias': torch.tensor([0.5])})
        >>> sd2 = ArithmeticStateDict({'weight': torch.tensor([2.0, 3.0]), 'bias': torch.tensor([1.0])})
        >>> result = sd1 + sd2  # Element-wise addition
        >>> result = sd1 - sd2  # Element-wise subtraction
        >>> result = sd1 * 2.0  # Scalar multiplication
        >>> result = sd1 / 2.0  # Scalar division
        >>> result = sd1 @ sd2  # Hadamard product
    """

    def __init__(self, *args, **kwargs):
        """Initialize ArithmeticStateDict with the same interface as OrderedDict."""
        super().__init__(*args, **kwargs)

    def __add__(
        self, other: Union["ArithmeticStateDict", StateDictType, Number]
    ) -> "ArithmeticStateDict":
        """
        Element-wise addition with another state dict or scalar.

        Args:
            other: Another state dict to add or a scalar to add to all elements.

        Returns:
            A new ArithmeticStateDict with the element-wise sum.
        """
        if isinstance(other, (int, float, Number)):
            # Scalar addition
            result_dict = state_dict_add_scalar(self, other)
            return ArithmeticStateDict(result_dict)
        elif isinstance(other, (dict, OrderedDict)):
            # State dict addition
            result_dict = state_dict_add(self, other, strict=True)
            return ArithmeticStateDict(result_dict)
        else:
            raise TypeError(
                f"Cannot add ArithmeticStateDict with {type(other).__name__}"
            )

    def __radd__(
        self, other: Union["ArithmeticStateDict", StateDictType, Number]
    ) -> "ArithmeticStateDict":
        """
        Right addition (other + self).
        Handles the case where sum() starts with 0 and scalar addition.
        """
        if other == 0:  # sum() starts with 0 by default
            return self
        elif isinstance(other, (int, float, Number)):
            # Scalar addition is commutative
            return self.__add__(other)
        elif isinstance(other, (dict, OrderedDict)):
            return self.__add__(other)
        else:
            raise TypeError(
                f"Cannot add {type(other).__name__} with ArithmeticStateDict"
            )

    def __sub__(
        self, other: Union["ArithmeticStateDict", StateDictType, Number]
    ) -> "ArithmeticStateDict":
        """
        Element-wise subtraction with another state dict or scalar.

        Args:
            other: Another state dict to subtract or a scalar to subtract from all elements.

        Returns:
            A new ArithmeticStateDict with the element-wise difference.
        """
        if isinstance(other, (int, float, Number)):
            # Scalar subtraction: subtract scalar from all elements
            result_dict = state_dict_add_scalar(self, -other)
            return ArithmeticStateDict(result_dict)
        elif isinstance(other, (dict, OrderedDict)):
            # State dict subtraction
            result_dict = state_dict_sub(self, other, strict=True)
            return ArithmeticStateDict(result_dict)
        else:
            raise TypeError(
                f"Cannot subtract {type(other).__name__} from ArithmeticStateDict"
            )

    def __rsub__(
        self, other: Union["ArithmeticStateDict", StateDictType, Number]
    ) -> "ArithmeticStateDict":
        """Right subtraction (other - self)."""
        if isinstance(other, (int, float, Number)):
            # Scalar - ArithmeticStateDict: subtract each element from scalar
            result = ArithmeticStateDict()
            for key, tensor in self.items():
                result[key] = other - tensor
            return result
        elif isinstance(other, (dict, OrderedDict)):
            result_dict = state_dict_sub(other, self, strict=True)
            return ArithmeticStateDict(result_dict)
        else:
            raise TypeError(
                f"Cannot subtract ArithmeticStateDict from {type(other).__name__}"
            )

    def __mul__(
        self, scalar: Union[Number, "ArithmeticStateDict", StateDictType]
    ) -> "ArithmeticStateDict":
        """
        Scalar multiplication or Hadamard product.

        Args:
            scalar: A scalar value for element-wise multiplication, or another state dict
                   for Hadamard product.

        Returns:
            A new ArithmeticStateDict with the result.
        """
        if isinstance(scalar, (int, float, Number)):
            result_dict = state_dict_mul(self, scalar)
            return ArithmeticStateDict(result_dict)
        elif isinstance(scalar, (dict, OrderedDict)):
            # Hadamard product for dict-like objects
            result_dict = state_dict_hadamard_product(self, scalar)
            return ArithmeticStateDict(result_dict)
        else:
            raise TypeError(
                f"Cannot multiply ArithmeticStateDict with {type(scalar).__name__}"
            )

    def __rmul__(
        self, scalar: Union[Number, "ArithmeticStateDict", StateDictType]
    ) -> "ArithmeticStateDict":
        """Right multiplication (scalar * self)."""
        return self.__mul__(scalar)

    def __truediv__(self, scalar: Number) -> "ArithmeticStateDict":
        """
        Scalar division.

        Args:
            scalar: A scalar value to divide by.

        Returns:
            A new ArithmeticStateDict with each element divided by scalar.

        Raises:
            ZeroDivisionError: If scalar is zero.
            TypeError: If scalar is not a number.
        """
        if not isinstance(scalar, (int, float, Number)):
            raise TypeError(
                f"Cannot divide ArithmeticStateDict by {type(scalar).__name__}"
            )

        result_dict = state_dict_div(self, scalar)
        return ArithmeticStateDict(result_dict)

    def __pow__(self, exponent: Number) -> "ArithmeticStateDict":
        """
        Element-wise power operation.

        Args:
            exponent: The exponent to raise each element to.

        Returns:
            A new ArithmeticStateDict with each element raised to the power.
        """
        if not isinstance(exponent, (int, float, Number)):
            raise TypeError(
                f"Cannot raise ArithmeticStateDict to power of {type(exponent).__name__}"
            )

        result_dict = state_dict_power(self, exponent)
        return ArithmeticStateDict(result_dict)

    def __matmul__(
        self, other: Union["ArithmeticStateDict", StateDictType]
    ) -> "ArithmeticStateDict":
        """
        Hadamard product (element-wise multiplication) using @ operator.

        Args:
            other: Another state dict for element-wise multiplication.

        Returns:
            A new ArithmeticStateDict with the Hadamard product.
        """
        if not isinstance(other, (dict, OrderedDict)):
            raise TypeError(
                f"Cannot compute Hadamard product with {type(other).__name__}"
            )

        result_dict = state_dict_hadamard_product(self, other)
        return ArithmeticStateDict(result_dict)

    def __rmatmul__(
        self, other: Union["ArithmeticStateDict", StateDictType]
    ) -> "ArithmeticStateDict":
        """Right matrix multiplication (other @ self)."""
        return self.__matmul__(other)

    def __iadd__(
        self, other: Union["ArithmeticStateDict", StateDictType, Number]
    ) -> "ArithmeticStateDict":
        """In-place addition."""
        if isinstance(other, (int, float, Number)):
            # Scalar addition
            for key in self:
                self[key] = self[key] + other
        elif isinstance(other, (dict, OrderedDict)):
            # State dict addition
            for key in self:
                if key in other:
                    self[key] = self[key] + other[key]
        else:
            raise TypeError(f"Cannot add {type(other).__name__} to ArithmeticStateDict")
        return self

    def __isub__(
        self, other: Union["ArithmeticStateDict", StateDictType, Number]
    ) -> "ArithmeticStateDict":
        """In-place subtraction."""
        if isinstance(other, (int, float, Number)):
            # Scalar subtraction
            for key in self:
                self[key] = self[key] - other
        elif isinstance(other, (dict, OrderedDict)):
            # State dict subtraction
            for key in self:
                if key in other:
                    self[key] = self[key] - other[key]
        else:
            raise TypeError(
                f"Cannot subtract {type(other).__name__} from ArithmeticStateDict"
            )
        return self

    def __imul__(
        self, scalar: Union[Number, "ArithmeticStateDict", StateDictType]
    ) -> "ArithmeticStateDict":
        """In-place multiplication."""
        if isinstance(scalar, (int, float, Number)):
            for key in self:
                self[key] = self[key] * scalar
        elif isinstance(scalar, (dict, OrderedDict)):
            for key in self:
                if key in scalar:
                    self[key] = self[key] * scalar[key]
        else:
            raise TypeError(
                f"Cannot multiply ArithmeticStateDict with {type(scalar).__name__}"
            )
        return self

    def __itruediv__(self, scalar: Number) -> "ArithmeticStateDict":
        """In-place division."""
        if not isinstance(scalar, (int, float, Number)):
            raise TypeError(
                f"Cannot divide ArithmeticStateDict by {type(scalar).__name__}"
            )
        if scalar == 0:
            raise ZeroDivisionError("Cannot divide by zero")

        for key in self:
            self[key] = self[key] / scalar
        return self

    def __ipow__(self, exponent: Number) -> "ArithmeticStateDict":
        """In-place power operation."""
        if not isinstance(exponent, (int, float, Number)):
            raise TypeError(
                f"Cannot raise ArithmeticStateDict to power of {type(exponent).__name__}"
            )

        for key in self:
            self[key] = self[key] ** exponent
        return self

    def abs(self) -> "ArithmeticStateDict":
        """
        Element-wise absolute value.

        Returns:
            A new ArithmeticStateDict with absolute values.
        """
        result = ArithmeticStateDict()
        for key, tensor in self.items():
            result[key] = torch.abs(tensor)
        return result

    def sqrt(self) -> "ArithmeticStateDict":
        """
        Element-wise square root.

        Returns:
            A new ArithmeticStateDict with square roots.
        """
        result = ArithmeticStateDict()
        for key, tensor in self.items():
            result[key] = torch.sqrt(tensor)
        return result

    def sum(self) -> "ArithmeticStateDict":
        """
        Sum with other ArithmeticStateDicts using the + operator.

        Args:
            *others: Other ArithmeticStateDicts to sum with.

        Returns:
            A new ArithmeticStateDict with the sum.
        """
        # This is used for when sum() is called on a list of ArithmeticStateDicts
        return self

    def to_device(
        self,
        device: Union[torch.device, str],
        copy: bool = False,
        inplace: bool = False,
    ) -> "ArithmeticStateDict":
        """
        Move all tensors to the specified device.

        Args:
            device: Target device.
            copy: Whether to force a copy.
            inplace: Whether to modify in place.

        Returns:
            ArithmeticStateDict with tensors on the target device.
        """
        if inplace:
            for key, tensor in self.items():
                self[key] = tensor.to(device, non_blocking=True, copy=copy)
            return self
        else:
            result = ArithmeticStateDict()
            for key, tensor in self.items():
                result[key] = tensor.to(device, non_blocking=True, copy=copy)
            return result

    def clone(self) -> "ArithmeticStateDict":
        """
        Create a deep copy with cloned tensors.

        Returns:
            A new ArithmeticStateDict with cloned tensors.
        """
        result = ArithmeticStateDict()
        for key, tensor in self.items():
            result[key] = tensor.clone()
        return result

    def detach(self) -> "ArithmeticStateDict":
        """
        Detach all tensors from the computation graph.

        Returns:
            A new ArithmeticStateDict with detached tensors.
        """
        result = ArithmeticStateDict()
        for key, tensor in self.items():
            result[key] = tensor.detach()
        return result

    def num_params(self) -> int:
        """
        Calculate the total number of parameters.

        Returns:
            Total number of parameters in all tensors.
        """
        return sum(tensor.numel() for tensor in self.values())

    @classmethod
    def from_state_dict(cls, state_dict: StateDictType) -> "ArithmeticStateDict":
        """
        Create an ArithmeticStateDict from a regular state dict.

        Args:
            state_dict: A regular state dictionary.

        Returns:
            A new ArithmeticStateDict with the same data.
        """
        return cls(state_dict)

    @classmethod
    def weighted_sum(
        cls,
        state_dicts: List[Union["ArithmeticStateDict", StateDictType]],
        weights: List[float],
    ) -> "ArithmeticStateDict":
        """
        Compute a weighted sum of multiple state dicts.

        Args:
            state_dicts: List of state dicts to combine.
            weights: List of weights for the combination.

        Returns:
            A new ArithmeticStateDict with the weighted sum.
        """
        result_dict = state_dict_weighted_sum(state_dicts, weights)
        return cls(result_dict)

    @classmethod
    def average(
        cls, state_dicts: List[Union["ArithmeticStateDict", StateDictType]]
    ) -> "ArithmeticStateDict":
        """
        Compute the average of multiple state dicts.

        Args:
            state_dicts: List of state dicts to average.

        Returns:
            A new ArithmeticStateDict with the average.
        """
        result_dict = state_dict_avg(state_dicts)
        return cls(result_dict)


def _validate_state_dict_list_not_empty(state_dicts: List[StateDictType]) -> None:
    """
    Validate that the list of state dicts is not empty and contains valid state dicts.

    Args:
        state_dicts: List of state dictionaries to validate.

    Raises:
        TypeError: If state_dicts is not a list or contains non-dict items.
        ValueError: If the list is empty or contains empty state dicts.
    """
    if state_dicts is None:
        raise TypeError("state_dicts cannot be None")

    if not isinstance(state_dicts, (list, tuple)):
        raise TypeError(
            f"Expected list or tuple of state dicts, got {type(state_dicts).__name__}"
        )

    if not state_dicts:
        raise ValueError("The list of state_dicts must not be empty")

    for i, state_dict in enumerate(state_dicts):
        if state_dict is None:
            raise ValueError(f"State dict at index {i} is None")
        if not isinstance(state_dict, (dict, OrderedDict)):
            raise TypeError(
                f"Item at index {i} is not a dictionary, got {type(state_dict).__name__}"
            )
        if not state_dict:
            raise ValueError(f"State dict at index {i} is empty")


def _validate_state_dict_same_keys(state_dicts: List[StateDictType]) -> None:
    """
    Validate that all state dicts have the same keys and compatible tensor shapes.

    Args:
        state_dicts: List of state dictionaries to validate.

    Raises:
        ValueError: If state dicts have different keys or incompatible tensor shapes.
        TypeError: If tensors have incompatible types.
    """
    if not state_dicts:
        return

    if len(state_dicts) < 2:
        return

    reference_state_dict = state_dicts[0]
    reference_keys = set(reference_state_dict.keys())

    if not reference_keys:
        raise ValueError("Reference state dict (index 0) has no keys")

    for i, state_dict in enumerate(state_dicts[1:], 1):
        current_keys = set(state_dict.keys())

        # Check for missing keys
        missing_keys = reference_keys - current_keys
        if missing_keys:
            raise ValueError(
                f"State dict at index {i} is missing keys: {sorted(missing_keys)}"
            )

        # Check for extra keys
        extra_keys = current_keys - reference_keys
        if extra_keys:
            raise ValueError(
                f"State dict at index {i} has extra keys: {sorted(extra_keys)}"
            )

        # Check tensor shapes and dtypes for compatibility
        for key in reference_keys:
            ref_tensor = reference_state_dict[key]
            curr_tensor = state_dict[key]

            # Handle None values
            if ref_tensor is None and curr_tensor is None:
                continue
            if ref_tensor is None or curr_tensor is None:
                raise ValueError(
                    f"Tensor None mismatch for key '{key}' at index {i}: "
                    f"one is None, the other is not"
                )

            if not isinstance(curr_tensor, type(ref_tensor)):
                raise TypeError(
                    f"Tensor type mismatch for key '{key}' at index {i}: "
                    f"expected {type(ref_tensor).__name__}, got {type(curr_tensor).__name__}"
                )

            if hasattr(ref_tensor, "shape") and hasattr(curr_tensor, "shape"):
                if ref_tensor.shape != curr_tensor.shape:
                    raise ValueError(
                        f"Shape mismatch for key '{key}' at index {i}: "
                        f"expected {ref_tensor.shape}, got {curr_tensor.shape}"
                    )

                if hasattr(ref_tensor, "dtype") and hasattr(curr_tensor, "dtype"):
                    if ref_tensor.dtype != curr_tensor.dtype:
                        raise ValueError(
                            f"Dtype mismatch for key '{key}' at index {i}: "
                            f"expected {ref_tensor.dtype}, got {curr_tensor.dtype}"
                        )

                # Check device compatibility (warn but don't fail)
                if (
                    hasattr(ref_tensor, "device")
                    and hasattr(curr_tensor, "device")
                    and ref_tensor.device != curr_tensor.device
                ):
                    import warnings

                    warnings.warn(
                        f"Device mismatch for key '{key}' at index {i}: "
                        f"reference on {ref_tensor.device}, current on {curr_tensor.device}. "
                        f"This may cause issues during arithmetic operations."
                    )


def _validate_list_lengths_equal(
    list1: List,
    list2: List,
    name1: str = "the first list",
    name2: str = "the second list",
) -> None:
    """
    Validate that two lists have the same length and are valid.

    Args:
        list1: First list to compare.
        list2: Second list to compare.
        name1: Descriptive name for the first list.
        name2: Descriptive name for the second list.

    Raises:
        TypeError: If either argument is not a list or names are not strings.
        ValueError: If the lists have different lengths or are empty.
    """
    # Validate input types
    if not isinstance(name1, str) or not isinstance(name2, str):
        raise TypeError("List names must be strings")

    if list1 is None or list2 is None:
        raise TypeError("Lists cannot be None")

    if not isinstance(list1, (list, tuple)):
        raise TypeError(f"{name1} must be a list or tuple, got {type(list1).__name__}")
    if not isinstance(list2, (list, tuple)):
        raise TypeError(f"{name2} must be a list or tuple, got {type(list2).__name__}")

    if not list1 and not list2:
        raise ValueError(f"Both {name1} and {name2} are empty")

    len1, len2 = len(list1), len(list2)
    if len1 != len2:
        raise ValueError(
            f"Length mismatch: {name1} has {len1} items, " f"{name2} has {len2} items"
        )

    # Additional validation for numeric lists (common use case)
    if list1 and hasattr(list1[0], "__float__"):  # Likely numeric
        try:
            # Check for NaN or infinite values in numeric lists
            import math

            for i, val in enumerate(list1):
                if isinstance(val, (int, float)) and (
                    math.isnan(val) or math.isinf(val)
                ):
                    raise ValueError(
                        f"{name1} contains invalid numeric value at index {i}: {val}"
                    )
            for i, val in enumerate(list2):
                if isinstance(val, (int, float)) and (
                    math.isnan(val) or math.isinf(val)
                ):
                    raise ValueError(
                        f"{name2} contains invalid numeric value at index {i}: {val}"
                    )
        except (TypeError, AttributeError):
            # If we can't check numeric values, skip this validation
            pass


def load_state_dict_with_prefix(
    model: TorchModelType,
    state_dict: StateDictType,
    strict: bool = True,
    assign: bool = False,
    key_prefix: str = "model.",
    operation: Literal["add", "remove"] = "remove",
) -> TorchModelType:
    """
    Load a state dict into a model, adding or removing a prefix from the keys.

    This is useful when loading state dicts saved with DataParallel, pytorch lightning or similar wrappers.

    Args:
        model: The model to load the state dict into.
        state_dict: The state dictionary to load.
        key_prefix: The prefix to add or remove from the keys.
        operation: 'add' to add the prefix, 'remove' to remove it.

    Returns:
        The model with the loaded state dict.
    """
    if operation not in ("add", "remove"):
        raise ValueError("operation must be either 'add' or 'remove'")

    modified_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if operation == "add":
            new_key = f"{key_prefix}{key}"
        else:  # operation == "remove"
            if key.startswith(key_prefix):
                new_key = key[len(key_prefix) :]
            else:
                raise ValueError(
                    f"Key '{key}' does not start with prefix '{key_prefix}'"
                )
        modified_state_dict[new_key] = value

    model.load_state_dict(modified_state_dict, strict=strict, assign=assign)
    return model


def state_dict_to_device(
    state_dict: StateDictType,
    device: Union[torch.device, str],
    copy: bool = False,
    inplace: bool = False,
) -> StateDictType:
    """
    Move state dict tensors to the specified device.

    Args:
        state_dict: The state dictionary to move.
        device: Target device for the tensors.
        copy: Whether to force a copy even when the tensor is already on the target device.
        inplace: Whether to modify the input state dict in place.

    Returns:
        State dict with tensors moved to the specified device.
    """
    if inplace:
        ret_state_dict = state_dict
    else:
        ret_state_dict = OrderedDict()

    for key, tensor in state_dict.items():
        ret_state_dict[key] = cast(Tensor, tensor).to(
            device, non_blocking=True, copy=copy
        )
    return ret_state_dict


def state_dicts_check_keys(state_dicts: List[StateDictType]) -> None:
    """
    Check that all state dictionaries have the same keys.

    Args:
        state_dicts: A list of state dictionaries to check.

    Raises:
        ValueError: If the state dictionaries have different keys or the list is empty.
    """
    _validate_state_dict_list_not_empty(state_dicts)
    _validate_state_dict_same_keys(state_dicts)


def num_params_of_state_dict(state_dict: StateDictType) -> int:
    """
    Calculate the total number of parameters in a state dict.

    Args:
        state_dict: The state dict to count parameters in.

    Returns:
        The total number of parameters in the state dict.
    """
    return sum(tensor.numel() for tensor in state_dict.values())


def state_dict_flatten(state_dict: StateDictType) -> Tensor:
    """
    Flatten all tensors in a state dict into a single 1D tensor.

    Args:
        state_dict: The state dict to flatten.

    Returns:
        A single flattened tensor containing all parameters.
    """
    return torch.cat([tensor.flatten() for tensor in state_dict.values()])


def state_dict_avg(state_dicts: List[StateDictType]) -> StateDictType:
    """
    Calculate the element-wise average of a list of state dicts.

    Args:
        state_dicts: List of state dicts to average.

    Returns:
        A state dict containing the averaged parameters.

    Raises:
        ValueError: If the list is empty or state dicts have different keys.
    """
    _validate_state_dict_list_not_empty(state_dicts)
    _validate_state_dict_same_keys(state_dicts)

    num_state_dicts = len(state_dicts)
    avg_state_dict = OrderedDict()

    # Initialize with zeros_like for better performance
    for key in state_dicts[0]:
        avg_state_dict[key] = torch.zeros_like(state_dicts[0][key])

    # Accumulate all state dicts
    for state_dict in state_dicts:
        for key in avg_state_dict:
            avg_state_dict[key] += state_dict[key]

    # Divide by number of state dicts
    for key in avg_state_dict:
        avg_state_dict[key] /= num_state_dicts

    return avg_state_dict


def state_dict_sub(
    a: StateDictType,
    b: StateDictType,
    strict: bool = True,
    device: Optional[Union[torch.device, str]] = None,
) -> StateDictType:
    """
    Compute the element-wise difference between two state dicts (a - b).

    Args:
        a: The first state dict (minuend).
        b: The second state dict (subtrahend).
        strict: Whether to require exact key matching between state dicts.
        device: Optional device to move the result tensors to.

    Returns:
        A state dict containing the element-wise differences.

    Raises:
        ValueError: If strict=True and the state dicts have different keys or incompatible tensor shapes.
        TypeError: If tensors have incompatible types.
    """
    result = OrderedDict()

    if strict:
        _validate_state_dict_same_keys([a, b])
        keys_to_process = a.keys()
    else:
        keys_to_process = set(a.keys()) & set(b.keys())

    for key in keys_to_process:
        result_tensor = a[key] - b[key]
        if device is not None:
            result_tensor = result_tensor.to(device, non_blocking=True)
        result[key] = result_tensor

    return result


def state_dict_add(
    a: StateDictType,
    b: StateDictType,
    strict: bool = True,
    device: Optional[Union[torch.device, str]] = None,
    show_pbar: bool = False,
) -> StateDictType:
    """
    Compute the element-wise sum of two state dicts.

    Args:
        a: The first state dict.
        b: The second state dict.
        strict: Whether to require exact key matching between state dicts.
        device: Optional device to move the result tensors to.
        show_pbar: Whether to show a progress bar during computation.

    Returns:
        A state dict containing the element-wise sums.

    Raises:
        ValueError: If strict=True and the state dicts have different parameters.
    """
    result = OrderedDict()

    if strict:
        _validate_state_dict_same_keys([a, b])
        keys_to_process = a.keys()
    else:
        keys_to_process = set(a.keys()) & set(b.keys())

    keys_iter = (
        tqdm(keys_to_process, desc="Adding state dicts")
        if show_pbar
        else keys_to_process
    )

    for key in keys_iter:
        if key in b:  # This check is redundant when strict=True but harmless
            result[key] = a[key] + b[key]

    if device is not None:
        result = state_dict_to_device(result, device)

    return result


def state_dict_add_scalar(state_dict: StateDictType, scalar: Number) -> StateDictType:
    """
    Add a scalar value to all parameters in a state dict.

    Args:
        state_dict: The state dict to modify.
        scalar: The scalar value to add to each parameter.

    Returns:
        A new state dict with the scalar added to each parameter.
    """
    return OrderedDict((key, tensor + scalar) for key, tensor in state_dict.items())


def state_dict_mul(
    state_dict: StateDictType,
    scalar: float,
    *,
    keep_dtype_when_zero: bool = True,
    show_pbar: bool = False,
) -> StateDictType:
    """
    Multiply all parameters in a state dict by a scalar.

    Args:
        state_dict: The state dict to multiply.
        scalar (float): The scalar value to multiply each parameter by.
        keep_dtype_when_zero (bool): Whether to keep the original data type of the tensors if either the tensor is all zeros or the scalar is zero.
        show_pbar (bool): Whether to show a progress bar during computation.

    Returns:
        A new state dict with each parameter multiplied by the scalar.
    """
    new_state_dict = OrderedDict()
    for key, tensor in (
        state_dict.items()
        if not show_pbar
        else tqdm(state_dict.items(), desc="Multiplying state dict")
    ):
        if (
            keep_dtype_when_zero
            and not tensor.is_floating_point()  # when tensor is not floating point, multiplication by 0 keeps dtype
            and (scalar == 0 or torch.all(tensor == 0))
        ):
            new_state_dict[key] = tensor.clone()
        else:
            new_state_dict[key] = scalar * tensor
    return new_state_dict


def state_dict_div(
    state_dict: StateDictType,
    scalar: float,
    *,
    keep_dtype_when_zero: bool = True,
    show_pbar: bool = False,
) -> StateDictType:
    """
    Divide all parameters in a state dict by a scalar.

    Args:
        state_dict: The state dict to divide.
        scalar: The scalar value to divide each parameter by.
        keep_dtype_when_zero: Whether to keep the original data type of the tensors if the tensor is all zeros.
        show_pbar: Whether to show a progress bar during computation.

    Returns:
        A new state dict with each parameter divided by the scalar.

    Raises:
        ZeroDivisionError: If scalar is zero.
    """
    if scalar == 0:
        raise ZeroDivisionError("Cannot divide state dict by zero")

    new_state_dict = OrderedDict()
    for key, tensor in (
        state_dict.items()
        if not show_pbar
        else tqdm(state_dict.items(), desc="Dividing state dict")
    ):
        if (
            keep_dtype_when_zero
            and not tensor.is_floating_point()  # when tensor is not floating point, division by any scalar keeps dtype
            and torch.all(tensor == 0)  # only check tensor for zero
        ):
            new_state_dict[key] = tensor.clone()
        else:
            new_state_dict[key] = tensor / scalar
    return new_state_dict


def state_dict_power(state_dict: StateDictType, p: float) -> StateDictType:
    """
    Raise all parameters in a state dict to a power.

    Args:
        state_dict: The state dict to raise to a power.
        p: The exponent to raise each parameter to.

    Returns:
        A new state dict with each parameter raised to the power p.
    """
    return OrderedDict((key, tensor**p) for key, tensor in state_dict.items())


def state_dict_interpolation(
    state_dicts: List[StateDictType], scalars: List[float]
) -> StateDictType:
    """
    Interpolate between multiple state dicts using specified scalar weights.

    Args:
        state_dicts: List of state dicts to interpolate between.
        scalars: List of scalar weights for interpolation.

    Returns:
        A state dict containing the interpolated parameters.

    Raises:
        ValueError: If the lists have different lengths or are empty, or if state dicts have different keys.
    """
    _validate_state_dict_list_not_empty(state_dicts)
    _validate_list_lengths_equal(state_dicts, scalars, "state_dicts", "scalars")
    _validate_state_dict_same_keys(state_dicts)

    interpolated_state_dict = OrderedDict()

    # Initialize with zeros
    for key in state_dicts[0]:
        interpolated_state_dict[key] = torch.zeros_like(state_dicts[0][key])

    # Accumulate weighted contributions
    for state_dict, scalar in zip(state_dicts, scalars):
        for key in interpolated_state_dict:
            interpolated_state_dict[key] += scalar * state_dict[key]

    return interpolated_state_dict


def state_dict_sum(state_dicts: List[StateDictType]) -> StateDictType:
    """
    Compute the element-wise sum of multiple state dicts.

    Args:
        state_dicts: List of state dicts to sum.

    Returns:
        A state dict containing the element-wise sums.

    Raises:
        ValueError: If the list is empty or state dicts have different keys.
    """
    _validate_state_dict_list_not_empty(state_dicts)
    _validate_state_dict_same_keys(state_dicts)

    sum_state_dict = OrderedDict()

    # Initialize with zeros
    for key in state_dicts[0]:
        sum_state_dict[key] = torch.zeros_like(state_dicts[0][key])

    # Accumulate all state dicts
    for state_dict in state_dicts:
        for key in sum_state_dict:
            sum_state_dict[key] += state_dict[key]

    return sum_state_dict


def state_dict_weighted_sum(
    state_dicts: List[StateDictType],
    weights: List[float],
    device: Optional[Union[torch.device, str]] = None,
) -> StateDictType:
    """
    Compute the weighted sum of multiple state dicts.

    Args:
        state_dicts: List of state dicts to combine.
        weights: List of weights for the weighted sum.
        device: Optional device to move the result tensors to.

    Returns:
        A state dict containing the weighted sum of parameters.

    Raises:
        ValueError: If the lists have different lengths or are empty, or if state dicts have different keys.
    """
    _validate_state_dict_list_not_empty(state_dicts)
    _validate_list_lengths_equal(state_dicts, weights, "state_dicts", "weights")
    _validate_state_dict_same_keys(state_dicts)

    weighted_sum_state_dict = OrderedDict()

    # Single pass initialization and computation for better performance
    for key in state_dicts[0]:
        # Get reference tensor and handle sparse tensors
        ref_tensor = state_dicts[0][key]
        is_sparse = ref_tensor.is_sparse if hasattr(ref_tensor, "is_sparse") else False

        # Initialize result tensor
        if is_sparse:
            # For sparse tensors, start with zeros in dense format for efficient accumulation
            result_tensor = torch.zeros_like(ref_tensor).to_dense()
        else:
            result_tensor = torch.zeros_like(ref_tensor)

        # Accumulate weighted contributions in a single loop
        for state_dict, weight in zip(state_dicts, weights):
            tensor = state_dict[key]

            # Optimize for common cases
            if weight == 0.0:
                continue  # Skip zero weights
            elif weight == 1.0:
                result_tensor += tensor  # Avoid multiplication for unit weights
            else:
                # Use in-place operations when possible for memory efficiency
                if is_sparse and hasattr(tensor, "is_sparse") and tensor.is_sparse:
                    result_tensor += weight * tensor.to_dense()
                else:
                    result_tensor += weight * tensor

        # Move to target device if specified (do this once per tensor, not per operation)
        if device is not None:
            result_tensor = result_tensor.to(device, non_blocking=True)

        # Convert back to sparse if original was sparse and result is suitable
        if is_sparse and hasattr(result_tensor, "to_sparse"):
            try:
                # Only convert back to sparse if it would be memory efficient
                # (i.e., if the result has sufficient sparsity)
                if result_tensor.numel() > 0:
                    sparsity_ratio = (result_tensor == 0).float().mean().item()
                    if sparsity_ratio > 0.5:  # Convert back if >50% zeros
                        result_tensor = result_tensor.to_sparse()
            except (RuntimeError, AttributeError):
                # If conversion fails, keep as dense
                pass

        weighted_sum_state_dict[key] = result_tensor

    return weighted_sum_state_dict


def state_dict_diff_abs(a: StateDictType, b: StateDictType) -> StateDictType:
    """
    Compute the element-wise absolute difference between two state dicts.

    Args:
        a: The first state dict.
        b: The second state dict.

    Returns:
        A state dict containing the absolute differences.
    """
    diff = state_dict_sub(a, b)
    return OrderedDict((key, tensor.abs()) for key, tensor in diff.items())


def state_dict_binary_mask(
    a: StateDictType,
    b: StateDictType,
    compare_fn: Union[
        Literal["greater", "less", "equal", "not_equal"],
        Callable[[Tensor, Tensor], torch.BoolTensor],
    ] = "greater",
    strict: bool = True,
    show_pbar: bool = False,
) -> BoolStateDictType:
    """
    Create binary masks by comparing elements in two state dicts.

    Args:
        a: The first state dict.
        b: The second state dict.
        compare_fn: Comparison function to use. Can be a string literal
                   ("greater", "less", "equal", "not_equal") or a callable
                   that takes two tensors and returns a boolean tensor.
        strict: Whether to require exact key matching between state dicts.
        show_pbar: Whether to show a progress bar during computation.

    Returns:
        A dictionary containing boolean masks based on the comparison.

    Raises:
        ValueError: If compare_fn is not a valid string or callable, or if strict=True
                   and the state dicts have different keys or incompatible tensor shapes.
        TypeError: If tensors have incompatible types.
    """
    compare_fn_dict = {
        "greater": lambda x, y: x > y,
        "less": lambda x, y: x < y,
        "equal": lambda x, y: x == y,
        "not_equal": lambda x, y: x != y,
    }

    if isinstance(compare_fn, str):
        if compare_fn not in compare_fn_dict:
            raise ValueError(
                f"Invalid compare_fn string: {compare_fn}. Must be one of {list(compare_fn_dict.keys())}"
            )
        compare_fn = compare_fn_dict[compare_fn]
    elif not callable(compare_fn):
        raise ValueError(
            f"compare_fn must be a string or a callable, but got {type(compare_fn)}"
        )

    result = OrderedDict()

    if strict:
        _validate_state_dict_same_keys([a, b])
        keys_to_process = a.keys()
    else:
        keys_to_process = set(a.keys()) & set(b.keys())

    keys_iter = (
        tqdm(keys_to_process, desc="Creating binary masks")
        if show_pbar
        else keys_to_process
    )

    for key in keys_iter:
        result[key] = compare_fn(a[key], b[key])

    return result


def state_dict_hadamard_product(a: StateDictType, b: StateDictType) -> StateDictType:
    """
    Compute the Hadamard product (element-wise multiplication) of two state dicts.

    Args:
        a: The first state dict.
        b: The second state dict.

    Returns:
        A state dict containing the element-wise products.

    Raises:
        ValueError: If the state dicts have different keys or incompatible tensor shapes.
        TypeError: If tensors have incompatible types.
    """
    _validate_state_dict_same_keys([a, b])
    return OrderedDict((key, a[key] * b[key]) for key in a)
