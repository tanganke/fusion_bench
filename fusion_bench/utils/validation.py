"""
Validation utilities for FusionBench.

This module provides robust input validation functions to ensure data integrity
and provide clear error messages throughout the FusionBench framework.
"""

import logging
from pathlib import Path
from typing import Any, Optional, Union

__all__ = [
    "ValidationError",
    "validate_path_exists",
    "validate_file_exists",
    "validate_directory_exists",
    "validate_model_name",
]

log = logging.getLogger(__name__)


class ValidationError(ValueError):
    """Custom exception for validation errors with detailed context."""

    def __init__(self, message: str, field: Optional[str] = None, value: Any = None):
        self.field = field
        self.value = value
        detailed_message = message
        if field:
            detailed_message = f"Validation error for '{field}': {message}"
        if value is not None:
            detailed_message += f" (got: {value!r})"
        super().__init__(detailed_message)


def validate_path_exists(
    path: Union[str, Path],
    name: str = "path",
    create_if_missing: bool = False,
    must_be_file: bool = False,
    must_be_dir: bool = False,
) -> Path:
    """
    Validate that a path exists and optionally check its type.

    Args:
        path: Path to validate.
        name: Name of the path for error messages.
        create_if_missing: If True and path doesn't exist, create it as a directory.
        must_be_file: If True, ensure path points to a file.
        must_be_dir: If True, ensure path points to a directory.

    Returns:
        Path object of the validated path.

    Raises:
        ValidationError: If path validation fails.

    Examples:
        >>> validate_path_exists("./config", name="config_dir", must_be_dir=True)
        PosixPath('config')
    """
    if path is None:
        raise ValidationError(f"{name} cannot be None", field=name, value=path)

    assert not (
        create_if_missing and must_be_file
    ), "create_if_missing and must_be_file cannot both be True. By definition, a created path is a directory."

    path_obj = Path(path).expanduser().resolve()

    if not path_obj.exists():
        if create_if_missing:
            log.info(f"Creating missing directory: {path_obj}")
            path_obj.mkdir(parents=True, exist_ok=True)
        else:
            raise ValidationError(
                f"{name} does not exist: {path_obj}", field=name, value=str(path)
            )

    if must_be_file and not path_obj.is_file():
        raise ValidationError(
            f"{name} must be a file, but got directory: {path_obj}",
            field=name,
            value=str(path),
        )

    if must_be_dir and not path_obj.is_dir():
        raise ValidationError(
            f"{name} must be a directory, but got file: {path_obj}",
            field=name,
            value=str(path),
        )

    return path_obj


def validate_file_exists(path: Union[str, Path], name: str = "file") -> Path:
    """
    Validate that a file exists.

    Args:
        path: File path to validate.
        name: Name of the file for error messages.

    Returns:
        Path object of the validated file.

    Raises:
        ValidationError: If file doesn't exist or is not a file.
    """
    return validate_path_exists(path, name=name, must_be_file=True)


def validate_directory_exists(
    path: Union[str, Path], name: str = "directory", create_if_missing: bool = False
) -> Path:
    """
    Validate that a directory exists.

    Args:
        path: Directory path to validate.
        name: Name of the directory for error messages.
        create_if_missing: If True, create directory if it doesn't exist.

    Returns:
        Path object of the validated directory.

    Raises:
        ValidationError: If directory doesn't exist (and not creating) or is not a directory.
    """
    return validate_path_exists(
        path, name=name, must_be_dir=True, create_if_missing=create_if_missing
    )


def validate_model_name(
    model_name: str, allow_special: bool = True, field: str = "model_name"
) -> str:
    """
    Validate a model name string.

    Args:
        model_name: Model name to validate.
        allow_special: If True, allow special names like "_pretrained_". If False,
            names starting and ending with underscores will be rejected.
        field: Field name for error messages.

    Returns:
        The validated model name.

    Raises:
        ValidationError: If model name is invalid.

    Examples:
        >>> validate_model_name("openai/clip-vit-base-patch32")
        'openai/clip-vit-base-patch32'
        >>> validate_model_name("_pretrained_", allow_special=True)
        '_pretrained_'
        >>> validate_model_name("_pretrained_", allow_special=False)
        Traceback (most recent call last):
        ...
        ValidationError: Validation error for 'model_name': Special model names (starting and ending with '_') are not allowed (got: '_pretrained_')
    """
    if not model_name or not isinstance(model_name, str):
        raise ValidationError(
            "Model name must be a non-empty string", field=field, value=model_name
        )

    model_name = model_name.strip()
    if not model_name:
        raise ValidationError(
            "Model name cannot be empty or whitespace only",
            field=field,
            value=model_name,
        )

    # Check for special names (e.g., _pretrained_, _base_model_)
    if not allow_special and model_name.startswith("_") and model_name.endswith("_"):
        raise ValidationError(
            "Special model names (starting and ending with '_') are not allowed",
            field=field,
            value=model_name,
        )

    # Check for invalid characters that might cause issues
    invalid_chars = ["\n", "\r", "\t", "\0"]
    for char in invalid_chars:
        if char in model_name:
            raise ValidationError(
                f"Model name contains invalid character: {char!r}",
                field=field,
                value=model_name,
            )

    return model_name
