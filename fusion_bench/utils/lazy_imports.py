# Copyright (c) 2020, 2021 The HuggingFace Team
# Copyright (c) 2021 Philip May, Deutsche Telekom AG
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Lazy-Imports module.

This is code taken from the `HuggingFace team <https://huggingface.co/>`__.
Many thanks to HuggingFace for
`your consent <https://github.com/huggingface/transformers/issues/12861#issuecomment-886712209>`__
to publish it as a standalone package.
"""

import importlib
import os
from types import ModuleType
from typing import Any, Dict, List, Optional, Set, Union


class LazyImporter(ModuleType):
    """Lazy importer for modules and their components.

    This class allows for lazy importing of modules, meaning modules are only
    imported when they are actually accessed. This can help reduce startup
    time and memory usage for large packages with many optional dependencies.

    Attributes:
        _modules: Set of module names available for import.
        _class_to_module: Mapping from class/function names to their module names.
        _objects: Dictionary of extra objects to include in the module.
        _name: Name of the module.
        _import_structure: Dictionary mapping module names to lists of their exports.
    """

    # Very heavily inspired by optuna.integration._IntegrationModule
    # https://github.com/optuna/optuna/blob/master/optuna/integration/__init__.py
    def __init__(
        self,
        name: str,
        module_file: str,
        import_structure: Dict[str, List[str]],
        extra_objects: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the LazyImporter.

        Args:
            name: The name of the module.
            module_file: Path to the module file.
            import_structure: Dictionary mapping module names to lists of their exports.
            extra_objects: Optional dictionary of extra objects to include.
        """
        super().__init__(name)
        self._modules: Set[str] = set(import_structure.keys())
        self._class_to_module: Dict[str, str] = {}
        for key, values in import_structure.items():
            for value in values:
                self._class_to_module[value] = key
        # Needed for autocompletion in an IDE
        self.__all__: List[str] = list(import_structure.keys()) + sum(
            import_structure.values(), []
        )
        self.__file__ = module_file
        self.__path__ = [os.path.dirname(module_file)]
        self._objects: Dict[str, Any] = {} if extra_objects is None else extra_objects
        self._name = name
        self._import_structure = import_structure

    # Needed for autocompletion in an IDE
    def __dir__(self) -> List[str]:
        """Return list of available attributes for autocompletion.

        Returns:
            List of all available attribute names.
        """
        return super().__dir__() + self.__all__

    def __getattr__(self, name: str) -> Any:
        """Get attribute lazily, importing the module if necessary.

        Args:
            name: The name of the attribute to retrieve.

        Returns:
            The requested attribute.

        Raises:
            AttributeError: If the attribute is not found in any module.
        """
        if name in self._objects:
            return self._objects[name]
        if name in self._modules:
            value = self._get_module(name)
        elif name in self._class_to_module:
            module = self._get_module(self._class_to_module[name])
            value = getattr(module, name)
        else:
            raise AttributeError(f"module {self.__name__} has no attribute {name}")

        setattr(self, name, value)
        return value

    def _get_module(self, module_name: str) -> ModuleType:
        """Import and return the specified module.

        Args:
            module_name: Name of the module to import.

        Returns:
            The imported module.
        """
        return importlib.import_module("." + module_name, self.__name__)

    def __reduce__(self) -> tuple:
        """Support for pickling the LazyImporter.

        Returns:
            Tuple containing the class and arguments needed to reconstruct the object.
        """
        return (self.__class__, (self._name, self.__file__, self._import_structure))


class LazyPyModule(ModuleType):
    """Module wrapper for lazy import.

    Adapted from Optuna: https://github.com/optuna/optuna/blob/1f92d496b0c4656645384e31539e4ee74992ff55/optuna/__init__.py

    This class wraps specified module and lazily import it when they are actually accessed.
    This can help reduce startup time and memory usage by deferring module imports
    until they are needed.

    Args:
        name: Name of module to apply lazy import.

    Attributes:
        _name: The name of the module to be lazily imported.
    """

    def __init__(self, name: str) -> None:
        """Initialize the LazyPyModule.

        Args:
            name: The name of the module to be lazily imported.
        """
        super().__init__(name)
        self._name: str = name

    def _load(self) -> ModuleType:
        """Load the actual module and update this object's dictionary.

        Returns:
            The loaded module.
        """
        module = importlib.import_module(self._name)
        self.__dict__.update(module.__dict__)
        return module

    def __getattr__(self, item: str) -> Any:
        """Get attribute from the lazily loaded module.

        Args:
            item: The name of the attribute to retrieve.

        Returns:
            The requested attribute from the loaded module.
        """
        return getattr(self._load(), item)
