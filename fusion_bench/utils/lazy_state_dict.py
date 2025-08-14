import json
import logging
import os
from copy import deepcopy
from typing import TYPE_CHECKING, Dict, Iterator, List, Optional, Tuple, Type

import torch
from accelerate import init_empty_weights
from accelerate.utils.constants import SAFE_WEIGHTS_NAME, WEIGHTS_NAME
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import load_file
from torch import nn
from transformers import AutoConfig

from fusion_bench.utils.dtype import parse_dtype
from fusion_bench.utils.packages import import_object

if TYPE_CHECKING:
    from transformers import PretrainedConfig

log = logging.getLogger(__name__)

__all__ = ["resolve_checkpoint_path", "LazyStateDict"]


def resolve_checkpoint_path(
    checkpoint: str,
    hf_revision: Optional[str] = None,
    hf_cache_dir: Optional[str] = None,
    hf_proxies: Optional[Dict] = None,
):
    # If it's a local file or directory, return as is
    if os.path.exists(checkpoint):
        return checkpoint
    # If it's a HuggingFace Hub model id, download snapshot
    try:
        # This will download the model to the cache and return the local path
        local_path = snapshot_download(
            repo_id=checkpoint,
            revision=hf_revision,
            cache_dir=hf_cache_dir,
            proxies=hf_proxies,
        )
        return local_path
    except Exception as e:
        raise FileNotFoundError(
            f"Could not resolve checkpoint: {checkpoint}. Error: {e}"
        )


class LazyStateDict:
    """
    Dictionary-like object that lazily loads a state dict from a checkpoint path.
    """

    _local_path: str
    """local path to the checkpoint."""
    _state_dict_cache: Optional[Dict]
    """Cache for the state dict, if enabled."""
    _index_filename: Optional[str]
    _checkpoint_files: Optional[List[str]]
    _index: Optional[Dict[str, str]]
    """Mapping of parameter names to checkpoint files."""

    def __init__(
        self,
        checkpoint: str,
        meta_module_class: Optional[Type[nn.Module]] = None,
        meta_module: Optional[nn.Module] = None,
        cache_state_dict: bool = False,
        torch_dtype: Optional[torch.dtype] = None,
        device: str = "cpu",
        hf_revision: Optional[str] = None,
        hf_cache_dir: Optional[str] = None,
        hf_proxies: Optional[Dict] = None,
    ):
        """
        Args:
            checkpoint (str): Path to the checkpoint file or directory.
            meta_module_class (Type[nn.Module], optional): Class of the meta module to instantiate.
            meta_module (nn.Module, optional): Pre-initialized meta module.
            cache_state_dict (bool): Whether to cache the state dict in memory.
            torch_dtype (torch.dtype, optional): The dtype to use for the tensors.
            device (str): The device to load the tensors onto.
            hf_revision (str, optional): The revision of the model to download from Hugging Face Hub.
            hf_cache_dir (str, optional): The cache directory for Hugging Face models.
            hf_proxies (Dict, optional): Proxies to use for downloading from Hugging Face Hub.
        """
        self.cache_state_dict = cache_state_dict
        self.meta_module_class = meta_module_class
        if isinstance(self.meta_module_class, str):
            self.meta_module_class = import_object(self.meta_module_class)
        self.meta_module = meta_module
        if self.meta_module_class is not None:
            if self.meta_module is not None:
                raise ValueError(
                    "Cannot provide both meta_module_class and meta_module, please provide only one."
                )
            with init_empty_weights():
                self.meta_module = self.meta_module_class.from_pretrained(
                    checkpoint,
                    torch_dtype=torch_dtype,
                    revision=hf_revision,
                    cache_dir=hf_cache_dir,
                    proxies=hf_proxies,
                )

        self._checkpoint = checkpoint
        self._local_path = resolve_checkpoint_path(
            checkpoint,
            hf_revision=hf_revision,
            hf_cache_dir=hf_cache_dir,
            hf_proxies=hf_proxies,
        )

        self._index, self._index_filename, self._checkpoint_files = (
            self._resolve_checkpoint_files(self._local_path)
        )

        if self._index is not None:
            # if meta_module is provided, remove the keys that are not in the meta_module
            if self.meta_module is not None:
                meta_module_state_dict = self.meta_module.state_dict()
                for key in tuple(self._index.keys()):
                    if key not in meta_module_state_dict:
                        self._index.pop(key)
            if cache_state_dict:
                self._state_dict_cache = {}
            else:
                self._state_dict_cache = None
        elif len(self._checkpoint_files) == 1 and self._checkpoint_files[0].endswith(
            SAFE_WEIGHTS_NAME
        ):
            # let the keys of self._index be the keys of the state dict, the values are the checkpoint file
            with safe_open(
                self._checkpoint_files[0], framework="pt", device=device
            ) as f:
                self._index = {key: self._checkpoint_files[0] for key in f.keys()}
                if cache_state_dict:
                    self._state_dict_cache = {}
                else:
                    self._state_dict_cache = None
        elif len(self._checkpoint_files) == 1 and self._checkpoint_files[0].endswith(
            WEIGHTS_NAME
        ):
            log.info(f"Loading full state dict from {WEIGHTS_NAME}")
            self._state_dict_cache = torch.load(self._checkpoint_files[0])
            # if meta_module is provided, remove the keys that are not in the meta_module
            if self.meta_module is not None:
                meta_module_state_dict = self.meta_module.state_dict()
                for key in tuple(self._state_dict_cache.keys()):
                    if key not in meta_module_state_dict:
                        self._state_dict_cache.pop(key)
        else:
            raise ValueError(
                f"Cannot determine the type of checkpoint, please provide a checkpoint path to a file containing a whole state dict with file name {WEIGHTS_NAME} or {SAFE_WEIGHTS_NAME}, or the index of a sharded checkpoint ending with `.index.json`."
            )

        self._torch_dtype = parse_dtype(torch_dtype)
        self._device = device

    @property
    def checkpoint(self) -> str:
        return self._checkpoint

    @property
    def config(self) -> "PretrainedConfig":
        return AutoConfig.from_pretrained(self._checkpoint)

    def state_dict(self, keep_vars: bool = False) -> "LazyStateDict":
        """
        Args:
            keep_vars (bool): Ignored, as LazyStateDict does not support keep_vars. Just for compatibility.
        """
        return self

    def _resolve_checkpoint_files(self, checkpoint: str):
        # reference: https://huggingface.co/docs/accelerate/v0.17.1/en/usage_guides/big_modeling
        checkpoint_files = None
        index_filename = None
        if os.path.isfile(checkpoint):
            if str(checkpoint).endswith(".json"):
                index_filename = checkpoint
            else:
                checkpoint_files = [checkpoint]
        elif os.path.isdir(checkpoint):
            # check if the whole state dict is present
            potential_state_bin = [
                f for f in os.listdir(checkpoint) if f == WEIGHTS_NAME
            ]
            potential_state_safetensor = [
                f for f in os.listdir(checkpoint) if f == SAFE_WEIGHTS_NAME
            ]
            if len(potential_state_bin) == 1:
                checkpoint_files = [os.path.join(checkpoint, potential_state_bin[0])]
            elif len(potential_state_safetensor) == 1:
                checkpoint_files = [
                    os.path.join(checkpoint, potential_state_safetensor[0])
                ]
            else:
                # otherwise check for sharded checkpoints
                potential_index = [
                    f for f in os.listdir(checkpoint) if f.endswith(".index.json")
                ]
                if len(potential_index) == 0:
                    raise ValueError(
                        f"{checkpoint} is not a folder containing a `.index.json` file or a {WEIGHTS_NAME} or a {SAFE_WEIGHTS_NAME} file"
                    )
                elif len(potential_index) == 1:
                    index_filename = os.path.join(checkpoint, potential_index[0])
                else:
                    raise ValueError(
                        f"{checkpoint} containing more than one `.index.json` file, delete the irrelevant ones."
                    )
        else:
            raise ValueError(
                "`checkpoint` should be the path to a file containing a whole state dict, or the index of a sharded "
                f"checkpoint, or a folder containing a sharded checkpoint or the whole state dict, but got {checkpoint}."
            )

        if index_filename is not None:
            checkpoint_folder = os.path.split(index_filename)[0]
            with open(index_filename) as f:
                index = json.loads(f.read())

            if "weight_map" in index:
                index = index["weight_map"]
            checkpoint_files = sorted(list(set(index.values())))
            checkpoint_files = [
                os.path.join(checkpoint_folder, f) for f in checkpoint_files
            ]
        else:
            index = None
        return index, index_filename, checkpoint_files

    def _load_tensor_from_checkpoint_file(
        self, checkpoint_file: str, key: str, update_cache: bool = True
    ) -> torch.Tensor:
        if checkpoint_file.endswith(".safetensors"):
            with safe_open(checkpoint_file, framework="pt", device=self._device) as f:
                tensor = f.get_tensor(key)
                if self._torch_dtype is not None:
                    tensor = tensor.to(self._torch_dtype)
                if update_cache and self._state_dict_cache is not None:
                    self._state_dict_cache[key] = tensor
                return tensor
        else:
            state_dict = torch.load(checkpoint_file, map_location=self._device)
            if update_cache:
                if self._state_dict_cache is not None:
                    self._state_dict_cache.update(state_dict)
                else:
                    log.warning(
                        f"Load full state dict from file {checkpoint_file}, but state dict cache is disabled."
                    )
            return state_dict[key]

    def __getitem__(self, key: str) -> torch.Tensor:
        if self._state_dict_cache is not None and key in self._state_dict_cache:
            return self._state_dict_cache[key]

        if self._index is None:
            if len(self._checkpoint_files) == 1 and os.path.isfile(
                self._checkpoint_files[0]
            ):
                checkpoint_file = self._checkpoint_files[0]
                tensor = self._load_tensor_from_checkpoint_file(
                    checkpoint_file, key, update_cache=True
                )
                return tensor
            else:
                if len(self._checkpoint_files) > 1:
                    raise RuntimeError(
                        "Get multiple checkpoint files, but index is not provided."
                    )
                if not os.path.isfile(self._checkpoint_files[0]):
                    raise FileNotFoundError(
                        f"Checkpoint file {self._checkpoint_files[0]} not found."
                    )
                raise RuntimeError("Unexpected error.")
        else:
            if key not in self._index:
                raise KeyError(f"Key {key} not found in index.")
            checkpoint_file = os.path.join(self._local_path, self._index[key])
            if not os.path.isfile(checkpoint_file):
                raise FileNotFoundError(f"Checkpoint file {checkpoint_file} not found.")
            tensor = self._load_tensor_from_checkpoint_file(
                checkpoint_file, key, update_cache=True
            )
            return tensor

    def __setitem__(self, key: str, value: torch.Tensor) -> None:
        """
        Set a tensor in the LazyStateDict. This will update the state dict cache if it is enabled.
        """
        assert key in list(
            self.keys()
        ), "KeyError: Cannot set a tensor for a key that does not exist in the LazyStateDict."
        if self._state_dict_cache is not None:
            self._state_dict_cache[key] = value
        else:
            log.warning(
                "State dict cache is disabled, setting a tensor will not update the cache."
            )
            self._state_dict_cache = {key: value}

    def __contains__(self, key: str) -> bool:
        if self._state_dict_cache is not None and key in self._state_dict_cache:
            return True
        if self._index is not None and key in self._index:
            return True
        if len(self._checkpoint_files) == 1 and os.path.isfile(
            self._checkpoint_files[0]
        ):
            try:
                tensor = self._load_tensor_from_checkpoint_file(
                    self._checkpoint_files[0], key, update_cache=False
                )
                return tensor is not None
            except Exception:
                return False
        return False

    def __len__(self) -> int:
        if self._index is not None:
            return len(self._index)
        if len(self._checkpoint_files) == 1 and os.path.isfile(
            self._checkpoint_files[0]
        ):
            checkpoint_file = self._checkpoint_files[0]
            if checkpoint_file.endswith(".safetensors"):
                with safe_open(checkpoint_file, framework="pt", device="cpu") as f:
                    return len(tuple(f.keys()))
            else:
                return len(
                    tuple(torch.load(checkpoint_file, map_location="cpu").keys())
                )
        raise RuntimeError(
            "Unexpected error: cannot determine the number of keys in the state dict."
        )

    def __iter__(self) -> Iterator[str]:
        if self._index is not None:
            return iter(self._index)
        elif self._state_dict_cache is not None:
            return iter(self._state_dict_cache)
        else:
            raise RuntimeError(
                "Unexpected error: cannot determine the keys in the state dict."
            )

    def keys(self) -> Iterator[str]:
        for key in self:
            yield key

    def values(self) -> Iterator[torch.Tensor]:
        for key in self:
            yield self[key]

    def items(self) -> Iterator[Tuple[str, torch.Tensor]]:
        for key in self:
            yield key, self[key]

    def __repr__(self) -> str:
        if self._index is not None:
            return f"{self.__class__.__name__}(keys={list(self.keys())})"
        else:
            return (
                f"{self.__class__.__name__}(checkpoint_files={self._checkpoint_files})"
            )

    def get_parameter(self, target: str) -> torch.Tensor:
        return self[target]

    def get_submodule(self, target: str) -> nn.Module:
        if self.meta_module is not None:
            module: nn.Module = deepcopy(self.meta_module.get_submodule(target))
            module.to_empty(device=self._device)
            state_dict = {}
            for name, _ in module.named_parameters():
                state_dict[name] = self[f"{target}.{name}"]
            module.load_state_dict(state_dict)
            return module
        else:
            raise RuntimeError(
                "Cannot get submodule because meta_module is not provided."
            )

    def load_state_dict(
        self, state_dict: Dict[str, torch.Tensor], strict: bool = True
    ) -> None:
        """
        Load a state dict into this LazyStateDict.
        This method is only for compatibility with nn.Module and it overrides the cache of LazyStateDict.

        Args:
            state_dict (Dict[str, torch.Tensor]): The state dict to load.
            strict (bool): Whether to enforce that all keys in the state dict are present in this LazyStateDict.
        """
        log.warning(
            "Loading state dict into LazyStateDict is not recommended, as it may lead to unexpected behavior. "
            "Use with caution."
        )
        if strict:
            for key in state_dict:
                if key not in self:
                    raise KeyError(f"Key {key} not found in LazyStateDict.")
        for key, value in state_dict.items():
            self[key] = value
