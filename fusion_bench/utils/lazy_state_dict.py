import json
import logging
import os
from typing import TYPE_CHECKING, Dict, Iterator, List, Optional, Tuple

import torch
from accelerate.utils.constants import SAFE_WEIGHTS_NAME, WEIGHTS_NAME
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoConfig

from fusion_bench.utils.dtype import parse_dtype

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
    _state_dict_cache: Optional[Dict]
    _index_filename: Optional[str]
    _checkpoint_files: Optional[List[str]]
    _index: Optional[Dict]

    def __init__(
        self,
        checkpoint: str,
        cache_state_dict: bool = False,
        torch_dtype: Optional[torch.dtype] = None,
        device: str = "cpu",
        hf_revision: Optional[str] = None,
        hf_cache_dir: Optional[str] = None,
        hf_proxies: Optional[Dict] = None,
    ):
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

        if cache_state_dict:
            self._state_dict_cache = {}
        else:
            self._state_dict_cache = None

        self._torch_dtype = parse_dtype(torch_dtype)
        self._device = device

    @property
    def checkpoint(self) -> str:
        return self._checkpoint

    @property
    def config(self) -> "PretrainedConfig":
        return AutoConfig.from_pretrained(self._checkpoint)

    def state_dict(self) -> "LazyStateDict":
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
        return iter(self._checkpoint_files)

    def keys(self) -> List[str]:
        return list(self)

    def values(self) -> List[torch.Tensor]:
        return [self[key] for key in self]

    def items(self) -> Iterator[Tuple[str, torch.Tensor]]:
        return ((key, self[key]) for key in self)

    def __repr__(self) -> str:
        if self._index is not None:
            return f"{self.__class__.__name__}(index={self._index})"
        else:
            return (
                f"{self.__class__.__name__}(checkpoint_files={self._checkpoint_files})"
            )
