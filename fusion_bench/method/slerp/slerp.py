import logging
import os
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, Optional

import torch
from torch import nn
from tqdm import tqdm
from typing_extensions import override

from fusion_bench import LazyStateDict, create_default_model_card, timeit_context
from fusion_bench.method import BaseAlgorithm
from fusion_bench.mixins import SimpleProfilerMixin, auto_register_config
from fusion_bench.modelpool import BaseModelPool, CausalLMPool
from fusion_bench.utils.type import StateDictType

from .slerp_utils import slerp

if TYPE_CHECKING:
    from transformers import PreTrainedModel

log = logging.getLogger(__name__)


def slerp_on_state_dicts(
    t,
    primary_state_dict,
    secondary_state_dict,
    *,
    DOT_THRESHOLD: float = 0.9995,
    epsilon: float = 1e-8,
    show_pbar: bool = False,
) -> StateDictType:
    """
    Perform spherical linear interpolation (slerp) on the state dictionaries of two models.

    Args:
        t (float): The interpolation factor, typically between 0 and 1.
        primary_state_dict (dict): The state dictionary of the primary model.
        secondary_state_dict (dict): The state dictionary of the secondary model.
        DOT_THRESHOLD (float, optional): Threshold for considering the vectors as collinear. Defaults to 0.9995.
        epsilon (float, optional): Small value to avoid division by zero. Defaults to 1e-8.

    Returns:
        dict: The interpolated state dictionary.
    """
    state_dict = {}
    pbar = secondary_state_dict if not show_pbar else tqdm(secondary_state_dict)
    for key in pbar:
        v0 = primary_state_dict[key]
        v1 = secondary_state_dict[key]
        if v0.shape != v1.shape:
            log.warning(
                f"Skipping key {key} because the shapes of the tensors are different: {v0.shape} vs {v1.shape}. Base model parameters will be used."
            )
            state_dict[key] = v0
        else:
            state_dict[key] = slerp(t, v0, v1, DOT_THRESHOLD, epsilon)
    return state_dict


@auto_register_config
class SlerpMergeAlgorithm(BaseAlgorithm):
    """
    General purpose implementation of Slerp (Spherical Linear Interpolation) for PyTorch models.
    """

    def __init__(
        self,
        t: float,
        DOT_THRESHOLD: float = 0.9995,
        epsilon: float = 1e-8,
        **kwargs,
    ):
        """
        Initialize the SlerpMergeAlgorithm.

        Args:
            t (float): The interpolation parameter. Must be in the range [0, 1].
            DOT_THRESHOLD (float, optional): The threshold for the dot product of the two vectors. Defaults to 0.9995.
            epsilon (float, optional): The epsilon value for numerical stability. Defaults to 1e-8.
        """
        super().__init__(**kwargs)

    @override
    def run(self, modelpool: BaseModelPool) -> nn.Module:
        """
        Run the SlerpMergeAlgorithm on the given model pool.

        Args:
            modelpool (BaseModelPool): The pool of models to fuse.

        Returns:
            nn.Module: The fused model.
        """
        assert len(modelpool.all_model_names) == 2, "Slerp expect exactly 2 models"
        primary_model = modelpool.load_model(modelpool.all_model_names[0])
        secondary_model = modelpool.load_model(modelpool.all_model_names[1])

        with torch.no_grad():
            primary_state_dict = primary_model.state_dict()
            secondary_state_dict = secondary_model.state_dict()
            state_dict = slerp_on_state_dicts(
                self.t,
                primary_state_dict,
                secondary_state_dict,
                DOT_THRESHOLD=self.DOT_THRESHOLD,
                epsilon=self.epsilon,
            )

        primary_model.load_state_dict(state_dict)
        return primary_model


@auto_register_config
class SlerpForCausalLM(
    SimpleProfilerMixin,
    BaseAlgorithm,
):
    """
    Slerp (Spherical Linear Interpolation) for Causal Language Models.
    """

    def __init__(
        self,
        t: float,
        DOT_THRESHOLD: float = 0.9995,
        epsilon: float = 1e-8,
        model_save_path: Optional[str] = None,
        show_pbar: bool = False,
        **kwargs,
    ):
        """
        Initialize the SlerpForCausalLM algorithm.

        Args:
            t (float): The interpolation parameter. Must be in the range [0, 1].
                      t=0 returns the first model, t=1 returns the second model,
                      t=0.5 provides balanced interpolation.
            DOT_THRESHOLD (float, optional): The threshold for the dot product of normalized vectors.
                                           When the absolute dot product exceeds this threshold,
                                           vectors are considered nearly collinear and linear
                                           interpolation (LERP) is used instead of SLERP for
                                           numerical stability. Defaults to 0.9995.
            epsilon (float, optional): Small value used for numerical stability to avoid
                                     division by zero during vector normalization.
                                     Defaults to 1e-8.
            model_save_path (Optional[str], optional): Path where the merged model should be saved.
                                                     If None, the model is not saved to disk.
                                                     Defaults to None.
            show_pbar (bool, optional): Whether to display a progress bar during the interpolation
                                      process. Useful for debugging or monitoring progress with
                                      large models. Defaults to False.
            **kwargs: Additional keyword arguments passed to the parent BaseAlgorithm class.
        """
        super().__init__(**kwargs)

    @override
    def run(self, modelpool: CausalLMPool):
        assert len(modelpool.all_model_names) == 2, "Slerp expect exactly 2 models"
        primary_model = modelpool.load_model(modelpool.all_model_names[0])
        secondary_model = modelpool.load_model(modelpool.all_model_names[1])

        with torch.no_grad():
            primary_state_dict = primary_model.state_dict()
            secondary_state_dict = secondary_model.state_dict()
            state_dict = slerp_on_state_dicts(
                self.t,
                primary_state_dict,
                secondary_state_dict,
                DOT_THRESHOLD=self.DOT_THRESHOLD,
                epsilon=self.epsilon,
            )

        if isinstance(primary_model, nn.Module):
            model = primary_model
            model.load_state_dict(state_dict)
        elif isinstance(primary_model, LazyStateDict):
            model: "PreTrainedModel" = deepcopy(primary_model.meta_module)
            model.to(device=primary_model._device)
            model.load_state_dict(state_dict)
        else:
            raise TypeError(
                f"Unsupported model type: {type(primary_model)}. "
                "Expected nn.Module or LazyStateDict."
            )
        if self.model_save_path is not None:
            with timeit_context(f"Saving the model to {self.model_save_path}"):
                tokenizer = modelpool.load_tokenizer()
                tokenizer.save_pretrained(self.model_save_path)
                model.save_pretrained(self.model_save_path)
                model_card_str = create_default_model_card(
                    models=[modelpool.get_model_path(m) for m in modelpool.model_names],
                    description="Merged model using Slerp.",
                    algorithm_config=self.config,
                    modelpool_config=modelpool.config,
                )
                with open(os.path.join(self.model_save_path, "README.md"), "w") as f:
                    f.write(model_card_str)
        return model
