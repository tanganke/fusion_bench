import os

import torch
from tqdm import tqdm

from fusion_bench import BaseAlgorithm, BaseModelPool, auto_register_config

from .utils import (
    subspace_consistency_spectral_calibration,
    subspace_consistency_spectral_calibration_accelerated,
)


@auto_register_config
class SingularValueCalibration(BaseAlgorithm):
    """
    Implements the Singular Value Calibration (SVC) method from the paper:
    "When Shared Knowledge Hurts: Spectral Over-Accumulation in Model Merging"
    (https://arxiv.org/abs/2602.05536)
    """

    def __init__(self, alpha: float, accelerator=None, **kwargs):
        """
        Initializes the Singular Value Calibration method.

        Args:
            alpha (float): Calibration strength hyperparameter that controls how much to scale down
                the merged responses along shared spectral subspaces. Higher values lead to more aggressive
                calibration, while lower values retain more of the original merged responses. Default is 1.0.
        """
        super().__init__(**kwargs)
        self.alpha = alpha
        self.accelerator = accelerator

    def _calibration_impl(self):
        impl = os.environ.get("FUSION_BENCH_SVC_IMPL", "accelerated").lower()
        if impl in {"original"}:
            return subspace_consistency_spectral_calibration
        return subspace_consistency_spectral_calibration_accelerated

    @torch.no_grad()
    def run(self, modelpool):
        """
        Runs the Singular Value Calibration method on the given model pool.

        Args:
            modelpool (BaseModelPool): The pool of models to calibrate.

        Returns:
            nn.Module: The calibrated merged model.
        """
        if not isinstance(modelpool, BaseModelPool):
            modelpool = BaseModelPool(modelpool)

        assert (
            modelpool.has_merged
        ), "Merged model not found in the model pool (with model name '_merged_'). Please run the merging step first."
        assert (
            modelpool.has_pretrained
        ), "Pretrained model not found in the model pool (with model name '_pretrained_'). Please run the pretraining step first."

        merged_model = modelpool.load_model("_merged_")
        pretrained_model = modelpool.load_pretrained_model()
        task_models = [
            modelpool.load_model(model_name) for model_name in modelpool.model_names
        ]
        calibration_impl = self._calibration_impl()

        for name, param in tqdm(
            tuple(merged_model.named_parameters()),
            desc="Calibrating merged model",
        ):
            if param.dim() == 2:  # Only calibrate weight matrices
                tqdm.write(f"Calibrating parameter: {name}, shape: {param.shape}")
                base_weight = pretrained_model.get_parameter(name).data
                task_weights = [
                    task_model.get_parameter(name).data for task_model in task_models
                ]
                merged_weight = param.data

                calibrated_weight = calibration_impl(
                    base_weight=base_weight,
                    task_weights=task_weights,
                    merged_weight=merged_weight,
                    alpha=self.alpha,
                    accelerator=self.accelerator,
                )
                param.data.copy_(calibrated_weight, non_blocking=True)
            else:
                # For non-weight parameters (e.g., biases, LayerNorm weights), we keep them unchanged.
                pass

        return merged_model
