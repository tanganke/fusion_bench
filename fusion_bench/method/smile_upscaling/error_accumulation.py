import os
from typing import Literal, cast

import pandas as pd
import torch
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPVisionModel

from fusion_bench import BaseAlgorithm, BaseModelPool, auto_register_config
from fusion_bench.dataset import CLIPDataset
from fusion_bench.method import SmileUpscalingAlgorithm
from fusion_bench.mixins import LightningFabricMixin, SimpleProfilerMixin
from fusion_bench.modelpool import CLIPVisionModelPool
from fusion_bench.taskpool.clip_vision.taskpool import LayerWiseFeatureSaver
from fusion_bench.utils.devices import clear_cuda_cache


@auto_register_config
class LowRankApproximation(BaseAlgorithm):
    def __init__(self, rank: int, device: str = "cuda", **kwargs):
        """Low-rank approximation of fine-tuned updates."""
        super().__init__(**kwargs)

    def run(self, modelpool: BaseModelPool):
        # Implement low-rank approximation logic here
        base_model = modelpool.load_pretrained_model()

        models = {}
        for model_name in tqdm(modelpool.model_names, "processing models"):
            task_model = modelpool.load_model(model_name)
            for module_name, module in task_model.named_modules():
                if isinstance(module, nn.Linear):
                    w = cast(
                        nn.Linear, base_model.get_submodule(module_name)
                    ).weight.to(dtype=torch.float32, device=self.device, copy=True)
                    w_ft = module.weight.to(
                        dtype=torch.float32, device=self.device, copy=True
                    )

                    # Compute low-rank approximation
                    w_diff = w_ft - w
                    u, s, vh = torch.linalg.svd(w_diff)
                    v = vh.T

                    u = u[:, : self.rank]
                    s = s[: self.rank]
                    v = v[:, : self.rank]

                    low_rank_w_diff = torch.linalg.multi_dot((u, torch.diag(s), v.T))
                    low_rank_w = w + low_rank_w_diff

                    module.weight.data = low_rank_w.to(
                        dtype=module.weight.dtype,
                        device=module.weight.device,
                    )

            models[model_name] = task_model
        return models


@auto_register_config
class ErrorAccumulationAnalysisForCLIP(
    LightningFabricMixin,
    BaseAlgorithm,
):
    def __init__(
        self,
        gate_k: int,
        k: int,
        seed: int = 42,
        top_k: int = 1,
        dataset_kwargs: DictConfig = None,
        max_samples: int = 1024,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if dataset_kwargs is None:
            self.dataset_kwargs = DictConfig(
                {
                    "batch_size": 32,
                    "num_workers": 4,
                }
            )

    def run(self, modelpool: CLIPVisionModelPool):
        assert self.fabric.world_size == 1, "Distributed inference is not supported."
        # get the smile model
        smile_algorithm = SmileUpscalingAlgorithm(
            gate_k=self.gate_k, k=self.k, top_k=self.top_k, device=self.fabric.device
        )
        smile_model = smile_algorithm.run(modelpool)
        # get low-rank models
        low_rank_models = LowRankApproximation(rank=self.k).run(modelpool)

        results = {
            "model_name": [],
            "method": [],
            "layer_index": [],
            "approximation_error": [],
        }

        for model_name in modelpool.model_names:
            dataset = modelpool.load_test_dataset(model_name)
            processor = modelpool.load_processor()
            dataset = CLIPDataset(dataset, processor)
            dataloader = DataLoader(dataset, shuffle=True, **self.dataset_kwargs)
            dataloader = self.fabric.setup_dataloaders(dataloader)

            # finetuned_model
            finetuned_model = modelpool.load_model(model_name)
            finetuned_model = self.to_device(finetuned_model)
            self.collect_hidden_states(
                finetuned_model,
                dataloader=dataloader,
                model_name=f"{model_name}/finetuned",
            )
            del finetuned_model
            clear_cuda_cache()

            # smile model
            smile_model = self.to_device(smile_model)
            self.collect_hidden_states(
                smile_model, dataloader=dataloader, model_name=f"{model_name}/smile"
            )
            smile_model.cpu()
            clear_cuda_cache()

            # low-rank models
            model = low_rank_models.pop(model_name)
            model = self.to_device(model)
            self.collect_hidden_states(
                model, dataloader=dataloader, model_name=f"{model_name}/low-rank"
            )
            del model
            clear_cuda_cache()

            del dataloader
            clear_cuda_cache()

    @torch.no_grad()
    def collect_hidden_states(
        self, model: CLIPVisionModel, dataloader, model_name: str
    ):
        self.fabric.seed_everything(
            self.seed, workers=True
        )  # make sure to get same data samples
        # register hooks
        hooks = {}
        hook_handles = {}
        for i, layer in enumerate(model.vision_model.encoder.layers):
            hooks[i] = LayerWiseFeatureSaver(
                save_path=os.path.join(self.log_dir, model_name, f"layer_{i}.pth"),
                first_token_only=True,
            )
            hook_handles[i] = layer.register_forward_hook(hooks[i])

        # forward pass
        num_total_samples = 0
        for images, _ in tqdm(dataloader, desc=f"Collecting features for {model_name}"):
            batch_size = images.size(0)
            model(images)
            num_total_samples += batch_size
            if num_total_samples >= self.max_samples:
                break

        # save features
        for i, hook in hooks.items():
            hook.save_features()

        # remove hooks
        for i, hook_handle in hook_handles.items():
            hook_handle.remove()

        return hooks
