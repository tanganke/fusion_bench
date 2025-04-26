import copy
import gc
import logging
import os
from abc import abstractmethod
from typing import Any, Callable, List, Mapping, Union, cast  # noqa: F401

import torch
from lightning.fabric.utilities.rank_zero import rank_zero_only
from omegaconf import DictConfig
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm

from fusion_bench.compat.method import ModelFusionAlgorithm
from fusion_bench.compat.modelpool import ModelPool
from fusion_bench.mixins.lightning_fabric import LightningFabricMixin
from fusion_bench.mixins.simple_profiler import SimpleProfilerMixin
from fusion_bench.modelpool import (
    CLIPVisionModelPool,
    GPT2ForSequenceClassificationPool,
)
from fusion_bench.models.wrappers.layer_wise_fusion import (
    LayerWiseMergedModel,
    get_layer_wise_weights,
)
from fusion_bench.utils.data import load_tensor_from_file

from .entropy_loss import entropy_loss

log = logging.getLogger(__name__)


# obtain the current GPU memory usage
def get_memory_usage(desc):
    allocated = torch.cuda.memory_allocated() / 1024**2  # 转换为 MB
    cached = torch.cuda.memory_reserved() / 1024**2  # 转换为 MB
    return (
        f"{desc}\nAllocated Memory: {allocated:.2f} MB\nCached Memory: {cached:.2f} MB"
    )


class ModelScheduler:
    """
    Manage the storage of models, schedule the order in which models are loaded to GPU
    transfer data between the CPU and GPu
    """

    def __init__(
        self,
        config: DictConfig,
        modelpool: ModelPool,
    ):
        self.pretrained_model = modelpool.load_model("_pretrained_")
        self.finetuned_models = [
            modelpool.load_model(name) for name in modelpool.model_names
        ]
        self.num_finetuned_models = len(self.finetuned_models)
        self.new_finetuned_models = copy.deepcopy(self.finetuned_models)
        self.finetuned_models_name = [name for name in modelpool.model_names]

        self.config = config

    @torch.no_grad()  # not sure whether to use this
    def __call__(self, model_id):
        """
        return models and relevant data in each step
        """
        pretrained_model = copy.deepcopy(self.pretrained_model)
        if self.config.topo == "ring":
            finetuned_models = [
                copy.deepcopy(
                    self.finetuned_models[(model_id + 1) % self.num_finetuned_models]
                ),
                copy.deepcopy(self.finetuned_models[model_id]),
                copy.deepcopy(
                    self.finetuned_models[(model_id - 1) % self.num_finetuned_models]
                ),
            ]
        elif "rotate" in self.config.topo:
            number = self.config.topo.split("_")[1]
            finetuned_models = [copy.deepcopy(self.finetuned_models[model_id])]
            for i in range(0, int(number)):
                finetuned_models.append(
                    copy.deepcopy(
                        self.finetuned_models[
                            (model_id + i + 1) % self.num_finetuned_models
                        ]
                    )
                )
        # initialize layer-wise weights using the provided configuration `init_values` or load from file if `weights` is provided
        if self.config.weights is None:
            layer_wise_weight = get_layer_wise_weights(
                num_models=len(finetuned_models),
                num_layers=len(
                    tuple(
                        filter(lambda p: p.requires_grad, pretrained_model.parameters())
                    )
                ),
                init_values=self.config.init_values,
            )
        else:
            if isinstance(self.config.weights, str):
                # self.config.weights is a path to a saved tensor
                layer_wise_weight = load_tensor_from_file(self.config.weights)
            else:
                raise ValueError(f"Unsupported weights format: {self.config.weights}")

        module = LayerWiseMergedModel(
            layer_wise_weight=layer_wise_weight,
            pretrained_model=pretrained_model,
            finetuned_models=finetuned_models,
            clamp_weights=self.config.clamp_weights,
            tie_weights=self.config.tie_weights,
            strict=self.config.strict,
        )
        print(f"{layer_wise_weight.size()=}, {layer_wise_weight.numel()=}")
        return module

    def store_model(self, new_finetuned_model_dict, model_id):
        """
        store new finetuned model after every turn of adamerging
        """
        self.new_finetuned_models[model_id].load_state_dict(new_finetuned_model_dict)

    def update_models(self):
        self.finetuned_models = copy.deepcopy(self.new_finetuned_models)

    def get_final_models(self, idx=None):
        # need a check
        if idx is not None:
            return copy.deepcopy(self.finetuned_models[idx])

        final_models = [
            {"name": name, "model": model}
            for name, model in zip(self.finetuned_models_name, self.finetuned_models)
        ]
        num_finetuned_models = len(self.finetuned_models)

        average_model = copy.deepcopy(self.pretrained_model)
        state_dict = average_model.state_dict(keep_vars=True)
        for name, _ in self.finetuned_models[0].named_parameters():
            state_dict[name].data.zero_()
        for model in self.finetuned_models:
            for name, param in model.named_parameters():
                state_dict[name] = state_dict[name] + 1 / num_finetuned_models * param

        average_model.load_state_dict(state_dict)
        final_models += [{"name": "average model", "model": average_model}]

        return final_models

    def move_to(self, device):
        self.pretrained_model.to(device=device)
        for model in self.finetuned_models:
            model.to(device=device)


class LayerWiseGossipAlgorithm(
    ModelFusionAlgorithm,
    LightningFabricMixin,
    SimpleProfilerMixin,
):
    """
    Implements the Layer-Wise AdaMerging Algorithm.

    This class merges the layers of a pretrained model with those of several fine-tuned models.
    The merging is controlled by layer-wise weights, which can be initialized based on a provided configuration or loaded from a file.
    """

    def __init__(self, algorithm_config: DictConfig):
        """
        Initialize the LayerWiseAdaMergingAlgorithm with the given configuration.

        Args:
            algorithm_config (DictConfig): The configuration for the algorithm.
        """
        super().__init__(algorithm_config)
        self._program = None

    @rank_zero_only
    def save_merging_weights(self, file_path: str, merging_weights: torch.Tensor):
        """
        Save the merging weights to a file.

        Args:
            file_path (str): The path to save the merging weights.
            merging_weights (torch.Tensor): The merging weights to save.
        """
        if self.fabric.is_global_zero and self.config.get(
            "save_merging_weights", False
        ):
            if isinstance(file_path, str) and not file_path.startswith(("/", ".")):
                # if the file path is not absolute or relative to current working directory, save it in the log directory
                save_path = os.path.join(self.log_dir, file_path)
            else:
                save_path = file_path
            log.info(f"saving merging weights to {save_path}.")
            if os.path.dirname(save_path):
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(merging_weights.detach().cpu(), save_path)

    def free_gpu_memory(self, module: LayerWiseMergedModel):
        module.pretrained_model.to("cpu")
        for model in module.task_vectors:
            model.to("cpu")
        del module
        gc.collect()
        torch.cuda.empty_cache()
        log.info(get_memory_usage("after freeing memory, the memory usage of GPU is:"))

    def update_datasets(self, datasets):
        """
        for evary epoch of local adamerging, we only use the data set corresponding to the model involved in the fusion
        """
        num_datasets = len(datasets)
        datasets_copy = datasets.copy()
        if self.config.topo == "ring":
            for i in range(num_datasets):
                datasets[i] = (
                    datasets_copy[i]
                    .union(datasets_copy[(i + 1) % num_datasets])
                    .union(datasets_copy[(i - 1) % num_datasets])
                )
        elif "rotate" in self.config.topo:
            number = self.config.topo.split("_")[1]
            for i in range(num_datasets):
                datasets[i] = datasets_copy[i]
                for j in range(0, int(number)):
                    datasets[i] = datasets[i].union(
                        datasets_copy[(i + j + 1) % num_datasets]
                    )
        return datasets

    def run(self, modelpool: ModelPool):
        """
        Run the Layer-Wise AdaMerging Algorithm.

        This method constructs the wrapped model and performs test-time adaptation if necessary.

        Args:
            modelpool (ModelPool): The model pool containing the pretrained and fine-tuned models.

        Returns:
            LayerWiseMergedModel: The merged model after test-time adaptation.
        """
        log.info("Fusing models using layer-wise adaptive merging.")
        self.modelpool = modelpool
        self.log_hyperparams(self.config)
        self.num_finetuned_models = len(modelpool.model_names)
        datasets = [{dataset} for dataset in modelpool.model_names]

        with self.profile("construct the wrapped model"):
            model_scheduler = ModelScheduler(
                modelpool=self.modelpool, config=self.config
            )

        if self.config.weights is not None:
            # skip the test-time adaptation
            return module.merge_and_unload()
        else:
            for step_idx in tqdm(
                range(self.config.gossip_max_steps),
                "Gossip merging",
                dynamic_ncols=True,
            ):
                datasets = self.update_datasets(datasets)
                log.info(f"Gossip merging step:, {step_idx}")
                for model_id in tqdm(
                    range(self.num_finetuned_models),
                    "local admerging",
                    dynamic_ncols=True,
                ):
                    if self.config.gossip_skip_adamerging == True:
                        # skip adamerging, only merge
                        with self.profile("construct the local wrapped model"):
                            module = model_scheduler(model_id)
                        log.info(
                            f"skip adamerging, only merge ({modelpool.model_names[model_id]})"
                        )
                        model_scheduler.store_model(module.merge_weights(), model_id)
                        self.free_gpu_memory(module)
                    else:
                        with self.profile("construct the local wrapped model"):
                            module = model_scheduler(model_id)

                        if self.config.improve_dataset == True:
                            log.info(
                                f"improved datasets, the datasets used in this local merging is {datasets[model_id]}"
                            )
                        else:
                            log.info(
                                f"unimproved datasets, the datasets used in this local merging is {modelpool.model_names}"
                            )
                        with self.profile("test-time adaptation"):
                            module = self.test_time_adaptation(
                                module, datasets[model_id]
                            )
                        model_scheduler.store_model(module.merge_weights(), model_id)
                        log.info(
                            get_memory_usage(
                                f"after local merging ({modelpool.model_names[model_id]}), the memory usage of GPU is:"
                            )
                        )
                        self.free_gpu_memory(
                            module
                        )  # simulate distributed GPU memory usage as much as possible

                model_scheduler.update_models()

                if "rotate" in self.config.topo:
                    number = self.config.topo.split("_")[1]
                    if int(number) == 1 and step_idx >= 20:
                        self._program.evaluate_merged_model(
                            self._program.taskpool, model_scheduler.get_final_models()
                        )
                        model_scheduler.move_to("cpu")
                else:
                    if (
                        self.config.accuracy_test_interval != 0
                        and (step_idx + 1) % self.config.accuracy_test_interval == 0
                    ):
                        self._program.evaluate_merged_model(
                            self._program.taskpool, model_scheduler.get_final_models()
                        )
                        model_scheduler.move_to("cpu")
        return model_scheduler.get_final_models()

    def on_test_time_adaptation_start(self):
        """
        Something to do before the test-time adaptation starts. Such as setting up the task-specific heads.
        """
        pass

    @abstractmethod
    def get_shuffled_test_loader_iter(self, task: str) -> DataLoader:
        """
        Loader of test dataset for test-time adaptation. labels are not needed.

        Args:
            task (str): The name of the task.

        Returns:
            DataLoader: The data loader for the test dataset.
        """
        pass

    @abstractmethod
    def compute_logits(self, module, images: Tensor, task: str) -> Tensor:
        """
        Compute the logits for the given images and task.

        Args:
            module: The model module.
            images (Tensor): The input images.
            task (str): The name of the task.

        Returns:
            Tensor: The computed logits.
        """
        pass

    def test_time_adaptation(self, module: LayerWiseMergedModel, datasets):
        """
        Perform test-time adaptation on the merged model.

        This method adapts the merging weights during test-time to improve performance.

        Args:
            module (LayerWiseMergedModel): The merged model.

        Returns:
            LayerWiseMergedModel: The adapted merged model.
        """
        self.on_test_time_adaptation_start()

        # configure optimizer
        if self.config.optimizer == "adam":
            optimizer = torch.optim.Adam([module.merge_weight], lr=self.config.lr)
            print(f"{optimizer=}")
            module, optimizer = self.fabric.setup(module, optimizer)
            log.info(
                get_memory_usage(
                    "after loading models and optimizer, the memory usage of GPU is:"
                )
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")

        module.train()
        module.merge_weights()
        for step_idx in (
            pbar := tqdm(
                range(self.config.max_steps if not self.is_debug_mode else 1),
                ("[DEBUG MODE] " if self.is_debug_mode else "")
                + "AdaMerging Test-time adaptation",
                dynamic_ncols=True,
            )
        ):
            # default behavior for first-order optimizers
            for task in self.modelpool.model_names:
                if self.config.improve_dataset == True and task not in datasets:
                    continue
                with self.profile("data loading"):
                    batch = next(self.get_shuffled_test_loader_iter(task))
                with self.profile("forward pass"):
                    if isinstance(self.modelpool, GPT2ForSequenceClassificationPool):
                        logits = self.compute_logits(module, batch, task)
                    elif isinstance(self.modelpool, CLIPVisionModelPool):
                        logits = self.compute_logits(module, batch[0], task)
                    loss = entropy_loss(logits)
                with self.profile("backward pass"):
                    self.fabric.backward(loss, retain_graph=True)

            with self.profile("optimizer step"):
                optimizer.step()
                optimizer.zero_grad()
            with self.profile("merging weights"):
                module.merge_weights()

            metrics = {
                "train/loss": loss.item(),
                "train/weight_max": module.merge_weight.max().item(),
                "train/weight_min": module.merge_weight.min().item(),
                "train/weight_mean": module.merge_weight.mean().item(),
            }
            self.fabric.log_dict(metrics, step=step_idx)
            pbar.set_postfix(metrics)

        self.print_profile_summary()
        del optimizer
        gc.collect()
        torch.cuda.empty_cache()
        return module
