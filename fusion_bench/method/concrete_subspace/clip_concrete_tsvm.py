import logging
import os
from copy import deepcopy
from typing import TYPE_CHECKING, Dict, Iterable, List, Literal, Optional

import torch
from omegaconf import DictConfig
from tqdm import tqdm

from fusion_bench import (
    BaseAlgorithm,
    OpenCLIPClassificationMixin,
    OpenCLIPVisionModelPool,
    SimpleProfilerMixin,
    StateDictType,
    auto_register_config,
    get_rankzero_logger,
    instantiate,
)
from fusion_bench.method.adamerging.entropy_loss import entropy_loss
from fusion_bench.method.task_singular_vector import TaskSingularVectorMerging
from fusion_bench.method.task_singular_vector.utils import (
    TSVM_utils,
    check_parameterNamesMatch,
    check_state_dicts_equal,
    state_dict_to_vector,
    vector_to_state_dict,
)
from fusion_bench.models.masks import MaskModel, mask_sparsity
from fusion_bench.models.open_clip import (
    ClassificationHead,
    ImageClassifier,
    ImageEncoder,
)
from fusion_bench.models.wrappers.task_wise_fusion import (
    TaskWiseMergedModel,
    get_task_wise_weights,
)
from fusion_bench.utils.devices import clear_cuda_cache
from fusion_bench.utils.dtype import parse_dtype
from fusion_bench.utils.parameters import print_parameters, print_trainable_parameters
from fusion_bench.utils.rich_utils import print_config_yaml
from fusion_bench.utils.state_dict_arithmetic import (
    _validate_state_dict_same_keys,
    state_dict_add,
    state_dict_hadamard_product,
    state_dict_mul,
    state_dict_sub,
)

log = get_rankzero_logger(__name__)


@auto_register_config
class ConcreteTSVMForOpenCLIP(
    OpenCLIPClassificationMixin,
    SimpleProfilerMixin,
    BaseAlgorithm,
):
    def __init__(
        self,
        dataloader_kwargs: DictConfig,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        max_steps: int,
        save_interval: int,
        initial_logits: float,
        temperature: float,
        eval_mask_type: Literal["continuous", "discrete"],
        mask_checkpoint: Optional[str],
        merge_dtype: str,
        clamp_weights: bool,
        tie_weights: bool,
        strict: bool,
        skip_training: bool,
        # === TSVM parameters ===
        exclude_keys: Optional[List[str]],
        alpha: float,
        return_single_task_models: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not return_single_task_models:
            log.warning("return_single_task_models is forced to be True here.")
            self.return_single_task_models = True

    @torch.no_grad()
    def setup_models(self):
        """
        load the pre-trained model, task vectors, and construct the mask model.
        """
        merge_dtype = parse_dtype(self.merge_dtype)
        modelpool = self.modelpool

        # load the pre-trained model
        pretrained_model = modelpool.load_pretrained_model()
        self.set_clip_processor(stage="test", processor=pretrained_model.val_preprocess)

        # constrcute mask model
        mask_model = MaskModel(
            pretrained_model, ignore_untrained_params=True, parameter_type="logits"
        )
        if merge_dtype is not None:
            mask_model.to(merge_dtype)
        mask_model.fill_(self.initial_logits)

        if self.fabric.is_global_zero:
            print("summary of mask model:")
            print_parameters(mask_model)

        if self.fabric.is_global_zero:
            tsvm_algo = TaskSingularVectorMerging(
                alpha=self.alpha,
                exclude_keys=self.exclude_keys,
                return_single_task_models=self.return_single_task_models,
            )
            tsvm_algo._fabric_instance = self.fabric
            models = tsvm_algo.run(modelpool)

            finetuned_models = [models[name] for name in modelpool.model_names]

            task_wise_weight = get_task_wise_weights(
                num_models=len(modelpool.model_names),
                init_values=self.alpha,
            )

            # create a wrapped model
            module = TaskWiseMergedModel(
                task_wise_weight=task_wise_weight,
                pretrained_model=pretrained_model,
                finetuned_models=finetuned_models,
                clamp_weights=self.clamp_weights,
                tie_weights=self.tie_weights,
                strict=self.strict,
                task_vector_dtype=merge_dtype,
            )
            module = module.to(dtype=merge_dtype)

            print("trainable parameter summary of merged model (TaskWiseMergedModel):")
            print_trainable_parameters(module)
        else:
            module = None

        with torch.no_grad():
            self.fabric.barrier()
            module = self.fabric.broadcast(module, src=0)

        return module, mask_model

    def train_mask(self, module: TaskWiseMergedModel, mask_model: MaskModel):
        """
        Train the mask model using the provided module.

        This method configures the optimizer, sets up the mask model, and performs test-time adaptation to train the mask model.

        Args:
            module (TaskWiseMergedModel): The wrapped model with task-wise weights.
            mask_model (MaskModel): The mask model to be trained.
        """
        config = self.config
        merge_dtype = parse_dtype(self.merge_dtype)
        log.info(f"Using merge dtype: {merge_dtype}")

        optimizer: "torch.optim.Optimizer" = instantiate(
            self.optimizer,
            params=filter(lambda p: p.requires_grad, mask_model.parameters()),
        )
        print(f"{optimizer=}")
        if self.lr_scheduler is not None:
            lr_scheduler = instantiate(
                self.lr_scheduler,
                optimizer=optimizer,
            )
            print(f"{lr_scheduler=}")
        else:
            lr_scheduler = None

        log.info("Setup models and optimizer with Fabric.")
        mask_model, optimizer = self.fabric.setup(mask_model, optimizer)

        log.info("Move the merged module to the correct device and disable gradients.")
        module.requires_grad_(False)
        module.to(mask_model.device)

        mask_model.train()
        optimizer.zero_grad()
        for step_idx in (
            pbar := tqdm(
                range(self.config.max_steps if not self.is_debug_mode else 5),
                ("[DEBUG MODE] " if self.is_debug_mode else "")
                + "Concrete TSVM Test-Time Adaptation",
                dynamic_ncols=True,
                disable=not self.fabric.is_global_zero,
            )
        ):
            metrics = {}
            # sample a shared mask and merge weights
            with self.profile("sample mask"):
                mask = mask_model.sample_mask(
                    mask_type="continuous", temperature=config.temperature
                )
                metrics["train/sparsity"] = mask_sparsity(mask)
            with self.profile("merge weights"):
                # rescale mask
                for name, m in mask.items():
                    mask[name] = m / torch.mean(m)
                module.merge_weights(task_vector_mask=mask)

            # ------ inner optimization goes here ------
            # NOTE:
            #   Because the algorithmic parameters of TSVM are assumed to be chosen on a validation test
            #   set, we do not need to perform inner optimization here. So here we skip the inner optimization step.
            # ------------------------------------------

            total_loss = None
            for task in self.modelpool.model_names:
                with self.profile("data loading"):
                    batch = next(self.get_shuffled_test_loader_iter(task))
                    # NOTE: The labels are not allowed to be used during test-time adaptation
                    images = batch[0].to(dtype=merge_dtype)
                with self.profile("forward pass"):
                    logits = self.compute_logits(module, images, task)
                    loss = entropy_loss(logits)
                    total_loss = loss if total_loss is None else total_loss + loss

            with self.profile("compute grad"):
                self.fabric.backward(total_loss)

            with self.profile("optimizer step"):
                optimizer.step()
                optimizer.zero_grad()

                if lr_scheduler is not None:
                    lr_scheduler.step()

            metrics.update({"train/loss": loss.item()})
            self.fabric.log_dict(metrics, step=step_idx)
            pbar.set_postfix(metrics)

            if (step_idx + 1) % self.config.save_interval == 0:
                with self.profiler.profile("save checkpoint"):
                    save_dir = os.path.join(self.fabric.logger.log_dir, "checkpoints")
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, f"mask_steps_{step_idx}.pt")
                    print(f"saving checkpoint to {save_path}")
                    state = {"model": mask_model}
                    self.fabric.save(save_path, state)

                    # Create or update a symbolic link to the latest checkpoint
                    if self.fabric.is_global_zero:
                        symlink_path = os.path.join(save_dir, "latest_checkpoint.pt")
                        if os.path.exists(symlink_path):
                            os.remove(symlink_path)
                        os.link(os.path.abspath(save_path), symlink_path)

                self.print_profile_summary()

    def run(self, modelpool: OpenCLIPVisionModelPool):
        self.modelpool = modelpool
        merge_dtype = parse_dtype(self.merge_dtype)

        with self.profile("setup models"):
            module, mask_model = self.setup_models()
            self.setup_zero_shot_classification_head(freeze=True, dtype=merge_dtype)

        if self.mask_checkpoint is None:
            if not self.skip_training:
                clear_cuda_cache()
                self.train_mask(module, mask_model=mask_model)
        else:
            if self.fabric.is_global_zero:
                print("loading mask from checkpoint", self.mask_checkpoint)
            self.fabric.load(self.mask_checkpoint, {"model": mask_model})

        with torch.no_grad():
            clear_cuda_cache()
            mask = mask_model.sample_mask(
                mask_type=self.eval_mask_type, temperature=self.temperature
            )
            # rescale mask
            for name, m in mask.items():
                mask[name] = m / torch.mean(m)
            model = module.merge_and_unload(mask)
        return model.to(dtype=torch.float32)
