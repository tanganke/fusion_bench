"""
Defense-Aware Task-wise & Layer-wise Concrete AdaMerging for CLIP ViT models

Examples:

```bash
fusion_bench \
    fabric_logger.name= \
    method=clip_safe_concrete_task_wise_adamerging \
    modelpool= \
    taskpool=
```

```bash
fusion_bench \
    fabric_logger.name= \
    method=clip_safe_concrete_layer_wise_adamerging \
    modelpool= \
    taskpool=
```
"""

import logging
import os
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.autonotebook import tqdm

from fusion_bench.compat.method import ModelFusionAlgorithm
from fusion_bench.compat.modelpool import to_modelpool
from fusion_bench.compat.modelpool.huggingface_clip_vision import (
    HuggingFaceClipVisionPool,
)
from fusion_bench.method.adamerging.entropy_loss import entropy_loss
from fusion_bench.mixins import CLIPClassificationMixin
from fusion_bench.mixins.simple_profiler import SimpleProfilerMixin
from fusion_bench.models.masks import MaskModel, mask_sparsity
from fusion_bench.models.wrappers.layer_wise_fusion import (
    LayerWiseMergedModel,
    get_layer_wise_weights,
)
from fusion_bench.models.wrappers.task_wise_fusion import (
    TaskWiseMergedModel,
    get_task_wise_weights,
)
from fusion_bench.utils.dtype import parse_dtype
from fusion_bench.utils.parameters import print_parameters

log = logging.getLogger(__name__)


class ConcreteSafeTaskWiseAdaMergingForCLIP(
    CLIPClassificationMixin,
    SimpleProfilerMixin,
    ModelFusionAlgorithm,
):
    @torch.no_grad()
    def setup_models(self):
        config = self.config
        self.merge_dtype = parse_dtype(config.get("merge_dtype", None))
        modelpool = self.modelpool
        # Load the pretrained model
        pretrained_model = modelpool.load_model("_pretrained_")

        # construct PGE mask model
        mask_model = MaskModel(
            pretrained_model,
            ignore_untrained_params=True,
            parameter_type="logits",
        )
        if self.merge_dtype is not None:
            mask_model.to(self.merge_dtype)
        mask_model.fill_(self.config.initial_logits)
        # TODO: ablation study for the initialization of mask model
        # for param in mask_model.parameters():
        #     param.data = param + 0.1 * torch.randn_like(param)
        print("Summary of mask model:")
        print_parameters(mask_model)

        # Load the fine-tuned models
        finetuned_models = [
            modelpool.load_model(name) for name in modelpool.model_names
        ]

        task_wise_weight = get_task_wise_weights(
            num_models=len(modelpool.model_names),
            init_values=self.config.scaling_factor,
        )
        self.init_task_wise_weight = deepcopy(task_wise_weight)

        # create a warpped model
        module = TaskWiseMergedModel(
            task_wise_weight=task_wise_weight,
            pretrained_model=pretrained_model,
            finetuned_models=finetuned_models,
            clamp_weights=self.config.clamp_weights,
            tie_weights=self.config.tie_weights,
            strict=self.config.strict,
            task_vector_dtype=self.merge_dtype,
        )

        self.pertubed_model = nn.Module()
        self.pertubed_model.perturbed_input = nn.Parameter(
            torch.zeros([len(modelpool.model_names), 3, 224, 224]), requires_grad=True
        )
        return module, mask_model

    def train_mask(self, module: TaskWiseMergedModel, mask_model: MaskModel):
        config = self.config
        self.init_task_wise_weight = self.to_device(self.init_task_wise_weight)

        # configure optimizer
        lr_scheduler = None
        if self.config.optimizer == "adam":

            ### for merge_weight
            base_optimizer = torch.optim.Adam(
                [module.merge_weight], lr=self.config.base_lr
            )
            module, base_optimizer = self.fabric.setup(module, base_optimizer)

            ### for mask
            optimizer = torch.optim.Adam(mask_model.parameters(), lr=self.config.lr)
            mask_model, optimizer = self.fabric.setup(mask_model, optimizer)

            ### for perturbed noise
            batch_opt_adv = torch.optim.Adam(
                params=self.pertubed_model.parameters(), lr=self.config.adv_lr
            )
            self.pertubed_model, batch_opt_adv = self.fabric.setup(
                self.pertubed_model, batch_opt_adv
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")

        module.train()
        mask_model.train()
        self.pertubed_model.train()
        for step_idx in (
            pbar := tqdm(
                range(self.config.max_steps if not self.is_debug_mode else 5),
                ("[DEBUG MODE] " if self.is_debug_mode else "")
                + "Concrete Safe AdaMerging Meta-Learn Mask (1/2)",
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

                # for inner optimization, we do not optimize the mask, so we detach it
                module.merge_weights(
                    task_vector_mask={name: m.detach() for name, m in mask.items()}
                )

            # ------ inner optimization goes here ------

            ### (1)optimize the merging weight
            module.merge_weight.data = deepcopy(self.init_task_wise_weight)
            total_loss = None
            for task in self.modelpool.model_names:
                with self.profile("data loading"):
                    batch = next(self.get_shuffled_test_loader_iter(task))
                    # NOTE: The labels are not allowed to be used during test-time adaptation
                    images = batch[0]

                with self.profile("forward pass"):
                    logits = self.compute_logits(module, images, task)
                    loss = entropy_loss(logits)
                    total_loss = loss if total_loss is None else total_loss + loss

            with self.profile("compute grad"):
                self.fabric.backward(total_loss)

            with self.profile("base optimizer step"):
                base_optimizer.step()
                base_optimizer.zero_grad()

            with self.profile("merge weights"):
                module.merge_weights(task_vector_mask=mask)

            # (2)noise optimization based on the merging model

            # detach merged state_dict
            merged_state_dict = module._merged_state_dict
            detached_merged_state_dict = {
                k: p.detach() for k, p in merged_state_dict.items()
            }
            module._merged_state_dict = detached_merged_state_dict

            total_loss = None
            for task_idx, task in enumerate(self.modelpool.model_names):
                with self.profile("data loading"):
                    batch = next(self.get_shuffled_test_loader_iter(task))
                    # NOTE: The labels are not allowed to be used during test-time adaptation
                    images = batch[0]
                    perturbed_images = (
                        images + self.pertubed_model.perturbed_input[task_idx]
                    )
                    combined_images = torch.cat((images, perturbed_images), dim=0)

                with self.profile("forward pass"):
                    combined_logits = self.compute_logits(module, combined_images, task)
                    logits = combined_logits[: images.size(0)]
                    logits_adv = combined_logits[images.size(0) :]
                    ori_label = torch.argmax(logits, axis=1).long()
                    loss = torch.mean(
                        -F.cross_entropy(logits_adv, ori_label, reduction="mean")
                    )
                    total_loss = loss if total_loss is None else total_loss + loss

            with self.profile("compute grad"):
                self.fabric.backward(total_loss)

            with self.profile("batch_opt_adv optimizer step"):
                batch_opt_adv.step()
                batch_opt_adv.zero_grad()

            # (3)mask optimization
            total_loss = None
            module._merged_state_dict = merged_state_dict

            for task_idx, task in enumerate(self.modelpool.model_names):
                with self.profile("data loading"), torch.no_grad():
                    batch = next(self.get_shuffled_test_loader_iter(task))
                    # NOTE: The labels are not allowed to be used during test-time adaptation
                    images = batch[0]
                    perturbed_images = (
                        images + self.pertubed_model.perturbed_input[task_idx]
                    )
                    perturbed_images = torch.clamp(perturbed_images, min=0, max=1)
                    combined_images = torch.cat((images, perturbed_images), dim=0)

                with self.profile("forward pass"):
                    combined_logits = self.compute_logits(module, combined_images, task)
                    logits = combined_logits[: images.size(0)]
                    logits_adv = combined_logits[images.size(0) :]

                    # # ### regu1
                    # ori_label = torch.argmax(logits, axis=1).long()
                    # loss_nat = entropy_loss(logits)
                    # loss_regu = torch.mean(-F.cross_entropy(logits_adv, ori_label, reduction='mean'))

                    ### regu2
                    loss_regu = entropy_loss(logits_adv)
                    loss_nat = entropy_loss(logits)

                    loss = loss_nat + self.config.adv_weight * loss_regu
                    total_loss = loss if total_loss is None else total_loss + loss

            with self.profile("compute grad"):
                self.fabric.backward(total_loss)

            with self.profile("optimizer step"):
                optimizer.step()
                optimizer.zero_grad()

                if lr_scheduler is not None:
                    lr_scheduler.step()

            # metrics.update({"train/loss": loss.item()})
            metrics.update(
                {
                    "train/loss": loss.item(),
                    "loss_nat": loss_nat.item(),
                    "loss_regu": loss_regu.item(),
                }
            )
            self.fabric.log_dict(metrics, step=step_idx)
            pbar.set_postfix(metrics)
            self.print_profile_summary()

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

    def run_adamerging(self, module: TaskWiseMergedModel, mask):
        module.merge_weight.data = deepcopy(self.init_task_wise_weight)
        base_optimizer = torch.optim.Adam(
            [module.merge_weight], lr=self.config.adamerging_lr
        )
        module, base_optimizer = self.fabric.setup(module, base_optimizer)
        module.train()
        for step_idx in (
            pbar := tqdm(
                range(
                    self.config.max_adamerging_steps if not self.is_debug_mode else 5
                ),
                ("[DEBUG MODE] " if self.is_debug_mode else "")
                + "Concrete AdaMerging AdaMerging (2/2)",
                dynamic_ncols=True,
                disable=not self.fabric.is_global_zero,
            )
        ):
            step_idx = step_idx + self.config.max_steps
            with self.profile("merge weights"):
                module.merge_weights(task_vector_mask=mask)

            metrics = {}
            total_loss = None
            for task in self.modelpool.model_names:
                with self.profile("data loading"):
                    batch = next(self.get_shuffled_test_loader_iter(task))
                    # NOTE: The labels are not allowed to be used during test-time adaptation
                    images = batch[0]
                with self.profile("forward pass"):
                    logits = self.compute_logits(module, images, task)
                    loss = entropy_loss(logits)
                    total_loss = loss if total_loss is None else total_loss + loss

            with self.profile("compute grad"):
                self.fabric.backward(total_loss)

            with self.profile("base optimizer step"):
                base_optimizer.step()
                base_optimizer.zero_grad()

            metrics.update({"train/loss": loss.item()})
            self.fabric.log_dict(metrics, step=step_idx)
            pbar.set_postfix(metrics)

            if (step_idx + 1) % self.config.save_interval == 0:
                with self.profiler.profile("save checkpoint"):
                    save_dir = os.path.join(self.fabric.logger.log_dir, "checkpoints")
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, f"merge_weight_{step_idx}.pt")
                    print(f"saving checkpoint to {save_path}")
                    state = {"merge_weight": module.merge_weight}
                    self.fabric.save(save_path, state)

                    # Create or update a symbolic link to the latest checkpoint
                    if self.fabric.is_global_zero:
                        symlink_path = os.path.join(
                            save_dir, "merge_weight_latest_checkpoint.pt"
                        )
                        if os.path.exists(symlink_path):
                            os.remove(symlink_path)
                        os.link(os.path.abspath(save_path), symlink_path)

                self.print_profile_summary()
        return module

    def run(self, modelpool: HuggingFaceClipVisionPool):
        self.modelpool = to_modelpool(modelpool)
        config = self.config
        self.log_hyperparams(config, filename="method_config.yaml")

        with self.profile("setup models"):
            module, mask_model = self.setup_models()
            mask_model: MaskModel = self.fabric.to_device(mask_model)
            module: TaskWiseMergedModel = self.fabric.to_device(module)
            self.pertubed_model = self.fabric.to_device(self.pertubed_model)
            self.setup_zero_shot_classification_head()

        if config.mask_checkpoint is None:
            self.train_mask(module=module, mask_model=mask_model)
        else:
            if self.fabric.is_global_zero:
                print("loading mask from checkpoint", config.mask_checkpoint)
            self.fabric.load(config.mask_checkpoint, {"model": mask_model})

        # run adamerging
        with torch.no_grad():
            mask = mask_model.sample_mask(
                mask_type=config.eval_mask_type,
                temperature=config.temperature,
            )
            # rescale mask
            for name, m in mask.items():
                mask[name] = m / torch.mean(m)
        module = self.run_adamerging(module, mask=mask)

        with torch.no_grad():
            model = module.merge_and_unload(mask)
        return model


class ConcreteSafeLayerWiseAdaMergingForCLIP(
    CLIPClassificationMixin,
    SimpleProfilerMixin,
    ModelFusionAlgorithm,
):
    @torch.no_grad()
    def setup_models(self):
        modelpool = self.modelpool
        self.merge_dtype = parse_dtype(config.get("merge_dtype", None))
        # Load the pretrained model
        pretrained_model = modelpool.load_model("_pretrained_")

        # construct PGE mask model
        mask_model = MaskModel(
            pretrained_model,
            ignore_untrained_params=True,
            parameter_type="logits",
        )
        if self.merge_dtype is not None:
            mask_model.to(self.merge_dtype)
        mask_model.fill_(self.config.initial_logits)
        # TODO: ablation study for the initialization of mask model
        # for param in mask_model.parameters():
        #     param.data = param + 0.1 * torch.randn_like(param)
        print("Summary of mask model:")
        print_parameters(mask_model)

        # Load the fine-tuned models
        finetuned_models = [
            modelpool.load_model(name) for name in modelpool.model_names
        ]

        layer_wise_weight = get_layer_wise_weights(
            num_models=len(modelpool.model_names),
            num_layers=len(
                tuple(filter(lambda p: p.requires_grad, pretrained_model.parameters()))
            ),
            init_values=self.config.scaling_factor,
        )
        self.init_layer_wise_weight = deepcopy(layer_wise_weight)

        # create a warpped model
        module = LayerWiseMergedModel(
            layer_wise_weight=layer_wise_weight,
            pretrained_model=pretrained_model,
            finetuned_models=finetuned_models,
            clamp_weights=self.config.clamp_weights,
            tie_weights=self.config.tie_weights,
            strict=self.config.strict,
            layer_vector_dtype=self.merge_dtype,
        )

        self.pertubed_model = nn.Module()
        self.pertubed_model.perturbed_input = nn.Parameter(
            torch.zeros([len(modelpool.model_names), 3, 224, 224]), requires_grad=True
        )
        return module, mask_model

    def train_mask(self, module: LayerWiseMergedModel, mask_model: MaskModel):
        config = self.config
        self.init_layer_wise_weight = self.to_device(self.init_layer_wise_weight)

        # configure optimizer
        lr_scheduler = None
        if self.config.optimizer == "adam":
            base_optimizer = torch.optim.Adam(
                [module.merge_weight], lr=self.config.base_lr
            )
            optimizer = torch.optim.Adam(mask_model.parameters(), lr=self.config.lr)
            print(f"{optimizer=}")
            # TODO: ablation study for the learning rate scheduler. It should yield similar results.
            # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            #     optimizer, self.config.max_steps, eta_min=0.1
            # )
            module, base_optimizer = self.fabric.setup(module, base_optimizer)
            mask_model, optimizer = self.fabric.setup(mask_model, optimizer)

            batch_opt_adv = torch.optim.Adam(
                params=self.pertubed_model.parameters(), lr=self.config.adv_lr
            )
            self.pertubed_model, batch_opt_adv = self.fabric.setup(
                self.pertubed_model, batch_opt_adv
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")

        module.train()
        mask_model.train()
        # self.pertubed_model.train()
        for step_idx in (
            pbar := tqdm(
                range(self.config.max_steps if not self.is_debug_mode else 5),
                ("[DEBUG MODE] " if self.is_debug_mode else "")
                + "Concrete Safe AdaMerging Meta-Learn Mask (1/2)",
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

                # for inner optimization, we do not optimize the mask, so we detach it
                module.merge_weights(
                    task_vector_mask={name: m.detach() for name, m in mask.items()}
                )

            # ------ inner optimization goes here ------
            module.merge_weight.data = deepcopy(self.init_layer_wise_weight)
            total_loss = None
            for task in self.modelpool.model_names:
                with self.profile("data loading"):
                    batch = next(self.get_shuffled_test_loader_iter(task))
                    # NOTE: The labels are not allowed to be used during test-time adaptation
                    images = batch[0]
                with self.profile("forward pass"):
                    logits = self.compute_logits(module, images, task)
                    loss = entropy_loss(logits)
                    total_loss = loss if total_loss is None else total_loss + loss

            with self.profile("compute grad"):
                self.fabric.backward(total_loss)

            with self.profile("base optimizer step"):
                base_optimizer.step()
                base_optimizer.zero_grad()

            with self.profile("merge weights"):
                module.merge_weights(task_vector_mask=mask)

            # ------------------------------------------

            # (2)noise optimization based on the merging model

            # detach merged state_dict
            merged_state_dict = module._merged_state_dict
            detached_merged_state_dict = {
                k: p.detach() for k, p in merged_state_dict.items()
            }
            module._merged_state_dict = detached_merged_state_dict

            total_loss = None
            for task_idx, task in enumerate(self.modelpool.model_names):
                with self.profile("data loading"):
                    batch = next(self.get_shuffled_test_loader_iter(task))
                    # NOTE: The labels are not allowed to be used during test-time adaptation
                    images = batch[0]
                    perturbed_images = (
                        images + self.pertubed_model.perturbed_input[task_idx]
                    )
                    combined_images = torch.cat((images, perturbed_images), dim=0)

                with self.profile("forward pass"):
                    combined_logits = self.compute_logits(module, combined_images, task)
                    logits = combined_logits[: images.size(0)]
                    logits_adv = combined_logits[images.size(0) :]
                    ori_label = torch.argmax(logits, axis=1).long()
                    loss = torch.mean(
                        -F.cross_entropy(logits_adv, ori_label, reduction="mean")
                    )
                    total_loss = loss if total_loss is None else total_loss + loss

            with self.profile("compute grad"):
                self.fabric.backward(total_loss)

            with self.profile("batch_opt_adv optimizer step"):
                batch_opt_adv.step()
                batch_opt_adv.zero_grad()

            # (3)mask optimization
            total_loss = None
            module._merged_state_dict = merged_state_dict

            for task_idx, task in enumerate(self.modelpool.model_names):
                with self.profile("data loading"), torch.no_grad():
                    batch = next(self.get_shuffled_test_loader_iter(task))
                    # NOTE: The labels are not allowed to be used during test-time adaptation
                    images = batch[0]
                    perturbed_images = (
                        images + self.pertubed_model.perturbed_input[task_idx]
                    )
                    perturbed_images = torch.clamp(perturbed_images, min=0, max=1)
                    combined_images = torch.cat((images, perturbed_images), dim=0)

                with self.profile("forward pass"):
                    combined_logits = self.compute_logits(module, combined_images, task)
                    logits = combined_logits[: images.size(0)]
                    logits_adv = combined_logits[images.size(0) :]

                    # # ### regu1
                    # ori_label = torch.argmax(logits, axis=1).long()
                    # loss_nat = entropy_loss(logits)
                    # loss_regu = torch.mean(-F.cross_entropy(logits_adv, ori_label, reduction='mean'))

                    ### regu2
                    loss_regu = entropy_loss(logits_adv)
                    loss_nat = entropy_loss(logits)

                    loss = loss_nat + self.config.adv_weight * loss_regu
                    total_loss = loss if total_loss is None else total_loss + loss

            with self.profile("compute grad"):
                self.fabric.backward(total_loss)

            with self.profile("optimizer step"):
                optimizer.step()
                optimizer.zero_grad()

                if lr_scheduler is not None:
                    lr_scheduler.step()

            # metrics.update({"train/loss": loss.item()})
            metrics.update(
                {
                    "train/loss": loss.item(),
                    "loss_nat": loss_nat.item(),
                    "loss_regu": loss_regu.item(),
                }
            )
            self.fabric.log_dict(metrics, step=step_idx)
            pbar.set_postfix(metrics)
            self.print_profile_summary()

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

    def run_adamerging(self, module: LayerWiseMergedModel, mask):
        module.merge_weight.data = deepcopy(self.init_layer_wise_weight)
        base_optimizer = torch.optim.Adam(
            [module.merge_weight], lr=self.config.adamerging_lr
        )
        module, base_optimizer = self.fabric.setup(module, base_optimizer)
        module.train()
        for step_idx in (
            pbar := tqdm(
                range(
                    self.config.max_adamerging_steps if not self.is_debug_mode else 5
                ),
                ("[DEBUG MODE] " if self.is_debug_mode else "")
                + "Concrete AdaMerging AdaMerging (2/2)",
                dynamic_ncols=True,
                disable=not self.fabric.is_global_zero,
            )
        ):
            step_idx = step_idx + self.config.max_steps
            with self.profile("merge weights"):
                module.merge_weights(task_vector_mask=mask)

            metrics = {}
            total_loss = None
            for task in self.modelpool.model_names:
                with self.profile("data loading"):
                    batch = next(self.get_shuffled_test_loader_iter(task))
                    # NOTE: The labels are not allowed to be used during test-time adaptation
                    images = batch[0]
                with self.profile("forward pass"):
                    logits = self.compute_logits(module, images, task)
                    loss = entropy_loss(logits)
                    total_loss = loss if total_loss is None else total_loss + loss

            with self.profile("compute grad"):
                self.fabric.backward(total_loss)

            with self.profile("base optimizer step"):
                base_optimizer.step()
                base_optimizer.zero_grad()

            metrics.update({"train/loss": loss.item()})
            self.fabric.log_dict(metrics, step=step_idx)
            pbar.set_postfix(metrics)

            if (step_idx + 1) % self.config.save_interval == 0:
                with self.profiler.profile("save checkpoint"):
                    save_dir = os.path.join(self.fabric.logger.log_dir, "checkpoints")
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, f"merge_weight_{step_idx}.pt")
                    print(f"saving checkpoint to {save_path}")
                    state = {"merge_weight": module.merge_weight}
                    self.fabric.save(save_path, state)

                    # Create or update a symbolic link to the latest checkpoint
                    if self.fabric.is_global_zero:
                        symlink_path = os.path.join(
                            save_dir, "merge_weight_latest_checkpoint.pt"
                        )
                        if os.path.exists(symlink_path):
                            os.remove(symlink_path)
                        os.link(os.path.abspath(save_path), symlink_path)

                self.print_profile_summary()
        return module

    # def run_adamerging(self, module: LayerWiseMergedModel, mask):
    #     module.merge_weight.data = deepcopy(self.init_layer_wise_weight)
    #     base_optimizer = torch.optim.Adam(
    #         [module.merge_weight], lr=self.config.adamerging_lr
    #     )
    #     module, base_optimizer = self.fabric.setup(module, base_optimizer)
    #     module.train()
    #     for step_idx in (
    #         pbar := tqdm(
    #             range(
    #                 self.config.max_adamerging_steps if not self.is_debug_mode else 5
    #             ),
    #             ("[DEBUG MODE] " if self.is_debug_mode else "")
    #             + "Concrete AdaMerging AdaMerging (2/2)",
    #             dynamic_ncols=True,
    #             disable=not self.fabric.is_global_zero,
    #         )
    #     ):
    #         step_idx = step_idx + self.config.max_steps
    #         with self.profile("merge weights"):
    #             module.merge_weights(task_vector_mask=mask)

    #         metrics = {}
    #         total_loss = None
    #         for task_idx, task in enumerate(self.modelpool.model_names):
    #             with self.profile("data loading"), torch.no_grad():
    #                 batch = next(self.get_shuffled_test_loader_iter(task))
    #                 # NOTE: The labels are not allowed to be used during test-time adaptation
    #                 images = batch[0]
    #                 perturbed_images = images + self.pertubed_model.perturbed_input[task_idx]
    #                 perturbed_images = torch.clamp(perturbed_images, min=0, max=1)
    #                 combined_images = torch.cat((images, perturbed_images), dim=0)

    #             with self.profile("forward pass"):
    #                 combined_logits = self.compute_logits(module, combined_images, task)
    #                 logits = combined_logits[:images.size(0)]
    #                 logits_adv = combined_logits[images.size(0):]

    #                 # # ### regu1
    #                 # ori_label = torch.argmax(logits, axis=1).long()
    #                 # loss_nat = entropy_loss(logits)
    #                 # loss_regu = torch.mean(-F.cross_entropy(logits_adv, ori_label, reduction='mean'))

    #                 ### regu2
    #                 loss_regu = entropy_loss(logits_adv)
    #                 loss_nat = entropy_loss(logits)

    #                 loss = loss_nat + self.config.adv_weight*loss_regu
    #                 total_loss = loss if total_loss is None else total_loss + loss
    #                 metrics.update({"train/loss": loss.item(),"loss_nat": loss_nat.item(),"loss_regu": loss_regu.item()})

    #         self.fabric.log_dict(metrics, step=step_idx)
    #         pbar.set_postfix(metrics)
    #         self.print_profile_summary()

    #         if (step_idx + 1) % self.config.save_interval == 0:
    #             with self.profiler.profile("save checkpoint"):
    #                 save_dir = os.path.join(self.fabric.logger.log_dir, "checkpoints")
    #                 if not os.path.exists(save_dir):
    #                     os.makedirs(save_dir, exist_ok=True)
    #                 save_path = os.path.join(save_dir, f"merge_weight_{step_idx}.pt")
    #                 print(f"saving checkpoint to {save_path}")
    #                 state = {"merge_weight": module.merge_weight}
    #                 self.fabric.save(save_path, state)

    #                 # Create or update a symbolic link to the latest checkpoint
    #                 if self.fabric.is_global_zero:
    #                     symlink_path = os.path.join(
    #                         save_dir, "merge_weight_latest_checkpoint.pt"
    #                     )
    #                     if os.path.exists(symlink_path):
    #                         os.remove(symlink_path)
    #                     os.link(os.path.abspath(save_path), symlink_path)

    #             self.print_profile_summary()
    #     return module

    def run(self, modelpool: HuggingFaceClipVisionPool):
        self.modelpool = to_modelpool(modelpool)
        config = self.config
        self.log_hyperparams(config, filename="method_config.yaml")

        with self.profile("setup models"):
            module, mask_model = self.setup_models()
            mask_model: MaskModel = self.fabric.to_device(mask_model)
            module: LayerWiseMergedModel = self.fabric.to_device(module)
            self.pertubed_model = self.fabric.to_device(self.pertubed_model)
            self.setup_zero_shot_classification_head()

        if config.mask_checkpoint is None:
            self.train_mask(module=module, mask_model=mask_model)
        else:
            if self.fabric.is_global_zero:
                print("loading mask from checkpoint", config.mask_checkpoint)
            self.fabric.load(config.mask_checkpoint, {"model": mask_model})

        # run adamerging
        with torch.no_grad():
            mask = mask_model.sample_mask(
                mask_type=config.eval_mask_type,
                temperature=config.temperature,
            )
            # rescale mask
            for name, m in mask.items():
                mask[name] = m / torch.mean(m)
        module = self.run_adamerging(module, mask=mask)

        with torch.no_grad():
            model = module.merge_and_unload(mask)
        return model
