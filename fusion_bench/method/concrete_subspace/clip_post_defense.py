"""
Post-Defense Methods on the merged models (CLIP ViT)

Examples:

```bash
fusion_bench \
    fabric.loggers.name= \
    method=clip_post_defense_AWM \
    modelpool= \
    taskpool=
```

```bash
fusion_bench \
    fabric.loggers.name= \
    method=clip_post_defense_SAU \
    modelpool= \
    taskpool=
```
"""

import logging
import os
from typing import cast

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
from fusion_bench.tasks.clip_classification import get_classnames_and_templates
from fusion_bench.utils.dtype import parse_dtype
from fusion_bench.utils.parameters import print_parameters

log = logging.getLogger(__name__)


class PostDefenseAWMAlgorithmForCLIP(
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
        merge_model = modelpool.load_model("merge")

        # construct PGE mask model
        mask_model = MaskModel(
            merge_model,
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

        self.pertubed_model = nn.Module()
        self.pertubed_model.perturbed_input = nn.Parameter(
            torch.zeros([len(self.modelpool.config.tta_datasets), 3, 224, 224]),
            requires_grad=True,
        )

        return merge_model, pretrained_model, mask_model

    def train_mask(self, merge_model, pretrained_model, mask_model: MaskModel):
        config = self.config

        # configure optimizer
        lr_scheduler = None
        if self.config.optimizer == "adam":
            optimizer = torch.optim.Adam(mask_model.parameters(), lr=self.config.lr)
            mask_model, optimizer = self.fabric.setup(mask_model, optimizer)

            batch_opt_adv = torch.optim.Adam(
                params=self.pertubed_model.parameters(), lr=self.config.adv_lr
            )
            self.pertubed_model, batch_opt_adv = self.fabric.setup(
                self.pertubed_model, batch_opt_adv
            )

        merge_model.requires_grad_(False)
        pretrained_model.requires_grad_(False)

        mask_model.train()
        optimizer.zero_grad()

        self.pertubed_model.train()
        batch_opt_adv.zero_grad()
        # torch.autograd.set_detect_anomaly(True)

        pretrained_model_dict = pretrained_model.state_dict(keep_vars=True)
        for step_idx in (
            pbar := tqdm(
                range(self.config.max_steps if not self.is_debug_mode else 5),
                ("[DEBUG MODE] " if self.is_debug_mode else "")
                + "clip_post_defense_AWM",
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

                merged_state_dict = merge_model.state_dict(keep_vars=True)
                for name, parameter in merged_state_dict.items():
                    ## (1) mask--directly prune the merged model, the initial logits should be larger than 3
                    # merged_state_dict[name] = merged_state_dict[name]* mask[name]
                    ### (2) mask the task vector, similar to concrete mask, the initial logits can be set as 0
                    merged_state_dict[name] = (
                        merged_state_dict[name] - pretrained_model_dict[name]
                    ) * mask[name] + pretrained_model_dict[name]

            # ------ noise optimization based on the merging model ------
            # detach merged state_dict
            detached_merged_state_dict = {
                k: p.detach() for k, p in merged_state_dict.items()
            }
            merge_model_forward = lambda *args, **kwargs: torch.func.functional_call(
                merge_model, detached_merged_state_dict, args=args, kwargs=kwargs
            )

            total_loss = None
            for task_idx, task in enumerate(
                [c["name"] for c in self.modelpool.config.tta_datasets]
            ):
                with self.profile("data loading"):
                    batch = next(
                        self.get_shuffled_test_loader_iter(task)
                    )  # image ,label
                    # NOTE: The labels are not allowed to be used during test-time adaptation
                    # batch[0],batch[1] = batch[0].to(merge_model.device),batch[1].to(merge_model.device)
                    images = batch[0]
                    perturbed_images = (
                        images + self.pertubed_model.perturbed_input[task_idx]
                    )
                    combined_images = torch.cat((images, perturbed_images), dim=0)

                with self.profile("forward pass"):
                    combined_logits = self.compute_logits(
                        merge_model_forward, combined_images, task
                    )
                    # print(combined_logits.size())
                    num_image = images.size(0)
                    logits, logits_adv = (
                        combined_logits[:num_image],
                        combined_logits[num_image:],
                    )
                    ori_label = torch.argmax(logits, dim=1).long()

                    loss = torch.mean(
                        -F.cross_entropy(logits_adv, ori_label, reduction="mean")
                    )
                    # print(loss)
                    total_loss = loss if total_loss is None else total_loss + loss

            with self.profile("compute grad"):
                self.fabric.backward(total_loss)

            with self.profile("batch_opt_adv optimizer step"):
                batch_opt_adv.step()
                batch_opt_adv.zero_grad()

            # ------ inner optimization goes here ------
            # NOTE:
            #   Because the algorithmic parameters of task arithmetic are assumed to be chosen on a validation test
            #   set, we do not need to perform inner optimization here. So here we skip the inner optimization step.
            # -----------------------------------------
            ### mask optimization
            merge_model_forward = lambda *args, **kwargs: torch.func.functional_call(
                merge_model, merged_state_dict, args=args, kwargs=kwargs
            )
            total_loss = None

            # trigger_norm = self.config.trigger_norm
            # pert = batch_pert * min(1, trigger_norm / torch.sum(torch.abs(batch_pert)))
            # pert = pert.detach()

            for task_idx, task in enumerate(
                [c["name"] for c in self.modelpool.config.tta_datasets]
            ):
                with self.profile("data loading"), torch.no_grad():
                    batch = next(self.get_shuffled_test_loader_iter(task))
                    # NOTE: The labels are not allowed to be used during test-time adaptation
                    images = batch[0]

                    # perturbed_images = images + self.pertubed_model.perturbed_input[task_idx]
                    # perturbed_images = torch.clamp(perturbed_images, min=0, max=1)

                    perturbed_images = torch.clamp(
                        images + self.pertubed_model.perturbed_input[task_idx],
                        min=0,
                        max=1,
                    )
                    combined_images = torch.cat((images, perturbed_images), dim=0)

                with self.profile("forward pass"):
                    combined_logits = self.compute_logits(
                        merge_model_forward, combined_images, task
                    )
                    num_image = images.size(0)
                    logits, logits_adv = (
                        combined_logits[:num_image],
                        combined_logits[num_image:],
                    )

                    loss_nat = entropy_loss(logits)

                    # ### regu1
                    # ori_label = torch.argmax(logits, dim=1).long()
                    # loss_regu = -torch.mean(
                    #     F.cross_entropy(logits_adv, ori_label, reduction="mean")
                    # )

                    ### regu2
                    loss_regu = entropy_loss(logits_adv)

                    loss = loss_nat + self.config.adv_weight * loss_regu
                    total_loss = loss if total_loss is None else total_loss + loss

                    # loss = entropy_loss(logits)
                    # total_loss = loss if total_loss is None else total_loss + loss

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

    def run(self, modelpool: HuggingFaceClipVisionPool):
        self.modelpool = to_modelpool(modelpool)
        config = self.config
        self.log_hyperparams(config, filename="method_config.yaml")

        with self.profile("setup models"):
            merge_model, pretrained_model, mask_model = self.setup_models()
            mask_model: MaskModel = self.fabric.to_device(mask_model)
            merge_model = self.fabric.to_device(merge_model)
            pretrained_model = self.fabric.to_device(pretrained_model)
            self.pertubed_model = self.fabric.to_device(self.pertubed_model)
            self.setup_zero_shot_classification_head(
                task_names=[c["name"] for c in self.modelpool.config.tta_datasets]
            )

        if config.mask_checkpoint is None:
            self.train_mask(
                merge_model=merge_model,
                pretrained_model=pretrained_model,
                mask_model=mask_model,
            )
        else:
            if self.fabric.is_global_zero:
                print("loading mask from checkpoint", config.mask_checkpoint)
            self.fabric.load(config.mask_checkpoint, {"model": mask_model})

        with torch.no_grad():
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            mask = mask_model.sample_mask(
                mask_type=config.eval_mask_type,
                temperature=config.temperature,
            )
            # rescale mask
            for name, m in mask.items():
                mask[name] = m / torch.mean(m)
            pretrained_model_dict = pretrained_model.state_dict(keep_vars=True)
            merged_state_dict = merge_model.state_dict(keep_vars=True)
            for name, parameter in merged_state_dict.items():
                ## (1) mask--directly prune the merged model, the initial logits should be larger than 3
                # merged_state_dict[name] = merged_state_dict[name]* mask[name]
                ### (2) mask the task vector, similar to concrete mask, the initial logits can be set as 0
                merged_state_dict[name] = (
                    merged_state_dict[name] - pretrained_model_dict[name]
                ) * mask[name] + pretrained_model_dict[name]
            merge_model.load_state_dict(merged_state_dict)
        return merge_model


class PostDefenseSAUAlgorithmForCLIP(
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
        merge_model = modelpool.load_model("merge")
        merge_model_ref = modelpool.load_model("merge")

        # construct PGE mask model
        mask_model = MaskModel(
            merge_model,
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

        self.pertubed_model = nn.Module()
        self.pertubed_model.perturbed_input = nn.Parameter(
            torch.zeros([len(self.modelpool.config.tta_datasets), 3, 224, 224]),
            requires_grad=True,
        )

        return merge_model, merge_model_ref, pretrained_model, mask_model

    def train_mask(
        self, merge_model, merge_model_ref, pretrained_model, mask_model: MaskModel
    ):
        config = self.config

        # configure optimizer
        lr_scheduler = None
        if self.config.optimizer == "adam":
            optimizer = torch.optim.Adam(mask_model.parameters(), lr=self.config.lr)
            mask_model, optimizer = self.fabric.setup(mask_model, optimizer)

            batch_opt_adv = torch.optim.Adam(
                params=self.pertubed_model.parameters(), lr=self.config.adv_lr
            )
            self.pertubed_model, batch_opt_adv = self.fabric.setup(
                self.pertubed_model, batch_opt_adv
            )

        merge_model.requires_grad_(False)
        merge_model_ref.requires_grad_(False)
        pretrained_model.requires_grad_(False)

        mask_model.train()
        optimizer.zero_grad()

        self.pertubed_model.train()
        batch_opt_adv.zero_grad()
        # torch.autograd.set_detect_anomaly(True)

        pretrained_model_dict = pretrained_model.state_dict(keep_vars=True)
        for step_idx in (
            pbar := tqdm(
                range(self.config.max_steps if not self.is_debug_mode else 5),
                ("[DEBUG MODE] " if self.is_debug_mode else "")
                + "clip_post_defense_SAU",
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

                merged_state_dict = merge_model.state_dict(keep_vars=True)
                for name, parameter in merged_state_dict.items():
                    ## (1) directy mask/prune the merged model, the initial logits should be larger than 3, without decreasing the acc
                    # merged_state_dict[name] = merged_state_dict[name]* mask[name]
                    ### (2) mask the task vector, similar to concrete mask, the initial logits can be set as 0
                    merged_state_dict[name] = (
                        merged_state_dict[name] - pretrained_model_dict[name]
                    ) * mask[name] + pretrained_model_dict[name]

            # ------ noise optimization based on the merging model ------
            # detach merged state_dict
            detached_merged_state_dict = {
                k: p.detach() for k, p in merged_state_dict.items()
            }
            merge_model_forward = lambda *args, **kwargs: torch.func.functional_call(
                merge_model, detached_merged_state_dict, args=args, kwargs=kwargs
            )

            total_loss = None
            for task_idx, task in enumerate(
                [c["name"] for c in self.modelpool.config.tta_datasets]
            ):
                with self.profile("data loading"):
                    batch = next(
                        self.get_shuffled_test_loader_iter(task)
                    )  # image ,label
                    # NOTE: The labels are not allowed to be used during test-time adaptation
                    # batch[0],batch[1] = batch[0].to(merge_model.device),batch[1].to(merge_model.device)
                    images = batch[0]
                    perturbed_images = (
                        images + self.pertubed_model.perturbed_input[task_idx]
                    )
                    combined_images = torch.cat((images, perturbed_images), dim=0)

                with self.profile("forward pass"):
                    num_image = images.size(0)

                    combined_logits = self.compute_logits(
                        merge_model_forward, combined_images, task
                    )
                    logits, logits_adv = (
                        combined_logits[:num_image],
                        combined_logits[num_image:],
                    )
                    ori_label = torch.argmax(logits, dim=1).long()
                    pert_label = torch.argmax(logits_adv, dim=1).long()

                    combined_logits_ref = self.compute_logits(
                        merge_model_ref, combined_images, task
                    )
                    logits_ref, logits_adv_ref = (
                        combined_logits_ref[:num_image],
                        combined_logits_ref[num_image:],
                    )
                    ori_label_ref = torch.argmax(logits_ref, dim=1).long()
                    pert_label_ref = torch.argmax(logits_adv_ref, dim=1).long()

                    success_attack = pert_label != ori_label
                    success_attack_ref = pert_label_ref != ori_label_ref
                    common_attack = torch.logical_and(
                        success_attack, success_attack_ref
                    )
                    shared_attack = torch.logical_and(
                        common_attack, pert_label == pert_label_ref
                    )

                    # Shared loss
                    # JS divergence version (https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence)
                    p_model = F.softmax(logits_adv, dim=1).clamp(min=1e-8)
                    p_ref = F.softmax(logits_adv_ref, dim=1).clamp(min=1e-8)
                    mix_p = 0.5 * (p_model + p_ref)
                    loss_js = 0.5 * (
                        p_model * p_model.log() + p_ref * p_ref.log()
                    ) - 0.5 * (p_model * mix_p.log() + p_ref * mix_p.log())
                    loss_cross = (
                        loss_js[torch.logical_not(shared_attack)].sum(dim=1).sum()
                        / images.shape[0]
                    )

                    ### maximization perturbation loss
                    ## using the test data without the true label
                    loss_adv = torch.mean(
                        -F.cross_entropy(logits_adv, ori_label, reduction="mean")
                    )

                    loss = self.config.beta1 * loss_adv + self.config.beta2 * loss_cross

                    total_loss = loss if total_loss is None else total_loss + loss

            with self.profile("compute grad"):
                self.fabric.backward(total_loss)

            with self.profile("batch_opt_adv optimizer step"):
                batch_opt_adv.step()
                batch_opt_adv.zero_grad()

            # ------ inner optimization goes here ------
            # NOTE:
            #   Because the algorithmic parameters of task arithmetic are assumed to be chosen on a validation test
            #   set, we do not need to perform inner optimization here. So here we skip the inner optimization step.
            # -----------------------------------------
            ### mask optimization
            merge_model_forward = lambda *args, **kwargs: torch.func.functional_call(
                merge_model, merged_state_dict, args=args, kwargs=kwargs
            )
            total_loss = None

            # trigger_norm = self.config.trigger_norm
            # pert = batch_pert * min(1, trigger_norm / torch.sum(torch.abs(batch_pert)))
            # pert = pert.detach()

            for task_idx, task in enumerate(
                [c["name"] for c in self.modelpool.config.tta_datasets]
            ):
                classnames, templates = get_classnames_and_templates(
                    self.modelpool.get_train_dataset_config(task)["dataset"].name
                )
                num_classes = len(classnames)

                with self.profile("data loading"), torch.no_grad():
                    batch = next(self.get_shuffled_test_loader_iter(task))
                    # NOTE: The labels are not allowed to be used during test-time adaptation
                    images = batch[0]

                    perturbed_images = torch.clamp(
                        images + self.pertubed_model.perturbed_input[task_idx],
                        min=0,
                        max=1,
                    )
                    combined_images = torch.cat((images, perturbed_images), dim=0)

                with self.profile("forward pass"):

                    num_image = images.size(0)

                    ### loss_nat
                    combined_logits = self.compute_logits(
                        merge_model_forward, combined_images, task
                    )
                    logits, logits_adv = (
                        combined_logits[:num_image],
                        combined_logits[num_image:],
                    )
                    ori_label = torch.argmax(logits, dim=1).long()
                    pert_label = torch.argmax(logits_adv, dim=1).long()
                    loss_nat = entropy_loss(logits)

                    ########### loss_regu from noise
                    ### regu1
                    # ori_label = torch.argmax(logits, dim=1).long()
                    # loss_regu = -torch.mean(
                    #     F.cross_entropy(logits_adv, ori_label, reduction="mean")
                    # )
                    ### regu2
                    loss_regu = entropy_loss(logits_adv)

                    ### loss shared
                    combined_logits_ref = self.compute_logits(
                        merge_model_ref, combined_images, task
                    )
                    logits_ref, logits_adv_ref = (
                        combined_logits_ref[:num_image],
                        combined_logits_ref[num_image:],
                    )
                    ori_label_ref = torch.argmax(logits_ref, dim=1).long()
                    pert_label_ref = torch.argmax(logits_adv_ref, dim=1).long()

                    success_attack = pert_label != ori_label

                    #### due to fact that we only use the test data without true label there, we replace the true label with ori_label
                    success_attack_ref = pert_label_ref != ori_label
                    success_attack_ref = success_attack_ref & (
                        pert_label_ref != ori_label_ref
                    )

                    common_attack = torch.logical_and(
                        success_attack, success_attack_ref
                    )
                    shared_attack = torch.logical_and(
                        common_attack, pert_label == pert_label_ref
                    )

                    potential_poison = success_attack_ref
                    if potential_poison.sum() == 0:
                        loss_shared = torch.tensor(0.0).to(merge_model.device)
                    else:
                        one_hot = F.one_hot(pert_label_ref, num_classes=num_classes)

                        neg_one_hot = 1 - one_hot
                        neg_p = (F.softmax(logits_adv, dim=1) * neg_one_hot).sum(dim=1)[
                            potential_poison
                        ]
                        pos_p = (F.softmax(logits_adv, dim=1) * one_hot).sum(dim=1)[
                            potential_poison
                        ]

                        # clamp the too small values to avoid nan and discard samples with p<1% to be shared
                        # Note: The below equation combine two identical terms in math. Although they are the same in math, they are different in implementation due to the numerical issue.
                        #       Combining them can reduce the numerical issue.

                        loss_shared = (
                            -torch.sum(torch.log(1e-6 + neg_p.clamp(max=0.999)))
                            - torch.sum(torch.log(1 + 1e-6 - pos_p.clamp(min=0.001)))
                        ) / 2
                        loss_shared = loss_shared / images.shape[0]

                    loss = (
                        loss_nat
                        + self.config.adv_weight * loss_regu
                        + self.config.shared_weight * loss_shared
                    )
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
                    "loss_shared": loss_shared.item(),
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

    def run(self, modelpool: HuggingFaceClipVisionPool):
        self.modelpool = to_modelpool(modelpool)
        config = self.config
        self.log_hyperparams(config, filename="method_config.yaml")

        with self.profile("setup models"):
            merge_model, merge_model_ref, pretrained_model, mask_model = (
                self.setup_models()
            )
            mask_model: MaskModel = self.fabric.to_device(mask_model)
            merge_model = self.fabric.to_device(merge_model)
            merge_model_ref = self.fabric.to_device(merge_model_ref)
            pretrained_model = self.fabric.to_device(pretrained_model)
            self.pertubed_model = self.fabric.to_device(self.pertubed_model)
            self.setup_zero_shot_classification_head(
                task_names=[c["name"] for c in self.modelpool.config.tta_datasets]
            )

        if config.mask_checkpoint is None:
            self.train_mask(
                merge_model=merge_model,
                merge_model_ref=merge_model_ref,
                pretrained_model=pretrained_model,
                mask_model=mask_model,
            )
        else:
            if self.fabric.is_global_zero:
                print("loading mask from checkpoint", config.mask_checkpoint)
            self.fabric.load(config.mask_checkpoint, {"model": mask_model})

        with torch.no_grad():
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            mask = mask_model.sample_mask(
                mask_type=config.eval_mask_type,
                temperature=config.temperature,
            )
            # rescale mask
            for name, m in mask.items():
                mask[name] = m / torch.mean(m)
            pretrained_model_dict = pretrained_model.state_dict(keep_vars=True)
            merged_state_dict = merge_model.state_dict(keep_vars=True)
            for name, parameter in merged_state_dict.items():
                ## (1) mask--directly prune the merged model, the initial logits should be larger than 3
                # merged_state_dict[name] = merged_state_dict[name]* mask[name]
                ### (2) mask the task vector, similar to concrete mask, the initial logits can be set as 0
                merged_state_dict[name] = (
                    merged_state_dict[name] - pretrained_model_dict[name]
                ) * mask[name] + pretrained_model_dict[name]
            merge_model.load_state_dict(merged_state_dict)
        return merge_model
