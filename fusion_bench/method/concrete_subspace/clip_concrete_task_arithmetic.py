"""
Examples:

```bash
fusion_bench \
    fabric.loggers.name=ViT-B-32/concrete_task_arithmetic \
    method=clip_concrete_task_arithmetic \
    modelpool=clip-vit-base-patch32_TA8 \
    taskpool=clip-vit-classification_TA8
```
"""

import logging
import os

import torch
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
from fusion_bench.models.wrappers.task_wise_fusion import (
    TaskWiseMergedModel,
    get_task_wise_weights,
)
from fusion_bench.utils.dtype import parse_dtype
from fusion_bench.utils.parameters import print_parameters

log = logging.getLogger(__name__)


class ConcreteTaskArithmeticAlgorithmForCLIP(
    CLIPClassificationMixin,
    SimpleProfilerMixin,
    ModelFusionAlgorithm,
):
    """
    ConcreteTaskArithmeticAlgorithmForCLIP is a class for performing task arithmetic on CLIP models with learned masking.

    This class extends the CLIPClassificationMixin, SimpleProfilerMixin, and ModelFusionAlgorithm classes.
    It provides methods for setting up models, training masks, and running the task arithmetic algorithm.

    Attributes:
        merge_dtype (torch.dtype): The data type for merging weights.
        modelpool (HuggingFaceClipVisionPool): The model pool containing the pretrained and fine-tuned models.
    """

    @torch.no_grad()
    def setup_models(self):
        """
        Set up the pretrained model, fine-tuned models, and mask model.

        This method loads the pretrained model, constructs the PGE mask model, and loads the fine-tuned models.
        It also creates a wrapped model with task-wise weights.

        Returns:
            Tuple[TaskWiseMergedModel, MaskModel]: The wrapped model and mask model.
        """
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
        # mask_model: MaskModel = self.fabric.to_device(mask_model)

        # configure optimizer
        lr_scheduler = None
        if self.config.optimizer == "adam":
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, mask_model.parameters()),
                lr=self.config.lr,
            )
            print(f"{optimizer=}")
            # TODO: ablation study for the learning rate scheduler. It should yield similar results.
            # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            #     optimizer, self.config.max_steps, eta_min=0.1
            # )
            mask_model, optimizer = self.fabric.setup(mask_model, optimizer)
        elif self.config.optimizer == "sgd":
            optimizer = torch.optim.SGD(mask_model.parameters(), lr=self.config.lr)
            print(f"{optimizer=}")
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, self.config.max_steps, eta_min=0.1
            )
            mask_model, optimizer = self.fabric.setup(mask_model, optimizer)
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")

        module.to(mask_model.device)
        module.requires_grad_(False)

        mask_model.train()
        optimizer.zero_grad()
        for step_idx in (
            pbar := tqdm(
                range(self.config.max_steps if not self.is_debug_mode else 5),
                ("[DEBUG MODE] " if self.is_debug_mode else "")
                + "Concrete Task Arithmetic Test-Time Adaptation",
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
            #   Because the algorithmic parameters of task arithmetic are assumed to be chosen on a validation test
            #   set, we do not need to perform inner optimization here. So here we skip the inner optimization step.
            # ------------------------------------------

            total_loss = None
            for task in self.modelpool.model_names:
                with self.profile("data loading"):
                    batch = next(self.get_shuffled_test_loader_iter(task))
                    # NOTE: The labels are not allowed to be used during test-time adaptation
                    images = batch[0].to(dtype=self.merge_dtype)
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

    def run(self, modelpool: HuggingFaceClipVisionPool):
        """
        Run the Concrete Task Arithmetic algorithm.

        This method sets up the models, trains the mask model if necessary, and performs the final merging of weights.

        Args:
            modelpool (HuggingFaceClipVisionPool): The model pool containing the pretrained and fine-tuned models.

        Returns:
            torch.nn.Module: The final merged model.
        """
        self.modelpool = to_modelpool(modelpool)
        config = self.config
        self.log_hyperparams(config, filename="method_config.yaml")

        with self.profile("setup models"):
            module, mask_model = self.setup_models()
            self.setup_zero_shot_classification_head()

        if config.mask_checkpoint is None:
            if not config.skip_training:
                torch.cuda.empty_cache()
                self.train_mask(module=module, mask_model=mask_model)
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
            model = module.merge_and_unload(mask)
        return model.to(dtype=torch.float32)
