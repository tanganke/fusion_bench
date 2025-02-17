import os
import random
import time
from copy import deepcopy
from typing import Optional, Tuple, cast

import lightning as L
import torch
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, PeftModel, get_peft_model
from peft.tuners.lora import LoraLayer
from safetensors.torch import save_file
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import CLIPModel, CLIPProcessor, CLIPVisionModel
from transformers.models.clip.modeling_clip import CLIPVisionTransformer

from fusion_bench import BaseAlgorithm, print_parameters
from fusion_bench.compat.modelpool import to_modelpool
from fusion_bench.dataset.clip_dataset import CLIPDataset
from fusion_bench.mixins import CLIPClassificationMixin
from fusion_bench.mixins.simple_profiler import SimpleProfilerMixin
from fusion_bench.modelpool import CLIPVisionModelPool
from fusion_bench.models.hf_clip import HFCLIPClassifier
from fusion_bench.models.linearized.linearized_model_utils import LinearizedModelWraper
from fusion_bench.taskpool import CLIPVisionModelTaskPool
from fusion_bench.utils.data import InfiniteDataLoader
from fusion_bench.utils.fabric import seed_everything_by_time
from fusion_bench.utils.json import load_from_json, save_to_json


class ContinualImageClassificationFineTuningForCLIP(
    CLIPClassificationMixin,
    SimpleProfilerMixin,
    BaseAlgorithm,
):
    # attributes to configuration keys mapping
    _config_mapping = BaseAlgorithm._config_mapping | {
        "seed": "seed",
        "shuffle_order": "shuffle_order",
        "learning_rate": "learning_rate",
        "weight_decay": "weight_decay",
        "num_steps": "num_steps",
        "batch_size": "batch_size",
        "num_workers": "num_workers",
        "save_interval": "save_interval",
        "state_dict_load_path": "state_dict_load_path",
        "state_dict_save_path": "state_dict_save_path",
        "skip_training": "skip_training",
        "use_lora": "use_lora",
        "lora_config": "lora_config",
    }

    def __init__(
        self,
        seed: int = 42,
        shuffle_order: bool = True,
        learning_rate: float = 1e-5,
        weight_decay: float = 0,
        num_steps: int = 4000,
        batch_size: int = 128,
        num_workers: int = 16,
        save_interval: int = 500,
        state_dict_load_path: Optional[str] = None,
        state_dict_save_path: Optional[str] = None,
        skip_training: bool = False,
        use_lora: bool = False,
        lora_config: Optional[LoraConfig] = None,
    ):
        self.seed = seed
        self.shuffle_order = shuffle_order
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.save_interval = save_interval
        self.state_dict_load_path = state_dict_load_path
        self.state_dict_save_path = state_dict_save_path
        self.skip_training = skip_training
        self.use_lora = use_lora
        self.lora_config = lora_config

    def run(self, modelpool: CLIPVisionModelPool):
        self.modelpool = to_modelpool(modelpool)
        config = self.config
        self.log_hyperparams(config, filename="method_config.yaml")
        self.finetune_method = "fine-tune"

        if self.seed is not None:
            L.seed_everything(self.seed)
        else:
            seed_everything_by_time(self.fabric)

        task_names = list(modelpool.train_dataset_names)
        if self.shuffle_order:
            random.shuffle(task_names)
        if self.fabric.is_global_zero:
            save_to_json(task_names, os.path.join(self.log_dir, "task_names.json"))

        if self._program.taskpool is not None and isinstance(
            self._program.taskpool, CLIPVisionModelTaskPool
        ):
            has_taskpool = True
            taskpool = cast(CLIPVisionModelTaskPool, self._program.taskpool)
            test_datasets = taskpool._test_datasets
        else:
            has_taskpool = False

        with self.profile("setup model and optimizer"):
            processor, classifier, optimizer, lr_scheduler = self.setup_model()

            if self.state_dict_load_path is not None:
                self.fabric.load(
                    self.state_dict_load_path,
                    {"vision_model": classifier.clip_model.vision_model},
                )
                if self.skip_training:
                    return classifier.clip_model.vision_model

            self.setup_zero_shot_classification_head(
                clip_processor=processor,
                clip_model=classifier.clip_model,
                task_names=task_names,
            )

            init_optimizer_state_dict = optimizer.state_dict()
            init_lr_scheduler_state_dict = lr_scheduler.state_dict()
            self.fabric.setup(classifier, optimizer)

        with self.profile("setup data"):
            train_datasets = [
                CLIPDataset(modelpool.load_train_dataset(task_name), processor)
                for task_name in task_names
            ]
            train_dataloaders = [
                DataLoader(
                    dataset,
                    shuffle=True,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                )
                for dataset in train_datasets
            ]
            train_dataloaders = self.fabric.setup_dataloaders(*train_dataloaders)
            if not isinstance(train_dataloaders, (list, tuple)):
                train_dataloaders = [train_dataloaders]
            train_dataloader_iters = [
                iter(InfiniteDataLoader(loader)) for loader in train_dataloaders
            ]

        # continual train
        for task_idx, task_name in tqdm(
            enumerate(task_names),
            dynamic_ncols=True,
            disable=not self.fabric.is_global_zero,
        ):
            train_dataloader_iter = train_dataloader_iters[task_idx]

            # reset optimizer and lr scheduler
            print("reset optimizer and lr scheduler")
            optimizer.load_state_dict(init_optimizer_state_dict)
            lr_scheduler.load_state_dict(init_lr_scheduler_state_dict)

            for step_idx in tqdm(
                range(self.num_steps),
                desc=f"continual fine-tune on {task_name}",
                disable=not self.fabric.is_global_zero,
                dynamic_ncols=True,
                leave=False,
            ):
                optimizer.zero_grad()
                loss = 0
                with self.profile("data loading"):
                    batch = next(train_dataloader_iter)
                    images, labels = batch
                with self.profile("forward"):
                    classifier.zeroshot_weights = self.zeroshot_weights[task_name]
                    logits = classifier(images)
                    assert (
                        labels.max() < logits.shape[1]
                    ), f"for task {task_name}, labels.max() = {labels.max()}, logits.shape[1] = {logits.shape[1]}"
                loss = loss + nn.functional.cross_entropy(logits, labels)

                with self.profile("backward"):
                    self.fabric.backward(loss)
                with self.profile("optimizer step"):
                    optimizer.step()
                    lr_scheduler.step()

                metrics = {"train/loss": loss}
                self.fabric.log_dict(metrics, step=step_idx)

                if (step_idx + 1) % self.save_interval == 0:
                    save_path = os.path.join(
                        self.log_dir,
                        "checkpoints",
                        f"task={task_idx}_step={step_idx}.ckpt",
                    )
                    self.save_model(classifier, save_path)

            if has_taskpool:
                taskpool._is_setup = False
                taskpool._test_datasets = DictConfig(
                    {t: test_datasets[t] for t in task_names[: task_idx + 1]}
                )
                eval_report = taskpool.evaluate(
                    deepcopy(classifier.clip_model.vision_model),
                    name=task_name,
                )
                if self.fabric.is_global_zero:
                    save_to_json(
                        eval_report,
                        os.path.join(self.log_dir, f"results_{task_idx}.json"),
                    )

        if self.state_dict_save_path is not None:
            self.save_model(classifier, self.state_dict_save_path)
        self.print_profile_summary()
        return classifier.clip_model.vision_model

    def save_model(
        self,
        model: HFCLIPClassifier | CLIPModel | CLIPVisionModel | CLIPVisionTransformer,
        save_path: str,
    ):
        """
        Save the vision model to the specified path.

        Args:
            model (Union[HFCLIPClassifier, CLIPModel, CLIPVisionModel, CLIPVisionTransformer]): The model to save.
            save_path (str): The path to save the model.
        """
        if isinstance(model, HFCLIPClassifier):
            vision_model = model.clip_model.vision_model
        elif isinstance(model, CLIPModel):
            vision_model = model.vision_model
        elif isinstance(model, CLIPVisionModel):
            vision_model = model.vision_model
        elif isinstance(model, CLIPVisionTransformer):
            vision_model = model
        else:
            raise ValueError(f"Unsupported model type: {type(model)}")

        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        self.fabric.save(save_path, {"vision_model": vision_model})

    def setup_model(self):
        """
        Sets up the model, optimizer, and learning rate scheduler.

        This method initializes the CLIP model, applies LoRA if specified, and configures the optimizer and learning rate scheduler.

        Returns:
            Tuple: A tuple containing the processor, classifier, optimizer, and learning rate scheduler.
        """
        config = self.config
        modelpool = self.modelpool

        clip_model: CLIPModel = modelpool.load_clip_model("_pretrained_")
        processor = modelpool.load_processor()

        self.finetune_method = "full fine-tune"
        if self.use_lora:
            self.finetune_method = "lora fine-tune"
            lora_config = LoraConfig(
                **OmegaConf.to_container(
                    self.lora_config, resolve=True, enum_to_str=True
                )
            )
            clip_model.vision_model = get_peft_model(
                clip_model.vision_model, lora_config
            )

        classifier = HFCLIPClassifier(clip_model, processor=processor)

        if self.fabric.is_global_zero:
            print("=== Model Summary (For Vision Model Only) ===")
            print_parameters(classifier.clip_model.vision_model)
        # configure optimizers
        optimizer = torch.optim.Adam(
            [
                p
                for p in classifier.clip_model.vision_model.parameters()
                if p.requires_grad
            ],
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=self.num_steps
        )

        return processor, classifier, optimizer, lr_scheduler
