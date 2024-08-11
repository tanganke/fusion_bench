"""
For example:

```bash
fusion_bench \
    method=clip_finetune \
    modelpool=clip-vit-base-patch32_mtl \
    taskpool=dummy
```
"""

import os

import lightning as L
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import CLIPModel, CLIPProcessor

from fusion_bench.method import ModelFusionAlgorithm
from fusion_bench.mixins.simple_profiler import SimpleProfilerMixin
from fusion_bench.modelpool import to_modelpool
from fusion_bench.modelpool.huggingface_clip_vision import HuggingFaceClipVisionPool
from fusion_bench.models.hf_clip import HFCLIPClassifier
from fusion_bench.tasks.clip_classification import get_classnames_and_templates
from fusion_bench.tasks.clip_classification.clip_mixin import CLIPClassificationMixin
from fusion_bench.utils.data import InfiniteDataLoader


class ImageClassificationFineTuningForCLIP(
    CLIPClassificationMixin,
    SimpleProfilerMixin,
    ModelFusionAlgorithm,
):
    def run(self, modelpool: HuggingFaceClipVisionPool):
        self.modelpool = to_modelpool(modelpool)
        config = self.config
        self.log_hyperparams(config, filename="method_config.yaml")

        L.seed_everything(config.seed)

        task_names = [
            dataset_config["name"] for dataset_config in modelpool.config.train_datasets
        ]
        with self.profile("setup model and optimizer"):
            processor, classifier, optimizer, lr_scheduler = self.setup_model()

            self.setup_zero_shot_classification_head(
                clip_processor=processor,
                clip_model=classifier.clip_model,
                task_names=task_names,
            )

            self.fabric.setup(classifier, optimizer)

        with self.profile("setup data"):
            train_datasets = [
                modelpool.get_train_dataset(task_name, processor)
                for task_name in task_names
            ]
            train_dataloaders = [
                DataLoader(
                    dataset,
                    shuffle=True,
                    batch_size=config.batch_size,
                    num_workers=config.num_workers,
                )
                for dataset in train_datasets
            ]
            train_dataloaders = self.fabric.setup_dataloaders(*train_dataloaders)
            train_dataloader_iters = [
                iter(InfiniteDataLoader(loader)) for loader in train_dataloaders
            ]

        # train
        for step_idx in tqdm(
            range(config.num_steps),
            "fine-tuning",
            disable=not self.fabric.is_global_zero,
        ):
            optimizer.zero_grad()
            loss = 0
            for task, loader in zip(task_names, train_dataloader_iters):
                with self.profile("data loading"):
                    batch = next(loader)
                    images, labels = batch
                with self.profile("forward"):
                    classifier.zeroshot_weights = self.zeroshot_weights[task]
                    logits = classifier(images)
                loss = loss + nn.functional.cross_entropy(logits, labels)

            with self.profile("backward"):
                self.fabric.backward(loss)
            with self.profile("optimizer step"):
                optimizer.step()

            metrics = {"train/loss": loss}

            self.fabric.log_dict(metrics, step=step_idx)

            if (step_idx + 1) % config.save_interval == 0:
                save_path = os.path.join(
                    self.log_dir, "checkpoints", f"step={step_idx}.ckpt"
                )
                _dir = os.path.dirname(save_path)
                if _dir and not os.path.exists(_dir):
                    os.makedirs(_dir, exist_ok=True)
                self.fabric.save(
                    save_path, {"vision_model": classifier.clip_model.vision_model}
                )

        self.print_profile_summary()
        return classifier.clip_model.vision_model

    def setup_model(self):
        config = self.config
        modelpool = self.modelpool

        pretrained_model_config = modelpool.get_model_config("_pretrained_")
        clip_model = CLIPModel.from_pretrained(pretrained_model_config.path)
        processor = CLIPProcessor.from_pretrained(pretrained_model_config.path)

        classifier = HFCLIPClassifier(clip_model, processor=processor)

        # configure optimizers
        optimizer = torch.optim.Adam(
            classifier.clip_model.vision_model.parameters(), lr=config.learning_rate
        )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=config.num_steps
        )

        return processor, classifier, optimizer, lr_scheduler
