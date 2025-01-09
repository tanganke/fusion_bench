"""
For example:

Fine-tune CLIP-ViT-B/32:

```bash
fusion_bench \
    method=clip_finetune \
    modelpool=clip-vit-base-patch32_mtl \
    taskpool=dummy
```

Fine-tune CLIP-ViT-L/14 on eight GPUs with a per-device per-task batch size of 2.

```bash
fusion_bench \
    fabric.devices=8 \
    method=clip_finetune \
        method.batch_size=2 \
    modelpool=clip-vit-base-patch32_mtl \
        modelpool.models.0.path=openai/clip-vit-large-patch14 \
    taskpool=dummy
```
"""

import os
from typing import Optional, Tuple

import lightning as L
import torch
from omegaconf import OmegaConf
from peft import LoraConfig, PeftModel, get_peft_model
from peft.tuners.lora import LoraLayer
from safetensors.torch import save_file
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import CLIPModel, CLIPProcessor, CLIPVisionModel
from transformers.models.clip.modeling_clip import CLIPVisionTransformer

from fusion_bench import print_parameters
from fusion_bench.compat.method import ModelFusionAlgorithm
from fusion_bench.compat.modelpool import to_modelpool
from fusion_bench.dataset.clip_dataset import CLIPDataset
from fusion_bench.mixins import CLIPClassificationMixin
from fusion_bench.mixins.simple_profiler import SimpleProfilerMixin
from fusion_bench.modelpool import CLIPVisionModelPool
from fusion_bench.models.hf_clip import HFCLIPClassifier
from fusion_bench.models.linearized.linearized_model_utils import LinearizedModelWraper
from fusion_bench.utils.data import InfiniteDataLoader


def _get_submodules(model, key) -> Tuple:
    """
    Retrieves the parent module, target module, and target module name for a given key in a PyTorch model.
    """
    parent = model.get_submodule(".".join(key.split(".")[:-1]))
    target_name = key.split(".")[-1]
    target = model.get_submodule(key)
    return parent, target, target_name


def linearize_lora_model_(model):
    """
    Linearizes the LoraLayer modules in a PyTorch model according to the PETA paper.
    """
    for key, module in model.named_modules():
        # if isinstance(module, LoraLayer) and isinstance(module, nn.Linear):
        if isinstance(module, LoraLayer):
            # print("L-LoRA MODULE : ", module)
            parent, target, target_name = _get_submodules(model, key)
            setattr(parent, target_name, LinearizedModelWraper(target))
            # print("Linearized Lora Layer")
    return model


def unlinearize_lora_model_(model):
    """
    Unloads the linearized LoraLayer modules in a PyTorch model.
    """
    LinearizedModelWraper.unload_linearized_modules_(model)
    return model


class ImageClassificationFineTuningForCLIP(
    CLIPClassificationMixin,
    SimpleProfilerMixin,
    ModelFusionAlgorithm,
):
    """
    A class for fine-tuning CLIP models for image classification tasks.
    """

    def run(self, modelpool: CLIPVisionModelPool):
        """
        Executes the fine-tuning process.

        Args:
            modelpool (CLIPVisionModelPool): The modelpool is responsible for loading the pre-trained model and training datasets.

        Returns:
            VisionModel: The fine-tuned vision model.
        """
        self.modelpool = to_modelpool(modelpool)
        config = self.config
        self.log_hyperparams(config, filename="method_config.yaml")
        self.finetune_method = "fine-tune"

        L.seed_everything(config.seed)

        task_names = modelpool.train_dataset_names
        with self.profile("setup model and optimizer"):
            processor, classifier, optimizer, lr_scheduler = self.setup_model()

            if config.state_dict_load_path is not None:
                self.fabric.load(
                    config.state_dict_load_path,
                    {"vision_model": classifier.clip_model.vision_model},
                )
                if config.skip_training:
                    return classifier.clip_model.vision_model

            self.setup_zero_shot_classification_head(
                clip_processor=processor,
                clip_model=classifier.clip_model,
                task_names=task_names,
            )

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
                    batch_size=config.batch_size,
                    num_workers=config.num_workers,
                )
                for dataset in train_datasets
            ]
            train_dataloaders = self.fabric.setup_dataloaders(*train_dataloaders)
            if not isinstance(train_dataloaders, (list, tuple)):
                train_dataloaders = [train_dataloaders]
            train_dataloader_iters = [
                iter(InfiniteDataLoader(loader)) for loader in train_dataloaders
            ]

        # train
        for step_idx in tqdm(
            range(config.num_steps),
            desc=self.finetune_method,
            disable=not self.fabric.is_global_zero,
            dynamic_ncols=True,
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
                lr_scheduler.step()

            metrics = {"train/loss": loss}

            self.fabric.log_dict(metrics, step=step_idx)

            if (step_idx + 1) % config.save_interval == 0:
                save_path = os.path.join(
                    self.log_dir, "checkpoints", f"step={step_idx}.ckpt"
                )
                self.save_model(classifier, save_path)

        if config.state_dict_save_path is not None:
            self.save_model(classifier, config.state_dict_save_path)
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
        if config.use_lora or config.use_l_lora:
            self.finetune_method = "lora fine-tune"
            lora_config = LoraConfig(
                **OmegaConf.to_container(
                    config.lora_config, resolve=True, enum_to_str=True
                )
            )
            clip_model.vision_model = get_peft_model(
                clip_model.vision_model, lora_config
            )

            if config.use_l_lora:
                # http://arxiv.org/abs/2310.04742
                # Anke Tang et al. Parameter Efficient Multi-task Model Fusion with Partial Linearization. ICLR 2024.
                self.finetune_method = "l-lora fine-tune"
                print("Linearizing Lora Layers")
                linearize_lora_model_(clip_model.vision_model)

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
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=config.num_steps
        )

        return processor, classifier, optimizer, lr_scheduler


def load_full_finetuned_vision_model(
    pretrained_path: str, state_dict_path: str, strict=True
):
    """
    Load a fully fine-tuned model from the state_dict_path and pretrained_path.

    Args:
        pretrained_path (str): The path to the pretrained model.
        state_dict_path (str): The path to the state dictionary.
        strict (bool, optional): Whether to strictly enforce that the keys in state_dict match the keys returned by this module's state_dict() function. Defaults to True.

    Returns:
        CLIPVisionModel: The loaded vision model.
    """
    model: CLIPVisionModel = CLIPVisionModel.from_pretrained(pretrained_path)
    model.vision_model.load_state_dict(
        torch.load(state_dict_path, map_location="cpu")["vision_model"], strict=strict
    )
    return model


def load_lora_vision_model(
    pretrained_path: str,
    lora_config: LoraConfig,
    state_dict_path: str,
) -> PeftModel:
    """
    Load LoRA model from the state_dict_path and pretrained_path.

    Args:
        pretrained_path (str): The path to the pretrained model.
        lora_config (LoraConfig): The configuration for LoRA.
        state_dict_path (str): The path to the state dictionary.

    Returns:
        PeftModel: The loaded LoRA model.
    """
    model: CLIPVisionModel = CLIPVisionModel.from_pretrained(pretrained_path)
    model = get_peft_model(model.vision_model, lora_config)
    state_dict = torch.load(state_dict_path, map_location="cpu")["vision_model"]
    for name, value in state_dict.items():
        model.get_parameter(name).data = value
    return model


def load_l_lora_vision_model(
    pretrained_path: str,
    lora_config: LoraConfig,
    state_dict_path: str,
    unload_linearized_modules: bool = False,
):
    """
    Load L-LoRA model from the state_dict_path and pretrained_path.

    The output folder should contain the following files:

    - README.md
    - adapter_config.json
    - linearized_adapter_model.safetensors

    Load the converted model using the following code:

    >>> from fusion_bench.models.linearized.vision_model import load_l_lora_vision_model_hf
    >>> model = load_l_lora_vision_model_hf("base_model_name", "peft_name")

    Args:
        pretrained_path (str): The path to the pretrained model.
        lora_config (LoraConfig): The configuration for LoRA.
        state_dict_path (str): The path to the state dictionary.
        unload_linearized_modules (bool, optional): Whether to unload the linearized modules. Defaults to False.

    Returns:
        PeftModel: The loaded L-LoRA model.
    """
    model: CLIPVisionModel = CLIPVisionModel.from_pretrained(pretrained_path)
    model = get_peft_model(model.vision_model, lora_config)
    linearize_lora_model_(model)
    state_dict = torch.load(state_dict_path, map_location="cpu")["vision_model"]
    for name, value in state_dict.items():
        model.get_parameter(name).data = value
    if unload_linearized_modules:
        LinearizedModelWraper.unload_linearized_modules_(model)
    return model


def convert_lora_state_dict_to_hf(
    pretrained_path: str,
    ckpt_path: str,
    lora_config: LoraConfig,
    output_path: str,
    base_model_name: Optional[str] = None,
):
    """
    Convert a LoRA model's checkpoint to Huggingface's format.

    Args:
        pretrained_path (str): The path to the pretrained model.
        ckpt_path (str): The path to the checkpoint.
        lora_config (LoraConfig): The configuration for LoRA.
        output_path (str): The path to save the converted model.
        base_model_name (Optional[str], optional): The name of the base model. Defaults to None.
    """
    model = load_lora_vision_model(
        pretrained_path=pretrained_path,
        state_dict_path=ckpt_path,
        lora_config=lora_config,
    )

    model.config._name_or_path = base_model_name
    model.peft_config["default"].base_model_name_or_path = base_model_name
    model.save_pretrained(output_path)


def convert_l_lora_state_dict_to_hf(
    pretrained_path: str,
    ckpt_path: str,
    lora_config: LoraConfig,
    output_path: str,
    base_model_name: Optional[str] = None,
):
    """
    Convert a linearized Lora model's checkpoint to Hugggingface's format.

    Args:
        pretrained_path (str): The path to the pretrained model.
        ckpt_path (str): The path to the checkpoint.
        lora_config (LoraConfig): The configuration for LoRA.
        output_path (str): The path to save the converted model.
        base_model_name (Optional[str], optional): The name of the base model. Defaults to None.
    """

    if base_model_name is None:
        base_model_name = pretrained_path

    # load l-lora model from ckpt file
    model = load_l_lora_vision_model(
        pretrained_path=pretrained_path,
        state_dict_path=ckpt_path,
        lora_config=lora_config,
        unload_linearized_modules=False,
    )

    # save lora config
    model.peft_config["default"].base_model_name_or_path = base_model_name
    model.peft_config["default"].save_pretrained(output_path)

    # save linearized adapter to safetensors
    linearized_adapter = {}
    for name, module in model.named_modules():
        if isinstance(module, LinearizedModelWraper):
            for (param_name, param), (param0_name, param0) in zip(
                module.model.named_parameters(),
                module.params0_values.named_parameters(),
            ):
                if param.requires_grad:
                    linearized_adapter[f"{name}.model.{param_name}"] = param
                    linearized_adapter[f"{name}.params0_values.{param0_name}"] = param0

    save_file(linearized_adapter, f"{output_path}/linearized_adapter_model.safetensors")

    readme = f"""---
base_model: {base_model_name}
library_name: peft
---

# Model Card L-LoRA Model
"""
    # open README.md
    with open(f"{output_path}/README.md", "w") as file:
        file.write(readme)
