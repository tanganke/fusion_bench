import functools
import itertools
import logging
import os
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Union

import lightning as L
import omegaconf
import torch
from lightning.fabric.strategies.fsdp import FSDPStrategy
from lightning.fabric.utilities import rank_zero_only
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing_extensions import TYPE_CHECKING, override

from fusion_bench import BaseAlgorithm, BaseModelPool
from fusion_bench.dataset.llama.collate import padded_collate_sft
from fusion_bench.mixins import FabricTrainingMixin
from fusion_bench.modelpool import CausalLMPool
from fusion_bench.utils import instantiate
from fusion_bench.utils.dtype import get_dtype

if TYPE_CHECKING:
    from lightning.fabric.wrappers import (
        _FabricDataLoader,
        _FabricModule,
        _FabricOptimizer,
    )
    from transformers.models.llama.modeling_llama import LlamaForCausalLM

log = logging.getLogger(__name__)


class FullFinetuneSFT(BaseAlgorithm, FabricTrainingMixin):

    model: Union[nn.Module, "_FabricModule", "LlamaForCausalLM"]
    optimizer: Union[torch.optim.Optimizer, "_FabricOptimizer"]
    train_dataloader: Union[DataLoader, "_FabricDataLoader"]
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler
    _latest_saved_checkpoint_global_step: int = -1

    def __init__(
        self,
        optimizer: DictConfig,
        lr_scheduler: Optional[DictConfig],
        dataloader_kwargs: DictConfig,
        max_epochs: int,
        max_steps: int = -1,
        max_steps_per_epoch: int = -1,
        lr_scheduler_interval: Literal["epoch", "step"] = "step",
        lr_scheduler_frequency: int = 1,
        checkpoint_save_interval: Literal["epoch", "step"] = "epoch",
        checkpoint_save_frequency: int = 1,
        accumulate_grad_batches: int = 1,
        gradient_clip_val: Optional[float] = None,
        gradient_clip_algorithm: Literal["value", "norm"] = "norm",
        save_optimizer_state: bool = False,
        save_full_model: bool = False,
        save_ckpt_type: Literal["lightning", "hf"] = "lightning",
        ckpt_path: Optional[str] = None,
        max_length: int = 6144,
        fix_token_embedding: bool = True,
        **kwargs,
    ):
        """
        Class for full finetuning of a language model on given SFT datasets.

        Args:
            optimizer(DictConfig): Configuration for the optimizer.
            lr_scheduler(DictConfig): Configuration for the learning rate scheduler.
            dataloader_kwargs(DictConfig): Configuration for the dataloader, such as batch size, num_workers, etc.
            max_epochs(int): Maximum number of epochs to train the model. If set to -1, the training will continue indefinitely or until max_steps is reached.
            max_steps(int): Maximum number of steps to train the model. If set to -1, the training will continue indefinitely or until max_epochs is reached.
            max_steps_per_epoch(int): Maximum number of steps to train the model in each epoch. If set to -1, the training will continue until the end of the epoch.
            lr_scheduler_interval(str): Interval at which to run the learning rate scheduler. Available options: 'epoch', 'step'. If set to 'epoch', the scheduler will run at the end of each epoch. If set to 'step', the scheduler will run at the end of each step.
            lr_scheduler_frequency(int): Frequency at which to run the learning rate scheduler. The scheduler will run every `lr_scheduler_frequency` epochs or steps, depending on the value of `lr_scheduler_interval`.
            checkpoint_save_interval(str): Interval at which to save the model checkpoint. Available options: 'epoch', 'step'. If set to 'epoch', the model will be saved at the end of each epoch. If set to 'step', the model will be saved at the end of each step.
            checkpoint_save_frequency(int): Frequency at which to save the model checkpoint. The model will be saved every `checkpoint_save_frequency` epochs or steps, depending on the value of `checkpoint_save_interval`.
            accumulate_grad_batches(int): Number of batches to accumulate gradients across before updating the model parameters.
            gradient_clip_val(float): Value to clip the gradients. If set to None, no gradient clipping will be applied.
            gradient_clip_algorithm(str): Algorithm to use for gradient clipping. Available options: 'value', 'norm'. If set to 'value', the gradients will be clipped to the specified value. If set to 'norm', the gradients will be clipped to the specified norm.
            save_optimizer_state(bool): Whether to save the optimizer and lr_scheduler state along with the model checkpoint.
            save_full_model(bool): Whether to save the full model or only the trainable parameters in the model checkpoint.
            save_ckpt_type (str): Type of checkpoint to save. Available options: 'lightning', 'hf'. If set to 'lightning', the checkpoint will be saved in the lightning format. If set to 'hf', the checkpoint will be saved in the huggingface format.
            ckpt_path(str): Path to the checkpoint to load before training. If set to None, no checkpoint will be loaded.
            max_length(int): Maximum input length to consider. If the input length exceeds this value, it will be truncated.
            fix_token_embedding(bool): Whether to fix the token embeddings during training. If set to True, the token embeddings will not be updated during training.
        """
        self._optimizer = optimizer
        self._lr_scheduler = lr_scheduler
        self.dataloader_kwargs = dataloader_kwargs
        self.max_epochs = max_epochs
        self.max_steps = max_steps
        self.max_steps_per_epoch = max_steps_per_epoch
        self.lr_scheduler_interval = lr_scheduler_interval
        self.lr_scheduler_frequency = lr_scheduler_frequency
        self.checkpoint_save_interval = checkpoint_save_interval
        self.checkpoint_save_frequency = checkpoint_save_frequency
        self.accumulate_grad_batches = accumulate_grad_batches
        self.gradient_clip_val = gradient_clip_val
        self.gradient_clip_algorithm = gradient_clip_algorithm
        self.save_optimizer_state = save_optimizer_state
        self.save_full_model = save_full_model
        self.save_ckpt_type = save_ckpt_type
        self.ckpt_path = ckpt_path
        self.max_length = max_length
        self.fix_token_embedding = fix_token_embedding
        super().__init__(**kwargs)

    def run(self, modelpool: CausalLMPool):
        self.modelpool = modelpool
        self.setup()
        self.train(self.model, self.optimizer, self.lr_scheduler)
        return self.model

    def setup_model(self):
        self.tokenizer = self.modelpool.load_tokenizer()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        model = self.modelpool.load_pretrained_model()
        self.model: "LlamaForCausalLM" = model

        if self.fix_token_embedding:
            self.model.model.embed_tokens.requires_grad_(False)

        if self.fabric.strategy == "fsdp" or isinstance(
            self.fabric.strategy, FSDPStrategy
        ):
            # https://github.com/Lightning-AI/pytorch-lightning/issues/19267
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": True}
            )
            self.use_cache = False
        else:
            self.use_cache = True
        self.model_dtype = get_dtype(self.model)

    def configure_optimizer(self):
        # compute expected total steps
        self.compute_expected_total_steps(self.train_dataloader)

        optimizer = instantiate(self._optimizer, self.model.parameters())
        if self._lr_scheduler is not None:
            for key, arg in self._lr_scheduler.items():
                if arg == "_T_max_":
                    log.info(
                        f"Setting key `{key}` of lr_scheduler configuration to {self.expected_total_steps}"
                    )
                    self._lr_scheduler[key] = self.expected_total_steps
            lr_scheduler: torch.optim.lr_scheduler.LRScheduler = instantiate(
                self._lr_scheduler,
                optimizer=optimizer,
            )
        else:
            lr_scheduler = None
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def setup_data(self):
        fabric = self.fabric
        modelpool = self.modelpool
        assert (
            len(modelpool.train_dataset_names) > 0
        ), "No training datasets found in modelpool."

        train_datasets = [
            modelpool.load_train_dataset(dataset_name)
            for dataset_name in modelpool.train_dataset_names
        ]
        if len(train_datasets) > 1:
            train_dataset = ConcatDataset(train_datasets)
        else:
            train_dataset = train_datasets[0]

        self.train_dataset = train_dataset
        self.train_dataloader = DataLoader(
            train_dataset,
            **self.dataloader_kwargs,
            shuffle=True,
            collate_fn=functools.partial(
                padded_collate_sft, pad_token_id=self.tokenizer.pad_token_id
            ),
        )
        self.train_dataloader = fabric.setup_dataloaders(self.train_dataloader)

    def setup(self):
        fabric = self.fabric

        self.setup_model()
        self.setup_data()

        optimizer = self.configure_optimizer()
        optimizer, lr_scheduler = optimizer["optimizer"], optimizer["lr_scheduler"]

        self.model, self.optimizer = fabric.setup(self.model, optimizer)
        self.lr_scheduler = lr_scheduler

    @override
    def train_epoch(self, *args, **kwargs):
        fabric = self.fabric

        accumulated_loss = 0
        for step_idx, batch in enumerate(
            pbar := tqdm(
                self.train_dataloader,
                desc="Training Batches",
                dynamic_ncols=True,
                leave=False,
                disable=not fabric.is_global_zero,
            )
        ):
            is_accumulating = (step_idx + 1) % self.accumulate_grad_batches != 0

            if self.max_length > 0 and batch["input_ids"].shape[1] > self.max_length:
                log.warning(
                    f"Input length exceeds max_length: {batch['input_ids'].shape[1]} > {self.max_length}. Truncating input."
                )
                batch["input_ids"] = batch["input_ids"][:, : self.max_length]
                batch["attention_mask"] = batch["attention_mask"][:, : self.max_length]
                batch["labels"] = batch["labels"][:, : self.max_length]

            # disable gradient synchronization if accumulating gradients across steps for improved performance
            with fabric.no_backward_sync(self.model, enabled=is_accumulating):
                # use_cache=True is not compatible with gradient checkpointing, so we disable it here
                output = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    use_cache=self.use_cache,
                )
                loss = output["loss"] / self.accumulate_grad_batches

                fabric.backward(loss)
                accumulated_loss += loss.item()

            if not is_accumulating:
                self.clip_gradients_if_needed(self.model, self.optimizer)

                # run lr_scheduler at the end of the step if interval is set to "step"
                if (
                    self.lr_scheduler_interval == "step"
                    and (self.global_step_idx + 1) % self.lr_scheduler_frequency == 0
                ):
                    self.lr_scheduler.step()

                # update the model parameters and zero the gradients
                self.optimizer.step()
                self.optimizer.zero_grad()

                metrics = {
                    "train/loss": accumulated_loss,
                    "train/epoch_idx": self.epoch_idx,
                    "train/lr": self.optimizer.param_groups[0]["lr"],
                }
                fabric.log_dict(metrics, step=self.global_step_idx)
                pbar.set_postfix(metrics)

                # save the model at the end of the step if interval is set to "step" and frequency is met
                self.conditional_checkpoint_save(stage="end_of_step")

                # break if max_steps_per_epoch is set, and exit epoch
                if (
                    self.max_steps_per_epoch > 0
                    and step_idx + 1 >= self.max_steps_per_epoch
                ):
                    break
                # break if max_steps is set, and exit training
                if self.max_steps > 0 and self.global_step_idx >= self.max_steps - 1:
                    self.is_training = False
                    break

                self.global_step_idx += 1
                accumulated_loss = 0

    def save_checkpoint(
        self,
        path: Union[str, Path],
        save_optimizer_state: Optional[bool] = None,
        overwrite: bool = False,
    ):
        if not overwrite and os.path.exists(path):
            return log.warning(f"Checkpoint already exists at {path}. Skipping save.")

        fabric = self.fabric

        if self.save_ckpt_type == "lightning":
            state = {"model": self.model}

            # save the optimizer and lr_scheduler state if needed
            if self.save_optimizer_state and save_optimizer_state is not False:
                state.update(
                    {
                        "optimizer": self.optimizer,
                        "lr_scheduler": self.lr_scheduler,
                        "global_step_idx": self.global_step_idx,
                        "epoch_idx": self.epoch_idx,
                    }
                )

            trainable_param_names = set(
                name
                for name, param in self.model.state_dict(keep_vars=True).items()
                if param.requires_grad
            )
            filter = (
                None
                if self.save_full_model
                else {"model": lambda k, p: k in trainable_param_names}
            )

            fabric.save(path, state=state, filter=filter)
        else:
            self.model.save_pretrained(path, is_main_process=fabric.is_global_zero)

        self._latest_saved_checkpoint_global_step = self.global_step_idx

    def load_checkpoint(self, path: Union[str, Path]):
        fabric = self.fabric

        state = {"model": self.model}

        # save the optimizer and lr_scheduler state if needed
        if self.save_optimizer_state:
            state.update(
                {
                    "optimizer": self.optimizer,
                    "lr_scheduler": self.lr_scheduler,
                }
            )

        fabric.load(path, state)


def load_checkpoint(
    fabric: L.Fabric,
    ckpt_path: Union[str, Path],
    model: Union[nn.Module, "LlamaForCausalLM"],
    strict: bool = True,
    **state_components,
):
    """
    Load a checkpoint into a model.
    """
    state = {"model": model}
    state.update(state_components)
    fabric.load(ckpt_path, state=state, strict=strict)


if __name__ == "__main__":
    # convert a checkpoint to hf format
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", type=str)
    parser.add_argument("--ckpt-path", type=str)
    parser.add_argument("--output-path", type=str)

    args = parser.parse_args()

    fabric = L.Fabric(devices=1, strategy="fsdp")
    fabric.launch()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    tokenizer.save_pretrained(args.output_path)

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path, torch_dtype=torch.bfloat16
    )
    model = fabric.setup_module(model)
    load_checkpoint(fabric, args.ckpt_path, model=model, strict=True)
    model.save_pretrained(args.output_path)
