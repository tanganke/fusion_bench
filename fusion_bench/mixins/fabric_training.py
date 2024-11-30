import itertools
import logging
import os
from abc import abstractmethod
from typing import TYPE_CHECKING, Literal, Union

import torch
from torch import Tensor, nn
from tqdm.auto import tqdm

from .lightning_fabric import LightningFabricMixin

if TYPE_CHECKING:
    from lightning.fabric.wrappers import (
        _FabricDataLoader,
        _FabricModule,
        _FabricOptimizer,
    )

log = logging.getLogger(__name__)


class FabricTrainingMixin(LightningFabricMixin):

    _latest_saved_checkpoint_global_step: int = -1
    """The global step index of the latest saved checkpoint."""
    _expected_total_steps: int = None
    """The expected total number of steps of the entire training."""
    is_training: bool
    """Whether the training is in progress. If set to False, the training will stop."""
    epoch_idx: int
    """The epoch index, which is the number of epochs completed."""
    global_step_idx: int
    """The global step index, which is the number of parameter update steps."""
    max_epochs: int
    """Max number of epochs of the entire training."""
    max_steps: int
    """Max number of parameter update steps of the entire training."""
    max_steps_per_epoch: int
    """Max number of parameter update steps per epoch."""
    gradient_clip_algorithm: Literal["value", "norm"]
    """The algorithm to clip gradients. Available options: 'value', 'norm'."""
    gradient_clip_val: float
    """The value to clip gradients. If None, no clipping is applied."""
    accumulate_grad_batches: int
    """The number of gradient accumulation steps. The effective global batch size is `the batch size per device` x `the number of devices` x `the number of gradient accumulation steps`."""
    lr_scheduler_interval: Literal["step", "epoch"]
    """The interval to run the learning rate scheduler. Available options: 'step', 'epoch'."""
    lr_scheduler_frequency: int
    """The frequency to run the learning rate scheduler."""
    checkpoint_save_interval: Literal["step", "epoch"]
    """The interval to save the model checkpoint. Available options: 'step', 'epoch'."""
    checkpoint_save_frequency: int
    """The frequency to save the model checkpoint."""

    def clip_gradients_if_needed(self, model, optimizer):
        fabric = self.fabric

        if self.gradient_clip_val is not None:
            if self.gradient_clip_algorithm == "value":
                fabric.clip_gradients(model, optimizer, clip_val=self.gradient_clip_val)
            elif self.gradient_clip_algorithm == "norm":
                fabric.clip_gradients(model, optimizer, max_norm=self.gradient_clip_val)
            else:
                raise ValueError(
                    f"Unknown gradient clip algorithm: {self.gradient_clip_algorithm}. Available options: 'value', 'norm'"
                )

    def compute_expected_total_steps(
        self, train_dataloader: torch.utils.data.DataLoader
    ):
        # compute expected total steps
        self._expected_total_steps = []
        if self.max_steps > 0:
            self._expected_total_steps.append(self.max_steps)
        if self.max_steps_per_epoch > 0 and self.max_epochs > 0:
            self._expected_total_steps.append(
                self.max_steps_per_epoch * self.max_epochs
            )
        if self.max_epochs > 0:
            self._expected_total_steps.append(
                len(train_dataloader) * self.max_epochs // self.accumulate_grad_batches
            )
        self._expected_total_steps = min(self._expected_total_steps)
        log.info(f"Expected total steps: {self._expected_total_steps}")

    @property
    def expected_total_steps(self):
        """The expected total number of steps of the entire training. You need to run `compute_expected_total_steps` method to compute this value before accessing it."""
        if self._expected_total_steps is None:
            raise ValueError(
                "The expected total steps have not been computed. Run `compute_expected_total_steps` method."
            )
        else:
            return self._expected_total_steps

    def conditional_checkpoint_save(
        self,
        stage: Literal["end_of_step", "end_of_epoch", "end_of_training"],
        *args,
        **kwargs,
    ):
        if stage == "end_of_step":
            if (
                self.checkpoint_save_interval == "step"
                and (self.global_step_idx + 1) % self.checkpoint_save_frequency == 0
            ):
                save_path = os.path.join(
                    self.log_dir, "checkpoints", f"step={self.global_step_idx}.ckpt"
                )
                self.save_checkpoint(save_path, *args, **kwargs)
        elif stage == "end_of_epoch":
            if (
                self.checkpoint_save_interval == "epoch"
                and (self.epoch_idx + 1) % self.checkpoint_save_frequency == 0
            ):
                save_path = os.path.join(
                    self.log_dir, "checkpoints", f"epoch={self.epoch_idx}.ckpt"
                )
                self.save_checkpoint(save_path, *args, **kwargs)
        elif stage == "end_of_training":
            # if the checkpoint has not been saved yet, save it
            if self.global_step_idx > self._latest_saved_checkpoint_global_step:
                save_path = os.path.join(
                    self.log_dir,
                    "checkpoints",
                    f"epoch={self.epoch_idx}_step={self.global_step_idx}.ckpt",
                )
                self.save_checkpoint(save_path, *args, **kwargs)
                try:
                    os.symlink(
                        save_path,
                        os.path.join(self.log_dir, "checkpoints", "latest_model.ckpt"),
                    )
                except Exception as e:
                    log.error(f"Failed to create symlink: {e}")
        else:
            raise ValueError(
                f"Unknown stage: {stage}. Available options: 'end_of_step', 'end_of_epoch', 'end_of_training'"
            )

    @abstractmethod
    def save_checkpoint(self, path, **kwargs):
        raise NotImplementedError("save_checkpoint method is not implemented")

    def train(
        self,
        model: Union[nn.Module, "_FabricModule"],
        optimizer: Union[torch.optim.Optimizer, "_FabricOptimizer"],
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    ):
        """
        The global batch size is `the batch size per device` x `the number of devices` x `the number of gradient accumulation steps`.
        """
        fabric = self.fabric
        self.is_training = True
        # number of parameter update iterations, not the number of batches
        self.global_step_idx = 0
        model.train()
        optimizer.zero_grad()
        for epoch_idx in tqdm(
            range(self.max_epochs) if self.max_epochs > 0 else itertools.count(0),
            "Training Epoch",
            dynamic_ncols=True,
            leave=False,
            disable=not fabric.is_global_zero,
        ):
            self.epoch_idx = epoch_idx
            self.train_epoch(model, optimizer, lr_scheduler)
            # run lr_scheduler at the end of the epoch if interval is set to "epoch"
            if (
                self.lr_scheduler_interval == "epoch"
                and (epoch_idx + 1) % self.lr_scheduler_frequency == 0
            ):
                lr_scheduler.step()

            # save the model at the end of the epoch if interval is set to "epoch" and frequency is met
            self.conditional_checkpoint_save(stage="end_of_epoch")

            if not self.is_training:
                break

        optimizer.zero_grad()
        # save the model at the end of training
        self.conditional_checkpoint_save(stage="end_of_training")
        return model

    @abstractmethod
    def train_epoch(
        self,
        model: Union[nn.Module, "_FabricModule"],
        optimizer: Union[torch.optim.Optimizer, "_FabricOptimizer"],
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    ):
        raise NotImplementedError(
            "Copy this as a template and implement your own train_epoch method"
        )
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

            # disable gradient synchronization if accumulating gradients across steps for improved performance
            with fabric.no_backward_sync(self.model, enabled=is_accumulating):
                # use_cache=True is not compatible with gradient checkpointing, so we disable it here
                output = self.compute_loss(batch)
                loss = output["loss"] / self.accumulate_grad_batches

                fabric.backward(loss)
                accumulated_loss += loss.item()

            # 1. update the model parameters if not accumulating gradients
            # 2. step the lr_scheduler if interval is set to "step" and frequency is met
            # 3. save the model if interval is set to "step" and frequency is met
            # 4. log metrics
            # 5. increase the global step index and reset the accumulated metrics
            if not is_accumulating:
                self.clip_gradients_if_needed(model, optimizer)

                # run lr_scheduler at the end of the step if interval is set to "step"
                if (
                    self.lr_scheduler_interval == "step"
                    and (self.global_step_idx + 1) % self.lr_scheduler_frequency == 0
                ):
                    lr_scheduler.step()

                # update the model parameters and zero the gradients
                optimizer.step()
                optimizer.zero_grad()

                metrics = {
                    "train/loss": accumulated_loss,
                    "train/lr": optimizer.param_groups[0]["lr"],
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
