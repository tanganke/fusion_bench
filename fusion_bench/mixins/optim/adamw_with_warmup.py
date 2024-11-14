import math
from abc import abstractmethod

import torch


def warmup_cosine_schedule(warmup_steps: int, total_steps: int, min_lr: float = 0):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # Cosine annealing
            progress = float(current_step - warmup_steps) / float(
                max(1, total_steps - warmup_steps)
            )
            return max(min_lr, 0.5 * (1 + math.cos(math.pi * progress)))

    return lr_lambda


class AdamWWithWarmUp:
    """
    An mixin for pl.LightningModule.
    """

    _optimizer_kwargs = {"lr": 6e-6, "weight_decay": 0.05}
    _lr_scheduler_kwargs = {"warmup_ratio": 0.1, "min_lr": 0}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), **self._optimizer_kwargs)
        lr_lambda = warmup_cosine_schedule(
            warmup_steps=self.trainer.estimated_stepping_batches
            * self._lr_scheduler_kwargs["warmup_ratio"],
            total_steps=self.trainer.estimated_stepping_batches,
            min_lr=self._lr_scheduler_kwargs["min_lr"],
        )
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": lr_scheduler, "interval": "step"},
        }
