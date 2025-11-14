import logging
import math
import os
import pickle
import time
from typing import List, Union

import numpy as np
import torch
from torch import Tensor, nn
from tqdm import tqdm

log = logging.getLogger(__name__)


class timeit_context:
    """
    Usage:

    ```python
    with timeit_context() as timer:
        ... # code block to be measured
    ```
    """

    def _log(self, msg):
        log.log(self.loglevel, msg, stacklevel=3)

    def __init__(self, msg: str = None, loglevel=logging.INFO) -> None:
        self.loglevel = loglevel
        self.msg = msg

    def __enter__(self) -> None:
        """
        Sets the start time and logs an optional message indicating the start of the code block execution.

        Args:
            msg: str, optional message to log
        """
        self.start_time = time.time()
        if self.msg is not None:
            self._log(self.msg)

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Calculates the elapsed time and logs it, along with an optional message indicating the end of the code block execution.
        """
        end_time = time.time()
        elapsed_time = end_time - self.start_time
        self._log(f"Elapsed time: {elapsed_time:.2f}s")


def num_devices(devices: Union[int, List[int], str]) -> int:
    """
    Return the number of devices.

    Args:
        devices: `devices` can be a single int to specify the number of devices, or a list of device ids, e.g. [0, 1, 2, 3]， or a str of device ids, e.g. "0,1,2,3" and "[0, 1, 2]".

    Returns:
        The number of devices.
    """
    if isinstance(devices, int):
        return devices
    elif isinstance(devices, str):
        return len(devices.split(","))
    elif isinstance(devices, list):
        return len(devices)
    else:
        raise TypeError(
            f"devices must be a single int or a list of ints, but got {type(devices)}"
        )


def num_parameters(model: nn.Module) -> int:
    """
    Return the number of parameters in a PyTorch model.

    Args:
        model (nn.Module): The PyTorch model to count the number of parameters in.

    Returns:
        The number of parameters in the model.
    """
    return sum(p.numel() for p in model.parameters())


def assign_learning_rate(param_group, new_lr):
    param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def cosine_lr(optimizer, base_lrs, warmup_length, steps):
    if not isinstance(base_lrs, list):
        base_lrs = [base_lrs for _ in optimizer.param_groups]
    assert len(base_lrs) == len(optimizer.param_groups)

    def _lr_adjuster(step):
        for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
            if step < warmup_length:
                lr = _warmup_lr(base_lr, warmup_length, step)
            else:
                e = step - warmup_length
                es = steps - warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
            assign_learning_rate(param_group, lr)

    return _lr_adjuster


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [
        float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
        for k in topk
    ]


def torch_load_old(save_path, device=None):
    with open(save_path, "rb") as f:
        classifier = pickle.load(f)
    if device is not None:
        classifier = classifier.to(device)
    return classifier


def torch_save(model, save_path):
    if os.path.dirname(save_path) != "":
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.cpu(), save_path)


def torch_load(save_path, device=None):
    model = torch.load(save_path)
    if device is not None:
        model = model.to(device)
    return model


def get_logits(inputs: Tensor, classifier) -> Tensor:
    assert callable(classifier)
    if hasattr(classifier, "to"):
        classifier = classifier.to(inputs.device)
    return classifier(inputs)


def get_probs(inputs, classifier):
    if hasattr(classifier, "predict_proba"):
        probs = classifier.predict_proba(inputs.detach().cpu().numpy())
        return torch.from_numpy(probs)
    logits = get_logits(inputs, classifier)
    return logits.softmax(dim=1)


class LabelSmoothing(torch.nn.Module):
    def __init__(self, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
