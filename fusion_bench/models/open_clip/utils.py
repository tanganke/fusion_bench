import copy
import os
import pickle
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


def compute_l1_norm(
    model1: nn.Module, model2: nn.Module
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Computes the L1 norm between the parameters of two models.

    Args:
        model1 (nn.Module): The first model.
        model2 (nn.Module): The second model.

    Returns:
        Tuple[torch.Tensor, Dict[str, float]]: A tuple containing the total L1 norm and a dictionary
        with the L1 norm for each layer.

    """
    norms = dict()
    l1_norm = 0.0
    for (n, p1), p2 in zip(model1.named_parameters(), model2.parameters()):
        layer_l1_norm = torch.norm(p1 - p2, 1)
        l1_norm += layer_l1_norm
        norms[n] = layer_l1_norm.item()

    return l1_norm, norms


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


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: List[int] = (1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [
        float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
        for k in topk
    ]


def torch_load_old(save_path: str, device=None):
    with open(save_path, "rb") as f:
        classifier = pickle.load(f)
    if device is not None:
        classifier = classifier.to(device)
    return classifier


def torch_save(model, save_path, save_state_dict=True):
    # TODO: hacky way to save state dict
    if save_state_dict and isinstance(model, torch.nn.Module):
        model = model.state_dict()
    if os.path.dirname(save_path) != "":
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model, save_path)


def torch_load(save_path, device=None):
    model = torch.load(save_path, map_location="cpu")
    if device is not None:
        model = model.to(device)
    return model


def get_logits(inputs, classifier):
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


class DotDict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def find_optimal_coef(
    results: Dict[str, Any],
    metric: str = "avg_normalized_top1",
    minimize: bool = False,
    control_metric: Optional[str] = None,
    control_metric_threshold: float = 0.0,
) -> float:
    """
    Finds the optimal coefficient based on the given results and metric.

    Args:
        results (Dict[str, Any]): A dictionary containing the results for different scaling coefficients.
        metric (str, optional): The metric to optimize. Defaults to "avg_normalized_top1".
        minimize (bool, optional): Whether to minimize the metric. Defaults to False.
        control_metric (str, optional): The control metric to check against. Defaults to None.
        control_metric_threshold (float, optional): The threshold value for the control metric. Defaults to 0.0.

    Returns:
        The optimal coefficient based on the given results and metric.
    """
    best_coef = None
    if minimize:
        best_metric = 1
    else:
        best_metric = 0
    for scaling_coef in results.keys():
        if control_metric is not None:
            if results[scaling_coef][control_metric] < control_metric_threshold:
                print(f"Control metric fell below {control_metric_threshold} threshold")
                continue
        if minimize:
            if results[scaling_coef][metric] < best_metric:
                best_metric = results[scaling_coef][metric]
                best_coef = scaling_coef
        else:
            if results[scaling_coef][metric] > best_metric:
                best_metric = results[scaling_coef][metric]
                best_coef = scaling_coef
    return best_coef


def nonlinear_advantage(nonlinear_acc, linear_acc, num_classes):
    """Computes the normalized non-linear advantage of a finetuned model.

    The nonlinear_advantage is defined as:
        error_rate(linear_model) - error_rate(nonlinear_model) / (1 - 1 / num_classes)
    and takes values between [-1, 1]. A value of 0 indicates that the nonlinear
    model is no better than the linear one. Meanwhile, a value of 1 indicates
    that the nonlinear model is perfect and the linear trivial, and a value of
    -1 indicates the opposite.
    """
    return (nonlinear_acc - linear_acc) / (1.0 - 1.0 / num_classes)


def to_cuda(input_dict):
    cuda_dict = {}
    for key, value in input_dict.items():
        cuda_dict[key] = value.to("cuda")
    return cuda_dict


def state_dict_to_vector(state_dict, remove_keys=[]):
    shared_state_dict = copy.deepcopy(state_dict)
    for key in remove_keys:
        if key in shared_state_dict:
            del shared_state_dict[key]
    sorted_shared_state_dict = OrderedDict(sorted(shared_state_dict.items()))
    return torch.nn.utils.parameters_to_vector(
        [value.reshape(-1) for key, value in sorted_shared_state_dict.items()]
    )


def vector_to_state_dict(vector, state_dict, remove_keys=[]):
    # create a reference dict to define the order of the vector
    reference_dict = copy.deepcopy(state_dict)
    for key in remove_keys:
        if key in reference_dict:
            del reference_dict[key]
    sorted_reference_dict = OrderedDict(sorted(reference_dict.items()))

    # create a shared state dict using the reference dict
    torch.nn.utils.vector_to_parameters(vector, sorted_reference_dict.values())

    # add back the encoder and decoder embedding weights.
    if "transformer.shared.weight" in sorted_reference_dict:
        for key in remove_keys:
            sorted_reference_dict[key] = sorted_reference_dict[
                "transformer.shared.weight"
            ]
    return sorted_reference_dict


def add_ptm_to_tv(tv_dict, ptm_dict):
    assert set(tv_dict.keys()) == set(
        ptm_dict.keys()
    ), "Differing parameter names in models."
    final_dict = copy.deepcopy(tv_dict)
    for k, v in ptm_dict.items():
        final_dict[k] = tv_dict[k] + v
    return final_dict


def check_parameterNamesMatch(checkpoints):
    parameter_names = set(checkpoints[0].keys())

    if len(checkpoints) >= 2:
        # raise ValueError("Number of models is less than 2.")
        for checkpoint in checkpoints[1:]:
            current_parameterNames = set(checkpoint.keys())
            if current_parameterNames != parameter_names:
                raise ValueError(
                    "Differing parameter names in models. "
                    f"The different parameters are {parameter_names.symmetric_difference(current_parameterNames)}"
                )


def check_state_dicts_equal(state_dict1, state_dict2):
    if set(state_dict1.keys()) != set(state_dict2.keys()):
        return False

    for key in state_dict1.keys():
        if not torch.equal(state_dict1[key], state_dict2[key]):
            return False

    return True


def topk_values_mask(M, K=0.7, return_mask=False, reshape_mask=False):
    if K == 100:
        # print("Not applying mask")
        if return_mask:
            return M, torch.ones_like(M), None
        else:
            return M, torch.ones_like(M)

    if K >= 1:
        K /= 100

    original_shape = M.shape
    if M.dim() == 1:
        M = M.unsqueeze(0)

    n, d = M.shape
    k = int(d * K)
    k = d - k  # Keep top k elements instead of bottom k elements

    # Find the k-th smallest element by magnitude for each row
    kth_values, _ = M.abs().kthvalue(k, dim=1, keepdim=True)
    # Create a mask tensor with True for the top k elements in each row
    mask = M.abs() >= kth_values
    final_mask = mask.squeeze() if original_shape == M.squeeze().shape else mask

    if reshape_mask:
        final_mask = final_mask.reshape(M.shape)

    if return_mask:
        return M * final_mask, final_mask.float().mean(dim=1), final_mask
    else:
        return M * final_mask, final_mask.float().mean(dim=1)


def cleanup_linear(state_dict):
    # The linear model also has keys for the reference point $\theta_0$ in the state dict with the prefix `params0`.
    state_dict = {k: v for k, v in state_dict.items() if "params." in k}
    return state_dict


def get_ptm_linear(state_dict: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    # rename keys so that they match afterwards
    state_dict_new = {
        k.replace("params0", "params"): v
        for k, v in state_dict.items()
        if "params0." in k
    }
    state_dict_remaining = {k: v for k, v in state_dict.items() if "params." not in k}

    return state_dict_new, state_dict_remaining
