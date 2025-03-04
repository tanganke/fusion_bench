import logging
from typing import List, Tuple, cast

import torch
import torch.nn as nn
from torch import Tensor
from tqdm.auto import tqdm
from transformers import LlamaForCausalLM, PreTrainedModel

from fusion_bench import timeit_context

from .data import get_loaders
from .layerwrapper import WrappedGPT

log = logging.getLogger(__name__)


def find_layers(module, layers=[nn.Linear], name=""):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(
            find_layers(
                child, layers=layers, name=name + "." + name1 if name != "" else name1
            )
        )
    return res


def check_sparsity(model):
    """
    Check the sparsity of the model by counting the number of zero weights.

    Args:
        model (PreTrainedModel): The model to check sparsity for.

    Returns:
        float: The sparsity ratio of the model.
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False

    layers = model.model.layers
    count = 0
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W == 0).sum().item()
            total_params += W.numel()

            sub_count += (W == 0).sum().item()
            sub_params += W.numel()

        print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    model.config.use_cache = use_cache
    return float(count) / total_params


def prepare_calibration_input(
    model: PreTrainedModel,
    dataloader: List[Tuple[Tensor, Tensor]],
    device: torch.device,
):
    """
    Prepare the calibration input for the model by collecting input to the first layer.

    Args:
        model (PreTrainedModel): The model to prepare calibration input for.
        dataloader (List[Tuple[Tensor, Tensor]]): The dataloader to use for calibration.
        device (torch.device): The device to use for calibration.

    Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor]: The prepared input, output, attention mask, and position IDs.
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    # dev = model.hf_device_map["model.embed_tokens"]
    if hasattr(model, "hf_device_map") and "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    # ? what if n_samples > 128
    inps = torch.zeros(
        (128, model.seqlen, model.config.hidden_size),
        dtype=dtype,
        device=device,
        requires_grad=False,
    )
    cache = {"i": 0, "attention_mask": None, "position_ids": None, 'position_embeddings': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            # collect attention_mask and position_ids
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]
            if "position_embeddings" in kwargs:
                cache["position_embeddings"] = kwargs["position_embeddings"]
            else:
                cache["position_embeddings"] = None
            raise ValueError  # stop the forward pass

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]
    position_embeddings = cache["position_embeddings"]
    model.config.use_cache = use_cache

    return inps, outs, attention_mask, position_ids, position_embeddings


def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    """
    Return the mask and current sparsity given an alpha value.

    Args:
        alpha (float): The alpha value.
        sort_res (Tensor): The sorted results.
        W_metric (Tensor): The weight metric.
        tmp_metric (Tensor): The temporary metric.
        sum_before (Tensor): The sum before the alpha value.

    Returns:
        Tuple[Tensor, float]: The mask and current sparsity.
    """
    thres_cumsum = sum_before * alpha
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1, 1))
    thres = torch.gather(
        sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True) - 1
    )
    W_mask = W_metric <= thres
    cur_sparsity = (W_mask == True).sum() / W_mask.numel()
    return W_mask, cur_sparsity


def llama_prune_wanda_(
    args,
    model: LlamaForCausalLM,
    tokenizer,
    device=torch.device("cuda:0"),
    prune_n=0,
    prune_m=0,
):
    """
    Perform Wanda pruning on a Llama model.

    Args:
        args: The arguments for pruning.
        model (LlamaForCausalLM): The model to prune.
        tokenizer: The tokenizer to use for calibration.
        device (torch.device, optional): The device to use for pruning. Defaults to torch.device("cuda:0").
        prune_n (int, optional): The number of elements to prune in each block. Defaults to 0.
        prune_m (int, optional): The size of each block. Defaults to 0.
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False

    with timeit_context("loading calibdation data"):
        dataloader, _ = get_loaders(
            "c4",
            nsamples=args.nsamples,
            seed=args.seed,
            seqlen=model.seqlen,
            tokenizer=tokenizer,
        )

    with torch.no_grad():
        # collect input to the first layer
        inps, outs, attention_mask, position_ids = prepare_calibration_input(
            model, dataloader, device
        )

    layers = model.model.layers
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        if (
            hasattr(model, "hf_device_map")
            and f"model.layers.{i}" in model.hf_device_map
        ):  ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = (
                inps.to(dev),
                outs.to(dev),
                attention_mask.to(dev) if attention_mask is not None else None,
                position_ids.to(dev) if position_ids is not None else None,
            )

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                cast(WrappedGPT, wrapped_layers[name]).add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )[0]
        for h in handles:
            h.remove()

        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(
                wrapped_layers[name].scaler_row.reshape((1, -1))
            )

            W_mask = (
                torch.zeros_like(W_metric) == 1
            )  ## initialize a mask to be all False
            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii : (ii + prune_m)].float()
                        W_mask.scatter_(
                            1,
                            ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1],
                            True,
                        )
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                if args.use_variant:
                    # wanda variant
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0.0, 0.8]
                    W_mask, cur_sparsity = return_given_alpha(
                        alpha, sort_res, W_metric, tmp_metric, sum_before
                    )
                    while (torch.abs(cur_sparsity - args.sparsity_ratio) > 0.001) and (
                        alpha_hist[1] - alpha_hist[0] >= 0.001
                    ):
                        if cur_sparsity > args.sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha

                        alpha = alpha_new
                        W_mask, cur_sparsity = return_given_alpha(
                            alpha, sort_res, W_metric, tmp_metric, sum_before
                        )
                    print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                else:
                    # unstructured pruning
                    indices = sort_res[1][
                        :, : int(W_metric.shape[1] * args.sparsity_ratio)
                    ]
                    W_mask.scatter_(1, indices, True)

            subset[name].weight.data[W_mask] = 0  ## set weights to zero

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()
