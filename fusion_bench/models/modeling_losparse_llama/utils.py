import logging
from copy import deepcopy

from accelerate import init_empty_weights
from transformers import LlamaForCausalLM

from fusion_bench.utils import get_device, print_parameters

from .configuration_losparse_llama import LoSparseLlamaConfig
from .modeling_losparse_llama import LoSparseLlamaForCausalLM

log = logging.getLogger(__name__)


def convert_to_losparse_llama(
    model: LlamaForCausalLM,
    *,
    rank: int,
    low_gpu_memory_usage: bool = True,
):
    config = model.config
    new_config = LoSparseLlamaConfig(rank=rank, **config.to_dict())

    with init_empty_weights():
        new_model = LoSparseLlamaForCausalLM(new_config)

    # convert the model to the desired dtype and device
    new_model.to(dtype=model.dtype)

    if hasattr(model, "hf_device_map"):
        new_model.hf_device_map = model.hf_device_map
        if low_gpu_memory_usage:
            model.cpu()
        for k, v in model.hf_device_map.items():
            new_model.get_submodule(k).to_empty(device=v)
    else:
        device = get_device(model)
        if low_gpu_memory_usage:
            model.cpu()
        new_model.to_empty(device=device)

    # copy over the weights and buffers
    result = new_model.load_state_dict(model.state_dict(), strict=False)
    assert (
        len(result.unexpected_keys) == 0
    ), f"Unexpected keys: {result.unexpected_keys}"
    for name, buffer in model.named_buffers():
        new_model.get_buffer(name).data = buffer.data.to(
            new_model.get_buffer(name).device
        )

    # copy over the generation config
    new_model.generation_config = deepcopy(model.generation_config)

    # print the parameter counts
    log.info("parameters of the original model")
    print_parameters(model, print_fn=log.info)
    log.info("parameters of the new model")
    print_parameters(new_model, print_fn=log.info)
    return new_model
