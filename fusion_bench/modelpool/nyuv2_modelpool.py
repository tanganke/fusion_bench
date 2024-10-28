import logging

import torch
from omegaconf import DictConfig
from torch import nn

from fusion_bench.compat.modelpool.base_pool import ModelPool
from fusion_bench.dataset.nyuv2 import NYUv2
from fusion_bench.models.nyuv2.aspp import DeepLabHead
from fusion_bench.models.nyuv2.lightning_module import NYUv2Model
from fusion_bench.models.nyuv2.resnet_dilated import ResnetDilated, resnet_dilated

log = logging.getLogger(__name__)


class NYUv2ModelPool(ModelPool):
    def load_model(
        self, model_config: str | DictConfig, encoder_only: bool = True
    ) -> ResnetDilated | NYUv2Model:
        if isinstance(model_config, str):
            model_config = self.get_model_config(model_config)

        encoder = resnet_dilated(model_config.encoder)
        decoders = nn.ModuleDict(
            {
                task: DeepLabHead(2048, NYUv2.num_out_channels[task])
                for task in model_config.decoders
            }
        )
        model = NYUv2Model(encoder=encoder, decoders=decoders)
        if model_config.get("ckpt_path", None) is not None:
            ckpt = torch.load(model_config.ckpt_path, map_location="cpu")
            if "state_dict" in ckpt:
                ckpt = ckpt["state_dict"]
            model.load_state_dict(ckpt, strict=False)

        if encoder_only:
            return model.encoder
        else:
            return model
