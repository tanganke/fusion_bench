import functools
import logging
from copy import deepcopy
from typing import List, Tuple

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import CLIPModel, CLIPProcessor, CLIPVisionModel
from transformers.models.clip.modeling_clip import CLIPEncoder, CLIPEncoderLayer

from fusion_bench.dataset import CLIPDataset
from fusion_bench.method.task_arithmetic import task_arithmetic_merge
from fusion_bench.mixins import CLIPClassificationMixin
from fusion_bench.modelpool import CLIPVisionModelPool
from fusion_bench.models.sparse_we_moe import (
    SparseWeightEnsemblingMoE,
    SparseWeightEnsemblingMoE_ShardGate,
    construct_weight_ensembling_gate,
)
from fusion_bench.utils.data import InfiniteDataLoader

from .sparse_we_moe import SparseWeightEnsemblingMoEAlgorithm

log = logging.getLogger(__name__)


class SparseCLIPWeightEnsemblingMoEAlgorithm(
    SparseWeightEnsemblingMoEAlgorithm,
    CLIPClassificationMixin,
):
    modelpool: CLIPVisionModelPool = None

    def load_checkpoint(self, model, checkpoint):
        """
        Load the checkpoint file.
        """
        state = {"model": model}
        self._fabric.load(checkpoint, state)

    def save_checkpoint(self, model, checkpoint):
        """
        Save the checkpoint file.
        """
        self._fabric.save(checkpoint, {"model": model})

    def construct_moe_model(self) -> SparseWeightEnsemblingMoE:
        """
        Construct the Mixture of Experts model using the models in the model pool.
        """
        base_model = self.modelpool.load_model("_pretrained_")
        expert_models = [
            self.modelpool.load_model(m) for m in self.modelpool.model_names
        ]

        # merge the models using task arithmetic
        moe_model = task_arithmetic_merge(
            # this function modifies the model in place, so we need to pass a deepcopy
            deepcopy(base_model),
            expert_models,
            scaling_factor=self.config.init_lambda,
        ).requires_grad_(False)

        # up-scale MLP modules
        base_encoder: CLIPEncoder = base_model.vision_model.encoder
        moe_encoder: CLIPEncoder = moe_model.vision_model.encoder
        expert_encoders = [m.vision_model.encoder for m in expert_models]

        num_layers = len(base_encoder.layers)
        for layer_idx in range(num_layers):
            base_mlp = base_encoder.layers[layer_idx].mlp
            expert_mlps = [e.layers[layer_idx].mlp for e in expert_encoders]

            moe_encoder.layers[layer_idx].mlp = SparseWeightEnsemblingMoE(
                hidden_size=base_encoder.config.hidden_size,
                base_model=base_mlp,
                expert_models=expert_mlps,
                init_lambda=self.config.init_lambda,
                batch_first=True,  # for open_clip models this is False
                router_hidden_layers=self.config.router_hidden_layers,
                batch_reduce=self.config.batch_reduce,
                num_layers=num_layers,
                layer_idx=layer_idx,
                tv_prune_ratio=self.config.tv_prune_ratio,
            )

        return moe_model

    def construct_moe_model_sharedgate(self) -> SparseWeightEnsemblingMoE_ShardGate:
        """
        Construct the Mixture of Experts model using the models in the model pool with a shared gate.
        """
        base_model = self.modelpool.load_model("_pretrained_")
        expert_models = [
            self.modelpool.load_model(m) for m in self.modelpool.model_names
        ]

        # merge the models using task arithmetic
        moe_model = task_arithmetic_merge(
            # this function modifies the model in place, so we need to pass a deepcopy
            deepcopy(base_model),
            expert_models,
            scaling_factor=self.config.init_lambda,
        ).requires_grad_(False)

        # up-scale MLP modules
        base_encoder: CLIPEncoder = base_model.vision_model.encoder
        moe_encoder: CLIPEncoder = moe_model.vision_model.encoder
        expert_encoders = [m.vision_model.encoder for m in expert_models]

        # shared gate
        shared_gate = construct_weight_ensembling_gate(
            hidden_size=(
                base_encoder.config.hidden_size + self.config.position_encoding_dim
                if self.config.position_encoding
                else base_encoder.config.hidden_size
            ),
            num_experts=len(expert_models),
            init_lambda=self.config.init_lambda,
            num_hidden_layers=self.config.router_hidden_layers,
        )

        # ------------------------------------------------------------------------------------
        # Calculate magnitude
        # num_layers = len(base_encoder.layers)
        # exp_id = 0
        # for e in expert_encoders:
        #     for layer_idx in range(num_layers):
        #         if layer_idx in [0,3,5,7,9,11]:
        #             print(f"layer_idx: {layer_idx}")
        #             v_e = torch.cat([param.view(-1) for param in e.layers[layer_idx].mlp.parameters()])
        #             v_base = torch.cat([param.view(-1) for param in base_encoder.layers[layer_idx].mlp.parameters()])
        #             absolute_vector = torch.abs(v_e - v_base)
        #             np.save(f"/home/enneng/fusion_bench/outputs/sparse_we_moe/magnitude/absolute_vector_expert_{exp_id}_layer_{layer_idx}.npy", absolute_vector.detach().numpy())
        #     exp_id += 1
        # print('succ')
        # ------------------------------------------------------------------------------------

        # ------------------------------------------------------------------------------------
        # Calculate l2 distance and cos similarity
        # key = 'att' # 'mlp' or 'att'
        # num_layers = len(base_encoder.layers)
        # l2_distance_ss = []
        # cos_sim_ss = []
        # for e in expert_encoders:
        #     l2_distance_s = []
        #     cos_sim_s = []
        #     for layer_idx in range(num_layers):
        #         print(f"layer_idx: {layer_idx}")
        #         v_e = torch.cat([param.view(-1) for param in e.layers[layer_idx].mlp.parameters()]) if key == 'mlp' \
        #             else torch.cat([param.view(-1) for param in e.layers[layer_idx].self_attn.parameters()])
        #         v_base = torch.cat([param.view(-1) for param in base_encoder.layers[layer_idx].mlp.parameters()]) if key == 'mlp' \
        #             else torch.cat([param.view(-1) for param in base_encoder.layers[layer_idx].self_attn.parameters()])
        #         l2_distance = torch.norm(v_e - v_base, p=2)
        #         print(f"L2 Distance: {l2_distance}")
        #         cos_sim = torch.nn.functional.cosine_similarity(v_e, v_base, dim=0)
        #         print(f"Cosine Similarity: {cos_sim}")
        #
        #         l2_distance_s.append(l2_distance.item())
        #         cos_sim_s.append(cos_sim.item())
        #     l2_distance_ss.append(l2_distance_s)
        #     cos_sim_ss.append(cos_sim_s)
        #
        # print("L2 Distances:")
        # print(l2_distance_ss)
        # print("Cosine Similarity:")
        # print(cos_sim_ss)
        # ------------------------------------------------------------------------------------

        num_layers = len(base_encoder.layers)
        for layer_idx in range(num_layers):
            base_mlp = base_encoder.layers[layer_idx].mlp
            expert_mlps = [e.layers[layer_idx].mlp for e in expert_encoders]

            moe_encoder.layers[layer_idx].mlp = SparseWeightEnsemblingMoE_ShardGate(
                hidden_size=base_encoder.config.hidden_size,
                base_model=base_mlp,
                expert_models=expert_mlps,
                init_lambda=self.config.init_lambda,
                batch_first=True,  # for open_clip models this is False
                router_hidden_layers=self.config.router_hidden_layers,
                batch_reduce=self.config.batch_reduce,
                num_layers=num_layers,
                layer_idx=layer_idx,
                tv_prune_ratio=self.config.tv_prune_ratio,
                sharedgate=shared_gate,
                position_encoding=self.config.position_encoding,
                position_encoding_dim=self.config.position_encoding_dim,
            )

        return moe_model

    @functools.cache
    def get_shuffled_test_loader_iter(self, tta_dataset: str):
        """
        Get an iterator for the shuffled test data loader.
        """
        log.info("get_shuffled_test_loader_iter")
        dataset = self.modelpool.load_test_dataset(tta_dataset)
        dataset = CLIPDataset(dataset, processor=self.clip_processor)
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )
        if self._fabric is not None:
            loader = self._fabric.setup_dataloaders(loader)
        return iter(InfiniteDataLoader(loader))

    def on_test_time_adaptation_start(self):
        """
        Here we load the CLIP processor and construct the zero-shot classification head for each task.
        """
        self.setup_zero_shot_classification_head()

    def compute_logits(
        self, module: CLIPVisionModel, batch: Tuple[Tensor, Tensor], task: str
    ) -> Tensor:
        """
        Compute the logits for the given batch and task.

        Args:
            module (CLIPVisionModel): The vision model to use for computing logits.
            batch (Tuple[Tensor, Tensor]): The batch of data.
            task (str): The task for which to compute logits.

        Returns:
            Tensor: The computed logits.
        """
        images, _ = batch
        text_embeds = self.zeroshot_weights[task]

        image_embeds = module(images)[1]
        image_embeds = self.visual_projection(image_embeds)

        # normalize embeddings
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity
        logits_per_text = (
            torch.matmul(text_embeds, image_embeds.t()) * self.logit_scale_exp
        )
        logits_per_image = logits_per_text.t()

        return logits_per_image
