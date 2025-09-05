"""
Whoever Started the Interference Should End It:  Guiding Data-Free Model Merging via Task Vectors
Arxiv: http://arxiv.org/abs/2503.08099
"""

from typing import List

import torch
from tqdm import tqdm

from fusion_bench import BaseAlgorithm, BaseModelPool, auto_register_config
from fusion_bench.mixins import LightningFabricMixin
from fusion_bench.utils import timeit_context
from fusion_bench.utils.state_dict_arithmetic import state_dict_add, state_dict_sub


def wudi_merging(
    task_vectors: List[torch.Tensor],
    accelerator="cuda",
    iter_num: int = 300,
    exclude_keys: List[str] = None,
):
    exclude_keys = [] if exclude_keys is None else exclude_keys

    with timeit_context("WUDI Merging"):
        new_vector = {}
        for key in tqdm(task_vectors[0], desc="WUDI Merging", leave=False):
            tqdm.write(f"key: {key}")
            original_device = task_vectors[0][key].device
            tvs = torch.stack(
                [
                    task_vector[key].to(device=accelerator, non_blocking=True)
                    for task_vector in task_vectors
                ]
            )
            num_tvs = len(tvs)
            new_vector[key] = torch.nn.Parameter(torch.sum(tvs, dim=0))

            if len(task_vectors[0][key].shape) == 2 and key not in exclude_keys:
                optimizer = torch.optim.Adam([new_vector[key]], lr=1e-5, weight_decay=0)
                l2_norms = torch.square(
                    torch.norm(tvs.reshape(tvs.shape[0], -1), p=2, dim=-1)
                )
                for i in tqdm(
                    range(iter_num),
                ):
                    disturbing_vectors = new_vector[key].unsqueeze(0) - tvs
                    product = torch.matmul(disturbing_vectors, tvs.transpose(1, 2))
                    loss = torch.sum(
                        torch.square(product) / l2_norms.unsqueeze(-1).unsqueeze(-1)
                    )
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            else:
                new_vector[key] = new_vector[key] / num_tvs
            new_vector[key] = new_vector[key].to(
                device=original_device, non_blocking=True
            )
    return new_vector


@auto_register_config
class WUDIMerging(
    LightningFabricMixin,
    BaseAlgorithm,
):
    """
    Whoever Started the Interference Should End It:  Guiding Data-Free Model Merging via Task Vectors
    """

    def __init__(
        self,
        iter_num: int,
        exclude_keys: List[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

    def run(self, modelpool: BaseModelPool):
        # load the pretrained model and the task vectors of all the finetuned models
        with torch.no_grad():
            pretrained_model = modelpool.load_pretrained_model()
            task_vectors = []
            for model_name in modelpool.model_names:
                finetuned_model = modelpool.load_model(model_name)
                task_vectors.append(
                    state_dict_sub(
                        finetuned_model.state_dict(), pretrained_model.state_dict()
                    )
                )
                del finetuned_model  # free memory

        merged_tv = wudi_merging(
            task_vectors,
            accelerator=self.fabric.device,
            iter_num=self.iter_num,
            exclude_keys=self.exclude_keys,
        )

        pretrained_model.load_state_dict(
            state_dict_add(pretrained_model.state_dict(), merged_tv)
        )

        return pretrained_model
