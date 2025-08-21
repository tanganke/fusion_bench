import os
from typing import Literal

import pandas as pd
import torch
from tqdm import tqdm

from fusion_bench import BaseAlgorithm, BaseModelPool, auto_register_config
from fusion_bench.mixins import LightningFabricMixin, SimpleProfilerMixin


class ProjectedEnergyAnalysis(
    SimpleProfilerMixin,
    LightningFabricMixin,
    BaseAlgorithm,
):
    def on_run_start(self):
        self.device = self.fabric.device

    def run(self, modelpool: BaseModelPool):
        with self.profile("model loading"):
            base_model = modelpool.load_pretrained_model()

        results = {
            "model_name": [],
            "module_index": [],
            "module_name": [],
            "projected_energy_I": [],
            "projected_energy_II": [],
            "projected_energy_II_III": [],
        }
        for model_name in tqdm(
            modelpool.model_names,
            "analyzing",
            dynamic_ncols=True,
        ):
            with self.profile("model loading"):
                finetuned_model = modelpool.load_model(model_name)

            module_index = 0
            for module_name, base_module in tqdm(
                list(base_model.named_modules()),
                "analyzing modules",
                dynamic_ncols=True,
            ):
                if isinstance(base_module, torch.nn.Linear):
                    with self.profile("weight analysis"):
                        _result = self.analyze_weight(
                            base_module.weight,
                            finetuned_model.get_submodule(module_name).weight,
                        )
                    results["model_name"].append(model_name)
                    results["module_index"].append(module_index)
                    results["module_name"].append(module_name)
                    for key, value in _result.items():
                        results[key].append(value)

                    module_index += 1

        # save results as csv
        results = pd.DataFrame(results)
        results.to_csv(
            os.path.join(self.log_dir, "projected_energy_analysis.csv"), index=True
        )

        self.print_profile_summary()
        return None

    @torch.no_grad()
    def analyze_weight(self, w: torch.Tensor, w_ft: torch.Tensor, k: int = -1):
        w = w.to(dtype=torch.float32, device=self.device)
        w_ft = w_ft.to(dtype=torch.float32, device=self.device)
        w_diff = w_ft - w

        # Perform analysis on the weight tensor
        u, s, vh = torch.linalg.svd(w, full_matrices=False)
        v = vh.T
        if k < 0:
            # find the position where the sum of singular values is larger than 50% of the total sum
            cumsum = s.cumsum(0)
            k = (cumsum < cumsum[-1] * 0.5).sum().item() + 1

        # subspace I
        w_diff_proj = self._project_subspace_low(u=u, s=s, v=v, k=k, w=w, w_ft=w_ft)
        projected_energy_I = (
            torch.linalg.norm(w_diff_proj, ord="fro") ** 2
            / torch.linalg.norm(w_diff, ord="fro") ** 2
        )

        # subspace II
        w_diff_proj = self._project_subspace_high(u=u, s=s, v=v, k=k, w=w, w_ft=w_ft)
        projected_energy_II = (
            torch.linalg.norm(w_diff_proj, ord="fro") ** 2
            / torch.linalg.norm(w_diff, ord="fro") ** 2
        )

        ## subspace II+III
        u, s, vh = torch.linalg.svd(w, full_matrices=True)
        v = vh.T
        w_diff_proj = self._project_subspace_high(u=u, s=s, v=v, k=k, w=w, w_ft=w_ft)
        projected_energy_II_III = (
            torch.linalg.norm(w_diff_proj, ord="fro") ** 2
            / torch.linalg.norm(w_diff, ord="fro") ** 2
        )

        return {
            "projected_energy_I": projected_energy_I.item(),
            "projected_energy_II": projected_energy_II.item(),
            "projected_energy_II_III": projected_energy_II_III.item(),
        }

    def _project_subspace_low(
        self,
        u: torch.Tensor,
        s: torch.Tensor,
        v: torch.Tensor,
        k: int,
        w: torch.Tensor,
        w_ft: torch.Tensor,
    ):
        u = u[:, :k]
        s = s[:k]
        v = v[:, :k]

        w_diff = w_ft - w
        w_diff_proj = torch.linalg.multi_dot((u, u.T, w_diff, v, v.T))
        return w_diff_proj

    def _project_subspace_high(
        self,
        u: torch.Tensor,
        s: torch.Tensor,
        v: torch.Tensor,
        k: int,
        w: torch.Tensor,
        w_ft: torch.Tensor,
    ):
        u = u[:, k:]
        s = s[k:]
        v = v[:, k:]

        w_diff = w_ft - w
        w_diff_proj = torch.linalg.multi_dot((u, u.T, w_diff, v, v.T))
        return w_diff_proj
