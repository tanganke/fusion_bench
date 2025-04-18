"""
Implementation of the Layer-Wise AdaMerging+Surgery Algorithm.

For more details, please refer to:

- (ICLR 2024) Yang, et.al. AdaMerging: Adaptive Model Merging for Multi-Task Learning. http://arxiv.org/abs/2310.02575
- (ICML 2024) Yang, et.al. Representation Surgery for Multi-Task Model Merging. https://arxiv.org/abs/2402.02705

Basic Example:

```shell
fusion_bench \
    method=surgery/adamerging_surgery \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8 \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8
```
"""

import copy
import functools
import gc
import logging
from typing import TYPE_CHECKING, cast

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPVisionModel

from fusion_bench.dataset.clip_dataset import CLIPDataset
from fusion_bench.method.adamerging.layer_wise_adamerging import (
    LayerWiseAdaMergingAlgorithm,
)
from fusion_bench.method.adamerging.utils import get_memory_usage
from fusion_bench.mixins import CLIPClassificationMixin
from fusion_bench.modelpool import CLIPVisionModelPool
from fusion_bench.models.surgery.surgerymodelwrapper import SurgeryModelWrapper
from fusion_bench.models.wrappers.layer_wise_fusion import LayerWiseMergedModel

log = logging.getLogger(__name__)


class CLIPLayerWiseAdaMergingSurgeryAlgorithm(
    CLIPClassificationMixin,
    LayerWiseAdaMergingAlgorithm,
):

    def on_test_time_adaptation_start(self):
        """
        Here we load the CLIP processor and construct the zero-shot classification head for each task.
        """
        self.setup_zero_shot_classification_head()

    @functools.cache
    def get_shuffled_test_loader_iter(self, task: str):
        return super().get_shuffled_test_loader_iter(
            task,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
        )

    def run(self, modelpool: CLIPVisionModelPool, **kwargs):
        """
        Run the Layer-Wise AdaMerging+Surgery Algorithm.

        This method constructs the wrapped model and performs test-time adaptation if necessary. Then, it will perform surgery.

        Args:
            modelpool (ModelPool): The model pool containing the pretrained and fine-tuned models.

        Returns:
            LayerWiseMergedModel: The merged model after test-time adaptation.
        """
        log.info("Fusing models using layer-wise adaptive merging.")
        self.modelpool = modelpool
        self.log_hyperparams(self.config)

        # === Start of the AdaMerging Algorithm ===
        with self.profile("construct the wrapped model"):
            module = cast(
                LayerWiseMergedModel[CLIPVisionModel],
                self.construct_layer_wise_merged_model(modelpool),
            )

        if self.config.weights is not None:
            # skip the test-time adaptation
            merge_weight: torch.Tensor = torch.load(self.config.weights)
            module.merge_weight.data = merge_weight.to(
                device=module.merge_weight.device
            )
            merged_model = copy.deepcopy(module.merge_and_unload())
            # setup the zero-shot classification head
            self.on_test_time_adaptation_start()

        else:
            with self.profile("test-time adaptation"):
                module = self.test_time_adaptation(module)
            if self.config.get("save_merging_weights", False):
                self.save_merging_weights(
                    self.config.save_merging_weights, module.merge_weight
                )
            merged_model = copy.deepcopy(module.merge_and_unload())

        # free memory
        del module
        gc.collect()
        torch.cuda.empty_cache()

        # === Start of the Surgery Algorithm ===
        log.info("start performing Surgery")
        alpha_model = SurgeryModelWrapper(
            merged_model,
            modelpool.model_names,
            projection_dim=merged_model.config.projection_dim,
        )
        alpha_model = self.fabric.setup(alpha_model)
        log.info(get_memory_usage("after freeing memory, the memory usage of GPU is:"))

        optimizer = torch.optim.Adam(
            alpha_model.collect_trainable_params(),
            lr=1e-3,
            betas=(0.9, 0.999),
            weight_decay=0.0,
        )

        finetuned_models = {
            model_name: modelpool.load_model(model_name)
            for model_name in modelpool.model_names
        }
        for name, model in finetuned_models.items():
            model.requires_grad_(False)
            model = self.fabric.to_device(model)
            model.eval()

        for iteration in tqdm(
            range(self.config.surgery_steps),
            "surgery",
            dynamic_ncols=True,
        ):
            for dataset_name in modelpool.model_names:
                batch = next(self.get_shuffled_test_loader_iter(dataset_name))
                finetuned_feature = self.compute_features(
                    finetuned_models[dataset_name], batch[0]
                )
                features, _, _ = alpha_model.compute_surgery_features(
                    lambda model: self.compute_features(model, batch[0]),
                    dataset_name,
                )

                loss = F.l1_loss(features, finetuned_feature)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if ((iteration + 1) % self.config.eval_iterations) == 0:
                # print(list(alpha_model.collect_trainable_params()))
                # Evaluate try to use the test module in fusion bench
                log.info(f"iteration: {iteration+1}")
                self._program.evaluate_merged_model(self._program.taskpool, alpha_model)

        log.info("test the result of Adamerging")
        return {"adamerging": merged_model, "surgery": alpha_model}
