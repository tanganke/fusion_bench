
import functools
import logging

from torch.utils.data import DataLoader
import torch

from fusion_bench.dataset.clip_dataset import CLIPDataset
from fusion_bench.mixins import CLIPClassificationMixin
from fusion_bench.compat.modelpool import ModelPool

import copy
import gc
from tqdm import tqdm


from ..surgery.surgerymodelwrapper import SurgeryModelWrapper
from .layer_wise_adamerging import LayerWiseAdaMergingAlgorithm
from .utils import get_memory_usage

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

    def run(self, modelpool: ModelPool, **kwargs):
        """
        Run the Layer-Wise AdaMerging+Aurgery Algorithm.

        This method constructs the wrapped model and performs test-time adaptation if necessary. Then, it will perform surgery.

        Args:
            modelpool (ModelPool): The model pool containing the pretrained and fine-tuned models.

        Returns:
            LayerWiseMergedModel: The merged model after test-time adaptation.
        """
        log.info("Fusing models using layer-wise adaptive merging.")
        self.modelpool = modelpool
        self.log_hyperparams(self.config)

        with self.profile("construct the wrapped model"):
            module = self.construct_layer_wise_merged_model(modelpool)

        if self.config.weights is not None:
            # skip the test-time adaptation
            merged_model = copy.deepcopy(module.merge_and_unload())
        else:
            with self.profile("test-time adaptation"):
                module = self.test_time_adaptation(module)
            if self.config.get("save_merging_weights", False):
                self.save_merging_weights(
                    self.config.save_merging_weights, module.merge_weight
                )
            merged_model = copy.deepcopy(module.merge_and_unload())


        log.info('start performing Surgery')

        # free memory
        del module
        gc.collect()
        torch.cuda.empty_cache()
        alpha_model = SurgeryModelWrapper(merged_model, modelpool.model_names, self)
        log.info(get_memory_usage('after freeing memory, the memory usage of GPU is:'))

        optimizer = torch.optim.Adam(alpha_model.collect_trainable_params(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0.)
        loss_func = torch.nn.L1Loss()

        finetuned_models = {model_name: modelpool.load_model(model_name) for model_name in modelpool.model_names}
        for name, model in finetuned_models.items():
            model = self.fabric.to_device(model)
            model.eval()

        for iteration in tqdm(
            range(self.config.surgery_steps),
            "surgery",
            dynamic_ncols=True,
        ):
            for dataset_name in modelpool.model_names:
                batch = next(self.get_shuffled_test_loader_iter(dataset_name))
                finetuned_feature = self.compute_features(finetuned_models[dataset_name], batch[0])
                outputs, features, _, _ = alpha_model(batch[0], dataset_name)

                loss = loss_func(features, finetuned_feature)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if ((iteration+1) % self.config.eval_iterations) == 0:
                # print(list(alpha_model.collect_trainable_params()))
                # Evaluate try to use the test module in fusion bench
                self._program.evaluate_merged_model(self._program.taskpool,  alpha_model)

        log.info('test the result of Adamerging')
        return merged_model