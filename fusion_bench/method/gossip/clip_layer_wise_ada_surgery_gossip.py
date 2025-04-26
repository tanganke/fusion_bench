"""
Example Usage:

```bash
fusion_bench \
    method=adamerging \
        method.name=clip_layer_wise_adamerging \
        method.save_merging_weights=merging_weights.pt \
    modelpool=clip-vit-base-patch32_TA8 \
    taskpool=clip-vit-classification_TA8 \
    fabric_logger.root_dir=outputs/logs/ViT-B-32 \
    fabric_logger.name=clip_layer_wise_adamerging_adam
```
"""

import logging
import functools
from fusion_bench.mixins import CLIPClassificationMixin
from fusion_bench.compat.modelpool import ModelPool
from tqdm import tqdm
import gc
import torch
from copy import deepcopy

from ..surgery.surgerymodelwrapper import SurgeryModelWrapper
from .layer_wise_gossip import LayerWiseGossipAlgorithm
from .layer_wise_gossip import ModelScheduler
from .utils import get_memory_usage

from types import SimpleNamespace
from omegaconf import DictConfig

log = logging.getLogger(__name__)

def get_tensor_size(tensor):
    if tensor.is_cuda:
        return tensor.element_size() * tensor.nelement()
    return 0  # 如果不在GPU上，则不计入显存

def format_bytes(bytes_size):
    """
    将字节数转换为更易读的格式（KB, MB, GB）。
    
    :param bytes_size: 字节数
    :return: 格式化后的字符串
    """
    for unit in ['Bytes', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024
    return f"{bytes_size:.2f} PB"

def get_features_memory(features):
    batches_bytes = 0
    features_bytes = 0
    for name, datas in features.items():
        for data in datas:
            batches_bytes += get_tensor_size(data[0])
            features_bytes += get_tensor_size(data[1])
    log.info(f'batches_size: {format_bytes(batches_bytes)}')
    log.info(f'features_size: {format_bytes(features_bytes)}')

class Surgery_ModelScheduler(
    CLIPClassificationMixin,
    ModelScheduler,
):
    def __init__(self, modelpool, config, Algorithm):
        self.modelpool = modelpool
        super().__init__(config, modelpool=modelpool)
        self.surgery_steps = config.surgery_steps
        self._fabric_instance = Algorithm.fabric

        if self.modelpool.has_pretrained:
            clip_model = self.modelpool.load_clip_model("_pretrained_")
        else:
            clip_model = self.modelpool.load_clip_model(
                self.modelpool.model_names[0]
            )
        self.visual_projection = deepcopy(clip_model.visual_projection)
        self.visual_projection = self.fabric.to_device(self.visual_projection)

        self.construct_finetuned_data()

    @functools.cache
    def get_shuffled_test_loader_iter(self, task: str):
        return super().get_shuffled_test_loader_iter(
            task,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
        )

    def construct_finetuned_data(self):
        # pre calculate the outputs of the finetuned models
        self.finetuned_models_features = {
            name: [] for name in self.finetuned_models_name
        }
        
        # move models to GPU
        for model in self.finetuned_models:
            model = self.fabric.to_device(model)
            model.eval()

        log.info('constructing finetuned models features')
        with torch.no_grad():
            for i, (task, data) in enumerate(tqdm(self.finetuned_models_features.items())):
                for iteration in tqdm(range(self.surgery_steps)):
                    batch = next(self.get_shuffled_test_loader_iter(task))
                    feature = self.compute_features(self.finetuned_models[i], batch[0])
                    data.append((batch[0], feature))
        
        log.info(get_memory_usage('after constructing finetuned models features, the memory usage of GPU is:'))
    
        # move models back to CPU
        for model in self.finetuned_models:
            model.to('cpu')
            gc.collect()
        torch.cuda.empty_cache()
        log.info(get_memory_usage('after freeing memory, the memory usage of GPU is:'))

        get_features_memory(self.finetuned_models_features)

class ExtraHandlerMixin:
    def __init__(self, *args, **kwargs): # handle redundant parameters
        super().__init__()

class CLIP_LayerWise_Ada_Surgery_GossipAlgorithm(
    CLIPClassificationMixin,
    LayerWiseGossipAlgorithm,
):
    def __init__(self, **kwargs):
        super().__init__(DictConfig(kwargs))
    def on_test_time_adaptation_start(self):
        """
        Here we load the CLIP processor and construct the zero-shot classification head for each task.
        """
        if self.whether_setup_zero_shot_classification_head == False:
            self.setup_zero_shot_classification_head()

    @functools.cache
    def get_shuffled_test_loader_iter(self, task: str):
        return super().get_shuffled_test_loader_iter(
            task,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
        )

    def run(self, modelpool: ModelPool):
        """
        Run the Layer-Wise AdaMerging+Surgery+Gossip Algorithm.

        This method constructs the wrapped model and performs test-time adaptation if necessary.

        Args:
            modelpool (ModelPool): The model pool containing the pretrained and fine-tuned models.

        Returns:
            LayerWiseMergedModel: The merged model after test-time adaptation.
        """
        log.info("Fusing models using layer-wise adaptive surgery with gossip merging.")
        self.modelpool = modelpool
        self.log_hyperparams(self.config)
        self.num_finetuned_models = len(modelpool.model_names)
        datasets = [{dataset} for dataset in modelpool.model_names]

        with self.profile("construct the wrapped model"):
            model_scheduler = Surgery_ModelScheduler(modelpool=self.modelpool, config=self.config, Algorithm=self) 
            # we need to get the feature of fine-tune models

        if self.config.weights is not None:
            # skip the test-time adaptation
            return module.merge_and_unload()
        else:
            for step_idx in tqdm(
                range(self.config.gossip_max_steps),
                "Gossip merging",
                dynamic_ncols=True
            ):
                datasets = self.update_datasets(datasets)
                log.info(f'Gossip merging step:, {step_idx}')
                for model_id in tqdm(
                    range(self.num_finetuned_models),
                    "local admerging",
                    dynamic_ncols=True
                ):
                    if self.config.gossip_skip_adamerging == True:
                        # skip adamerging, only merge
                        with self.profile("construct the local wrapped model"):
                            module = model_scheduler(model_id)
                        log.info(f'skip adamerging, only merge ({modelpool.model_names[model_id]})')
                        model_scheduler.store_model(module.merge_weights(), model_id)
                        self.free_gpu_memory(module)
                    else:
                        with self.profile("construct the local wrapped model"):
                            module = model_scheduler(model_id)

                        if self.config.improve_dataset == True:
                            log.info(f'improved datasets, the datasets used in this local merging is {datasets[model_id]}')
                        else:
                            log.info(f'unimproved datasets, the datasets used in this local merging is {modelpool.model_names}')
                        with self.profile("test-time adaptation"):
                            module = self.test_time_adaptation(module, datasets[model_id])
                        # if self.config.get("save_merging_weights", False):
                        #     self.save_merging_weights(
                        #         self.config.save_merging_weights, module.merge_weight
                        #     )
                        model_scheduler.store_model(module.merge_weights(), model_id)
                        log.info(get_memory_usage(f'after local merging ({modelpool.model_names[model_id]}), the memory usage of GPU is:'))
                        self.free_gpu_memory(module) # simulate distributed GPU memory usage as much as possible

                model_scheduler.update_models()
                # if ((step_idx+1) % self.config.accuracy_test_interval == 0):
                #     self._program.evaluate_merged_model(self._program.taskpool,  model_scheduler.get_final_models())
                #     model_scheduler.move_to('cpu')

        # # test the result of Gossip
        # self._program.evaluate_merged_model(self._program.taskpool,  model_scheduler.get_final_models())
        # model_scheduler.move_to('cpu')


        log.info('start surgery')
        final_model = model_scheduler.get_final_models(0) # choose the first model as the final model

        alpha_model = SurgeryModelWrapper(final_model, modelpool.model_names, self)

        optimizer = torch.optim.Adam(alpha_model.collect_trainable_params(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0.)
        loss_func = torch.nn.L1Loss()

        finetuned_features = model_scheduler.finetuned_models_features
        for iteration in tqdm(
            range(self.config.surgery_steps),
            "surgery",
            dynamic_ncols=True,
        ):
            for dataset_name in modelpool.model_names:
                x = finetuned_features[dataset_name][iteration][0]
                outputs, features, _, _ = alpha_model(x, dataset_name)
                finetuned_feature = finetuned_features[dataset_name][iteration][1]

                loss = loss_func(features, finetuned_feature)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if ((iteration+1) % self.config.eval_iterations) == 0:
                # print(list(alpha_model.collect_trainable_params()))
                # Evaluate try to use the test module in fusion bench
                self._program.evaluate_merged_model(self._program.taskpool,  alpha_model)

        log.info('test the result of Gossip')
        # this can test the result of Gossip
        return model_scheduler.get_final_models()
    