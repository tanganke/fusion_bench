import logging
import os
from collections import OrderedDict
from copy import deepcopy
from typing import Optional

import torch

from fusion_bench.modelpool import BaseModelPool
from fusion_bench.utils.parameters import count_parameters
from fusion_bench.utils.state_dict_arithmetic import (
    state_dict_add,
    state_dict_mul,
    state_dict_sub,
)

from .base_algorithm import SuperposedAlgorithmBase, compare_models

log = logging.getLogger(__name__)


class SuperposedTaskArithmeticAlgorithm(
    SuperposedAlgorithmBase,
):
    _config_mapping = SuperposedAlgorithmBase._config_mapping | {
        "scaling_factor": "scaling_factor",
        "model_path": "model_path",
    }

    def __init__(
        self,
        scaling_factor: float,
        model_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.scaling_factor = scaling_factor
        self.model_path = model_path

    @torch.no_grad()
    def run(self, modelpool: BaseModelPool):
        if not isinstance(modelpool, BaseModelPool):
            modelpool = BaseModelPool(models=modelpool)

        log.info("Compressing models using superposed task arithmetic.")
        task_vector = None
        with self.profile("load model"):
            pretrained_model = modelpool.load_model("_pretrained_")

        # Calculate the task vector superposition
        task_vectors = {}
        models = {}
        for model_name in modelpool.model_names:
            with self.profile("load model"):
                model = modelpool.load_model(model_name)
            for layer_name, layer in model.state_dict(keep_vars=True).items():
                if self.verbose >= 1:
                    log.info(f"{layer_name} | {layer.shape}")
            task_vector = state_dict_sub(
                model.state_dict(keep_vars=True),
                pretrained_model.state_dict(keep_vars=True),
            )
            task_vectors[model_name] = task_vector

        with self.profile("compress and retrieve"):
            retrieved_task_vectors, metadata = self._compress_and_retrieve(
                deepcopy(task_vectors), mode="superposed_task_arithmetic"
            )
        with self.profile("retrieve models"):
            for model_name in modelpool.model_names:
                retrieved_task_vector = state_dict_mul(
                    retrieved_task_vectors[model_name], self.scaling_factor
                )
                retrieved_state_dict = state_dict_add(
                    pretrained_model.state_dict(keep_vars=True), retrieved_task_vector
                )
                retrieved_model = deepcopy(pretrained_model)
                # FIXME: for 'all' mode
                for k, v in retrieved_state_dict.items():
                    if v.shape[0] == 1:
                        retrieved_state_dict[k] = v.squeeze(0)
                retrieved_model.load_state_dict(retrieved_state_dict)
                models[model_name] = retrieved_model

                if self.debug >= 1:
                    with self.profile("metadata"):
                        model = modelpool.load_model(model_name)
                        if torch.cuda.is_available():
                            retrieved_state_dict = {
                                k: v.cuda() for k, v in retrieved_state_dict.items()
                            }
                            retrieved_task_vectors[model_name] = {
                                k: v.cuda()
                                for k, v in retrieved_task_vectors[model_name].items()
                            }
                            task_vectors[model_name] = {
                                k: v.cuda() for k, v in task_vectors[model_name].items()
                            }
                            model_state_dict = {
                                k: v.cuda()
                                for k, v in model.state_dict(keep_vars=True).items()
                            }
                        # target_layers = metadata['target_layers']
                        metadata["task_vector_retrieval_similarity"][model_name] = (
                            compare_models(
                                retrieved_task_vectors[model_name],
                                task_vectors[model_name],
                            )
                        )
                        metadata["task_vector_svd_subspace_similarities"][
                            model_name
                        ] = self._compute_svd_subspace_similarities(
                            task_vectors[model_name], retrieved_task_vectors[model_name]
                        )
                        # overall retrieval performance
                        metadata["model_retrieval_similarity"][model_name] = (
                            compare_models(retrieved_state_dict, model_state_dict)
                        )
                        metadata["model_svd_subspace_similarities"][model_name] = (
                            self._compute_svd_subspace_similarities(
                                model_state_dict, retrieved_state_dict
                            )
                        )
                        # delete the cuda tensors
                        del (
                            retrieved_state_dict,
                            retrieved_task_vectors[model_name],
                            task_vectors[model_name],
                            model_state_dict,
                        )

        with self.profile("metadata"):
            if self.debug >= 0:
                (
                    metadata["trainable_param_count_pretrained_model"],
                    metadata["active_param_count_pretrained_model"],
                ) = count_parameters(pretrained_model)
                (
                    metadata["trainable_param_count_retrieved_model"],
                    metadata["active_param_count_retrieved_model"],
                ) = count_parameters(models[modelpool.model_names[0]])
                metadata["nonzero_parameter_count"] += metadata[
                    "active_param_count_pretrained_model"
                ]
                metadata["total_gb_retrieved"] += metadata["total_gb_original"]
                print(
                    f"Total storage (Gbs) for retrieval and original: {metadata['total_gb_retrieved']} | {metadata['total_gb_original']}"
                )

        if self.model_path is not None:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            torch.save(models, self.model_path)

        self.print_profile_summary()
        return {"models": models, "metadata": metadata}


class SuperposedTaskArithmeticLoRAAlgorithm(
    SuperposedAlgorithmBase,
):
    _config_mapping = SuperposedAlgorithmBase._config_mapping | {
        "scaling_factor": "scaling_factor",
        "model_path": "model_path",
    }

    def __init__(
        self,
        scaling_factor: float,
        model_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.scaling_factor = scaling_factor
        self.model_path = model_path

    @torch.no_grad()
    def run(self, modelpool: BaseModelPool):
        if not isinstance(modelpool, BaseModelPool):
            modelpool = BaseModelPool(models=modelpool)

        log.info("Compressing models using superposed task arithmetic.")
        task_vector = None
        with self.profile("load model"):
            pretrained_model = modelpool.load_model("_pretrained_")

        # Calculate the task vector superposition
        loras = {}
        models = {}
        for model_name in modelpool.model_names:
            with self.profile("load model"):
                model = modelpool.load_model(model_name)
            for layer_name, layer in model.items():
                if self.verbose >= 1:
                    log.info(f"{layer_name} | {layer.shape}")
            # task_vector = state_dict_sub(
            #     model.state_dict(keep_vars=True),
            #     pretrained_model.state_dict(keep_vars=True),
            # )
            loras[model_name] = model

        with self.profile("compress and retrieve"):
            retrieved_loras, metadata = self._compress_and_retrieve(
                deepcopy(loras), mode="superposed_task_arithmetic"
            )
        with self.profile("retrieve models"):
            for model_name in modelpool.model_names:
                retrieved_lora = retrieved_loras[model_name]
                # retrieved_lora = state_dict_mul(retrieved_loras[model_name], self.config.scaling_factor)
                # retrieved_state_dict = state_dict_add(pretrained_model.state_dict(keep_vars=True), retrieved_lora)
                retrieved_model = deepcopy(pretrained_model)
                sd = retrieved_model.state_dict(keep_vars=True)
                # for layer_name, layer in sd.items():
                #     print(layer_name)
                # manually merge the lora back
                lora_weights = {}
                lora_weights_ready_to_merge = OrderedDict()
                for layer_name, layer in retrieved_lora.items():
                    parts = layer_name.split(".")
                    # print(parts)
                    base_name = ".".join(parts[2:-2] + [parts[-1]])
                    if base_name not in lora_weights:
                        lora_weights[base_name] = []
                    lora_weights[base_name].append(layer)
                for base_name, layers in lora_weights.items():
                    lora_weight = layers[-1] @ layers[0]
                    # sd[base_name] += lora_weight
                    lora_weights_ready_to_merge[base_name] = lora_weight

                retrieved_lora_ready = state_dict_mul(
                    lora_weights_ready_to_merge, self.config.scaling_factor
                )
                for layer_name, layer in retrieved_lora_ready.items():
                    sd[layer_name] += layer
                retrieved_model.load_state_dict(sd)
                models[model_name] = retrieved_model

                # # FIXME: for 'all' mode
                # for k, v in retrieved_state_dict.items():
                #     if v.shape[0] == 1:
                #         retrieved_state_dict[k] = v.squeeze(0)
                # retrieved_model.load_state_dict(sd)
                # models[model_name] = retrieved_model

                if self.debug >= 1:
                    with self.profile("metadata"):
                        model = modelpool.load_model(model_name)
                        if torch.cuda.is_available():
                            retrieved_state_dict = {
                                k: v.cuda() for k, v in retrieved_state_dict.items()
                            }
                            retrieved_loras[model_name] = {
                                k: v.cuda()
                                for k, v in retrieved_loras[model_name].items()
                            }
                            loras[model_name] = {
                                k: v.cuda() for k, v in loras[model_name].items()
                            }
                            model_state_dict = {
                                k: v.cuda()
                                for k, v in model.state_dict(keep_vars=True).items()
                            }
                        # focus on the superposition retrieval performance on the target layers
                        target_layers = metadata["target_layers"]
                        metadata["lora_retrieval_similarity"][model_name] = (
                            compare_models(
                                retrieved_loras[model_name],
                                loras[model_name],
                                target_layers,
                            )
                        )
                        metadata["lora_svd_subspace_similarities"][model_name] = (
                            self._compute_svd_subspace_similarities(
                                loras[model_name],
                                retrieved_loras[model_name],
                                target_layers,
                            )
                        )
                        # overall retrieval performance
                        metadata["model_retrieval_similarity"][model_name] = (
                            compare_models(retrieved_state_dict, model_state_dict)
                        )
                        metadata["model_svd_subspace_similarities"][model_name] = (
                            self._compute_svd_subspace_similarities(
                                model_state_dict, retrieved_state_dict
                            )
                        )
                        # delete the cuda tensors
                        del (
                            retrieved_state_dict,
                            retrieved_loras[model_name],
                            loras[model_name],
                            model_state_dict,
                        )

        with self.profile("metadata"):
            if self.debug >= 0:
                (
                    metadata["trainable_param_count_pretrained_model"],
                    metadata["active_param_count_pretrained_model"],
                ) = count_parameters(pretrained_model)
                (
                    metadata["trainable_param_count_retrieved_model"],
                    metadata["active_param_count_retrieved_model"],
                ) = count_parameters(models[modelpool.model_names[0]])
                metadata["nonzero_parameter_count"] += metadata[
                    "active_param_count_pretrained_model"
                ]
                metadata["total_gb_retrieved"] += metadata["total_gb_original"]
                print(
                    f"Total storage (Gbs) for retrieval and original: {metadata['total_gb_retrieved']} | {metadata['total_gb_original']}"
                )

        if self.model_path is not None:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            torch.save(models, self.model_path)

        self.print_profile_summary()
        return {"models": models, "metadata": metadata}
