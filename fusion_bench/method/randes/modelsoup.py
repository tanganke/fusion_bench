import logging
from copy import deepcopy

import torch

from fusion_bench.modelpool import BaseModelPool
from fusion_bench.utils.parameters import count_parameters
from fusion_bench.utils.state_dict_arithmetic import (
    state_dict_mul,
)

from .base_algorithm import SuperposedAlgorithmBase, compare_models

log = logging.getLogger(__name__)


class SuperposedModelSoupAlgorithm(
    SuperposedAlgorithmBase,
):

    @torch.no_grad()
    def run(self, modelpool: BaseModelPool):
        if not isinstance(modelpool, BaseModelPool):
            modelpool = BaseModelPool(models=modelpool)

        log.info(
            f"Compressing models using superposed model soup.\n"
            f"Models: {modelpool.model_names}"
        )
        models = {}

        # load state dicts
        state_dicts = self._load_state_dicts(modelpool)
        with self.profile("load model"):
            pretrained_model = modelpool.load_model("_pretrained_")
        absorber_state_dict = self._compute_absorber(state_dicts, pretrained_model)
        if absorber_state_dict is not None:
            state_dicts["absorber"] = absorber_state_dict

        with self.profile("compress and retrieve"):
            retrieved_state_dicts, metadata = self._compress_and_retrieve(
                deepcopy(state_dicts), mode="superposed_model_soup"
            )

        with self.profile("retrieve models"):
            for model_idx, model_name in enumerate(modelpool.model_names):
                if self.ms_mode == "average":
                    coefficient = 1 / len(modelpool.model_names)
                    retrieved_state_dict = state_dict_mul(
                        retrieved_state_dicts[model_name], coefficient
                    )
                elif self.ms_mode == "original":
                    retrieved_state_dict = retrieved_state_dicts[model_name]
                else:
                    raise ValueError(f"Unsupported ms_mode: {self.ms_mode}")
                retrieved_model = modelpool.load_model(
                    model_name
                )  # TODO: avoid repeated loading
                # FIXME: for 'all' mode
                for k, v in retrieved_state_dict.items():
                    if v.shape[0] == 1:
                        retrieved_state_dict[k] = v.squeeze(0)
                retrieved_model.load_state_dict(retrieved_state_dict)
                models[model_name] = retrieved_model
                if self.debug >= 1:
                    with self.profile("metadata"):
                        if torch.cuda.is_available():
                            retrieved_state_dicts[model_name] = {
                                k: v.cuda()
                                for k, v in retrieved_state_dicts[model_name].items()
                            }
                            state_dicts[model_name] = {
                                k: v.cuda() for k, v in state_dicts[model_name].items()
                            }
                            retrieved_state_dict = {
                                k: v.cuda() for k, v in retrieved_state_dict.items()
                            }

                        target_layers = metadata["target_layers"]
                        # focus on the superposition retrieval performance on the target layers
                        metadata["superposed_model_retrieval_similarity"][
                            model_name
                        ] = compare_models(
                            retrieved_state_dicts[model_name],
                            state_dicts[model_name],
                            target_layers,
                        )
                        metadata["superposed_model_svd_subspace_similarities"][
                            model_name
                        ] = self._compute_svd_subspace_similarities(
                            state_dicts[model_name],
                            retrieved_state_dicts[model_name],
                            target_layers,
                        )
                        # overall retrieval performance
                        metadata["model_retrieval_similarity"][model_name] = (
                            compare_models(
                                retrieved_state_dict, state_dicts[model_name]
                            )
                        )
                        metadata["model_svd_subspace_similarities"][model_name] = (
                            self._compute_svd_subspace_similarities(
                                state_dicts[model_name], retrieved_state_dict
                            )
                        )
                        # delete the cuda tensors
                        del (
                            retrieved_state_dicts[model_name],
                            state_dicts[model_name],
                            retrieved_state_dict,
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
                print(
                    f"Total storage (Gbs) for retrieval and original: {metadata['total_gb_retrieved']} | {metadata['total_gb_original']}"
                )
        self.print_profile_summary()
        return {"models": models, "metadata": metadata}
