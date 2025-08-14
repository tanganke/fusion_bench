import logging
import random
from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
from scipy.stats import ortho_group
from torch import Tensor, nn

from fusion_bench.method.base_algorithm import BaseAlgorithm
from fusion_bench.mixins.simple_profiler import SimpleProfilerMixin
from fusion_bench.modelpool import BaseModelPool
from fusion_bench.utils.parameters import get_parameter_summary, human_readable
from fusion_bench.utils.state_dict_arithmetic import state_dict_avg
from fusion_bench.utils.type import StateDictType

log = logging.getLogger(__name__)


def cosine_similarity(tensor1: Tensor, tensor2: Tensor) -> float:
    if tensor1.shape != tensor2.shape:
        raise ValueError("Matrices must have the same shape")
    vec1 = tensor1.flatten()
    vec2 = tensor2.flatten()
    dot_product = torch.sum(vec1 * vec2)
    norm1 = torch.sqrt(torch.sum(vec1**2))
    norm2 = torch.sqrt(torch.sum(vec2**2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)


def svd_and_partition(
    A: torch.Tensor, num_chunks: int = 3
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    U, S, V = torch.svd(A)
    singular_values = len(S)
    chunk_size = singular_values // num_chunks
    U_chunks, S_chunks, V_chunks = [], [], []

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = singular_values if i == num_chunks - 1 else start_idx + chunk_size

        U_chunks.append(U[:, start_idx:end_idx])
        S_chunks.append(S[start_idx:end_idx])
        V_chunks.append(V[:, start_idx:end_idx])

    return U_chunks, S_chunks, V_chunks


def compute_svd_subspace_similarity(
    ref: torch.Tensor, retrieval: torch.Tensor, num_chunks: int = 3
) -> List[dict]:
    if torch.cuda.is_available():
        ref = ref.cuda()
        retrieval = retrieval.cuda()
    U_chunks, S_chunks, V_chunks = svd_and_partition(ref, num_chunks)
    similarities = []
    for i in range(num_chunks):
        retrieval_approx = (
            U_chunks[i] @ U_chunks[i].T @ retrieval @ V_chunks[i] @ V_chunks[i].T
        )
        frob_sim = torch.norm(ref - retrieval_approx, p="fro").item() / ref.numel()
        cos_sim = cosine_similarity(ref, retrieval_approx)
        if isinstance(cos_sim, torch.Tensor):
            cos_sim = cos_sim.item()
        similarities.append(
            {
                "subspace": i + 1,
                "frobenius_similarity": frob_sim,
                "cosine_similarity": cos_sim,
            }
        )
    return similarities


def pairwise_cosine_similarity_matrix(tensors: List[torch.Tensor]) -> torch.Tensor:
    if torch.cuda.is_available():
        tensors = [tensor.cuda() for tensor in tensors]
    n = len(tensors)
    similarity_matrix = torch.zeros((n, n))
    for i in range(n):
        for j in range(n):
            similarity = cosine_similarity(tensors[i], tensors[j])
            similarity_matrix[i, j] = similarity.item()
    return similarity_matrix


def compare_models(
    state_dict1: StateDictType, state_dict2: StateDictType, target_layers=None
):
    results = {
        "layerwise_l2": {},
        "layerwise_cosine_similarity": {},
        "total_l2": None,
        "average_l2": None,
        "total_cosine_similarity": None,
        "average_cosine_similarity": None,
    }
    # Initialize lists to store flattened parameters
    params1_list = []
    params2_list = []

    keys1 = set(state_dict1.keys())
    keys2 = set(state_dict2.keys())
    # filter out layers that are not in target_layers
    if target_layers is not None:
        keys1 = keys1.intersection(target_layers)
        keys2 = keys2.intersection(target_layers)

    common_keys = keys1 & keys2
    if keys1 != keys2:
        print(
            "Warning: State dicts have different keys. Comparison will be made on common keys only."
        )
    num_layers = len(common_keys)

    for key in common_keys:
        tensor1 = state_dict1[key].float()
        tensor2 = state_dict2[key].float()

        # Compute L2 norm difference
        l2_diff = torch.norm(tensor1 - tensor2, p=2) / tensor1.numel()
        results["layerwise_l2"][key] = l2_diff.item()

        # Compute cosine similarity
        tensor1_flat = tensor1.reshape(-1)
        tensor2_flat = tensor2.reshape(-1)
        cos_sim = cosine_similarity(tensor1_flat, tensor2_flat).item()
        results["layerwise_cosine_similarity"][key] = cos_sim

        # Collect parameters for total metrics
        params1_list.append(tensor1_flat)
        params2_list.append(tensor2_flat)

    # Compute total metrics over all parameters
    if params1_list and params2_list:
        params1 = torch.cat(params1_list)
        params2 = torch.cat(params2_list)
        # Compute total L2 norm difference
        total_l2_difference = (
            torch.norm(params1 - params2, p=2).item() / params1.numel()
        )
        results["total_l2"] = total_l2_difference
        # Compute total cosine similarity
        total_cosine_similarity = cosine_similarity(params1, params2).item()
        results["total_cosine_similarity"] = total_cosine_similarity
    else:
        results["total_l2"] = None
        results["total_cosine_similarity"] = None

    # Compute average metrics
    if num_layers > 0:
        average_l2 = sum(results["layerwise_l2"].values()) / num_layers
        average_cosine_similarity = (
            sum(results["layerwise_cosine_similarity"].values()) / num_layers
        )
        results["average_l2"] = average_l2
        results["average_cosine_similarity"] = average_cosine_similarity
    else:
        results["average_l2"] = None
        results["average_cosine_similarity"] = None

    return results


class SuperposedAlgorithmBase(
    BaseAlgorithm,
    SimpleProfilerMixin,
):
    _config_mapping = BaseAlgorithm._config_mapping | {
        "mode": "mode",
        "target_layer": "target_layer",
        "random_seed": "random_seed",
        "different_across_layers": "different_across_layers",
        "joint_matrix_mode": "joint_matrix_mode",
        "rank": "rank",
        "random_components": "random_components",
        "shift_layers": "shift_layers",
        "absorber": "absorber",
        "debug": "debug",
        "ms_mode": "ms_mode",
        "verbose": "verbose",
        "dropout_rate": "dropout_rate",
    }

    def __init__(
        self,
        mode: str,
        target_layer: str,
        random_seed: int,
        different_across_layers: bool,
        joint_matrix_mode: str,
        rank: int,
        random_components: bool,
        shift_layers: int,
        absorber: Literal["average", "pretrained", "None"],
        debug: int,
        ms_mode: str,
        verbose: int,
        dropout_rate: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mode = mode
        self.target_layer = target_layer
        self.random_seed = random_seed
        self.different_across_layers = different_across_layers
        self.joint_matrix_mode = joint_matrix_mode
        self.rank = rank
        self.random_components = random_components
        self.shift_layers = shift_layers
        self.absorber = absorber
        self.debug = debug
        self.ms_mode = ms_mode
        self.verbose = verbose
        self.dropout_rate = dropout_rate

    def _compute_svd_subspace_similarities(
        self,
        original_state_dict: StateDictType,
        retrieved_state_dict: StateDictType,
        target_layers: Optional[List[str]] = None,
    ) -> dict:
        svd_similarities = {}
        for layer_name, original_param in original_state_dict.items():
            if target_layers is not None and layer_name not in target_layers:
                continue
            if (
                original_param.dim() == 2
            ):  # Only compute for 2D tensors (weight matrices)
                retrieved_param = retrieved_state_dict[layer_name]
                svd_similarities[layer_name] = compute_svd_subspace_similarity(
                    original_param.float(), retrieved_param.float()
                )
        return svd_similarities

    def _load_state_dicts(self, modelpool: BaseModelPool) -> Dict[str, StateDictType]:
        """
        Load the state dicts of the models in the modelpool.

        Args:
            modelpool (BaseModelPool): The modelpool to load the state dicts from.

        Returns:
            Dict[str, StateDictType]: A dictionary of state dicts, keyed by model name.
        """
        state_dicts = {}
        for model_name in modelpool.model_names:
            with self.profile("load model"):
                model = modelpool.load_model(model_name)
            state_dicts[model_name] = model.state_dict(keep_vars=True)
        return state_dicts

    def _compute_absorber(
        self,
        state_dicts: Dict[str, StateDictType],
        pretrained_model: Optional[nn.Module] = None,
    ) -> Optional[StateDictType]:
        """
        Compute the absorber state dict.

        Args:
            state_dicts (Dict[str, StateDictType]): The state dicts of the models, keyed by model name, i.e. `{model_name: state_dict}`.
            pretrained_model (Optional[nn.Module]): The pretrained model.

        Returns:
            Optional[StateDictType]: The absorber state dict.
        """
        if self.absorber == "average":
            return state_dict_avg(list(state_dicts.values()))
        elif self.absorber == "pretrained":
            return pretrained_model.state_dict(keep_vars=True)
        elif self.absorber == "None":
            return None
        else:
            raise ValueError(
                f"Unsupported absorber type: {self.absorber}. Must be one of 'average', 'pretrained', or 'None'."
            )

    @staticmethod
    def svd_decomposition(A, r):
        if torch.cuda.is_available():
            A = A.cuda()
        U, S, V = torch.svd(A)
        return (U[:, :r] @ torch.diag(S[:r])).cpu(), V.t()[:r, :].cpu()

    @staticmethod
    def svd_decomposition_bm(A, r_most, r_mid):
        if torch.cuda.is_available():
            A = A.cuda()

        # Perform SVD
        U, S, V = torch.svd(A)

        # Get the most significant 'r_most' dimensions
        U_most = U[:, :r_most]
        S_most = S[:r_most]
        V_most = V[:, :r_most]

        # Get the middle 'r_mid' dimensions
        start_mid = len(S) // 2 - r_mid // 2
        end_mid = start_mid + r_mid
        U_mid = U[:, start_mid:end_mid]
        S_mid = S[start_mid:end_mid]
        V_mid = V[:, start_mid:end_mid]

        # Combine the results into two sets
        U_combined = torch.cat([U_most, U_mid], dim=1)
        S_combined = torch.cat([S_most, S_mid])
        V_combined = torch.cat([V_most, V_mid], dim=1)

        return (U_combined @ torch.diag(S_combined)).cpu(), V_combined.t().cpu()

    @staticmethod
    def svd_decomposition(A, r=None, r_most=None, r_mid=None, random_components=False):
        """
        Perform SVD decomposition with options for:
        1. Truncated SVD with 'r' components (if r is provided and random_components=False).
        2. Most significant 'r_most' and middle 'r_mid' components (if r_most and r_mid are provided).
        3. Randomly selected 'r' components (if r is provided and random_components=True).

        Args:
            A (torch.Tensor): The input matrix to decompose.
            r (int, optional): Number of components for standard or random SVD.
            r_most (int, optional): Number of most significant components.
            r_mid (int, optional): Number of middle components.
            random_components (bool, optional): Whether to sample 'r' random components.

        Returns:
            (torch.Tensor, torch.Tensor): Two matrices resulting from the SVD decomposition.
        """
        if torch.cuda.is_available():
            A = A.cuda()

        # Perform SVD
        U, S, V = torch.svd(A)

        if r is not None and not random_components:
            # Standard SVD decomposition with 'r' components
            return (U[:, :r] @ torch.diag(S[:r])).cpu(), V.t()[:r, :].cpu()

        elif r_most is not None and r_mid is not None:
            # SVD decomposition with 'r_most' most significant and 'r_mid' middle components
            # Most significant components
            U_most = U[:, :r_most]
            S_most = S[:r_most]
            V_most = V[:, :r_most]

            # Middle components
            start_mid = len(S) // 2 - r_mid // 2
            end_mid = start_mid + r_mid
            U_mid = U[:, start_mid:end_mid]
            S_mid = S[start_mid:end_mid]
            V_mid = V[:, start_mid:end_mid]

            # Combine the most and middle components
            U_combined = torch.cat([U_most, U_mid], dim=1)
            S_combined = torch.cat([S_most, S_mid])
            V_combined = torch.cat([V_most, V_mid], dim=1)

            return (U_combined @ torch.diag(S_combined)).cpu(), V_combined.t().cpu()

        elif r is not None and random_components:
            # SVD decomposition with random 'r' components
            indices = torch.randperm(len(S))[:r]
            U_rand = U[:, indices]
            S_rand = S[indices]
            V_rand = V[:, indices]

            return (U_rand @ torch.diag(S_rand)).cpu(), V_rand.t().cpu()

        else:
            raise ValueError(
                "Invalid combination of arguments. Provide correct parameters."
            )

    @staticmethod
    def _get_rank(A, rank):
        if isinstance(rank, str):
            r1, r2 = rank.split("-")
            r1 = int(float(r1) * min(A.shape)) if "." in r1 else int(r1)
            r2 = int(float(r2) * min(A.shape)) if "." in r2 else int(r2)
            return r1, r2
        if isinstance(rank, int):
            return rank
        elif isinstance(rank, float):
            return int(rank * min(A.shape))

    def _target_layer_flag(self, layer: str):
        """
        The method takes a layer name as input and returns a boolean indicating whether this layer should be targeted.

        Current implementation assume Transformer architecture and layer number is the first number in the layer name.

        Args:
            layer (str): The name of the layer.

        Returns:
            bool: True if the layer should be targeted, False otherwise.
        """
        target_layers = self.target_layer  # e.g. ["mlp_w", "attn_w"]
        # TODO: figure out what wo is in flan-t5
        mlp_flag = "mlp" in layer or "Dense" in layer
        attn_flag = "attn" in layer or "Attention" in layer
        weight_flag = "weight" in layer
        bias_flag = "bias" in layer
        target_flags = []
        for target_layer in target_layers:
            if target_layer == "mlp_w":
                target_flags.append(mlp_flag and not bias_flag)
            elif target_layer == "attn_w":
                target_flags.append(attn_flag and not bias_flag)
            elif target_layer == "all":
                target_flags.append(True)
            elif target_layer == "mlp":
                target_flags.append(mlp_flag)
            elif target_layer == "attn":
                target_flags.append(attn_flag)
            else:
                raise ValueError(f"Unsupported target layer: {target_layer}")
        target_flag = any(target_flags)
        return target_flag

    def _compress_and_retrieve(self, state_dicts: Dict[str, StateDictType], mode: str):
        """
        Compress and retrieve the state dicts.

        Args:
            state_dicts (Dict[str, StateDictType]): The state dicts of the models, keyed by model name, i.e. `{model_name: state_dict}`.
            mode (str): The mode of the compression and retrieval.

        Returns:
            Dict[str, StateDictType]: The compressed and retrieved state dicts, keyed by model name, i.e. `{model_name: state_dict}`.
        """
        # Assume the state_dicts have the same layers.
        layers = state_dicts[list(state_dicts.keys())[0]].keys()
        models = list(state_dicts.keys())
        compressed_layers = {}
        compression_context = {model: {} for model in models}
        retrieval_context = {model: {} for model in models}
        retrieval_models = deepcopy(state_dicts)
        # target_layer_flags = [self._target_layer_flag(layer) for layer in layers]
        # implement target_layer_flags with dropout
        target_layer_flags: List[bool] = []
        count = 0
        for layer in layers:
            if self._target_layer_flag(layer):
                # take the target layer per `self.dropout_rate` target layers.
                # e.g. if self.dropout_rate = 2, then take the 2nd and 4th target layers, skip the first and third target layers.
                # if self.dropout_rate = 1, then take all target layers.
                count += 1
                if count == self.dropout_rate:
                    target_layer_flags.append(True)
                    count = 0
                else:
                    target_layer_flags.append(False)
            else:
                target_layer_flags.append(False)

        target_layers = [
            layer for layer, flag in zip(layers, target_layer_flags) if flag
        ]
        log.info(
            f"filtered {len(target_layers)} target layers out of {len(layers)} layers"
        )

        metadata = {
            "nonzero_parameter_count": 0,
            "nonzero_param_count_context": 0,
            "task_vector_retrieval_similarity": {},
            "superposed_model_retrieval_similarity": {},
            "model_retrieval_similarity": {},
            "target_layers": target_layers,
            "task_vector_svd_subspace_similarities": {},
            "superposed_model_svd_subspace_similarities": {},
            "model_svd_subspace_similarities": {},
            "total_param_count_original": 0,
            "total_gb_original": 0,
            "total_gb_retrieved": 0,
        }

        if "absorber" in models:
            models.remove("absorber")
            absorber = state_dicts["absorber"]
        else:
            absorber = None

        # get the total number of parameters and bytes (in GB) of the original model
        original_param_summary = get_parameter_summary(state_dicts[models[0]])
        gbs = original_param_summary["bytes"] / 1e9
        log.info(
            f"Total parameters: {human_readable(original_param_summary['all_param'])}"
        )
        log.info(f"Total gigabytes: {gbs}")
        metadata["total_param_count_original"] = original_param_summary["all_param"]
        metadata["total_gb_original"] = gbs

        # for analysis purposes
        if self.debug >= 2:
            test_models = models
            # test_models = models[:2]
            # layers_old = {model: OrderedDict() for model in models}
            layers_old = {model: deepcopy(state_dicts[model]) for model in models}
            tv_new = {
                model: {model: OrderedDict() for model in models}
                for model in test_models
            }
            # layers_new = {model: {model: OrderedDict() for model in models} for model in test_models}

        # Shift the layers
        # TODO: make this more robust to other models.
        if self.shift_layers != 0:
            # random shuffling. Do not shuffle layers with no number in their name.
            # because they are likely to be special layers like text embeddings.
            if self.shift_layers == -1:
                layer_mappings = {model: {} for model in models}
                temp_state_dicts = deepcopy(state_dicts)

                # get layer number index, assume the first number in the layer name is the layer number
                # assume all numbered layers have their number at the same index
                # assume components separated by '.' in the layer name
                found_digit = False
                for layer_idx, layer in enumerate(layers):
                    if target_layer_flags[layer_idx]:
                        layer_parts = layer.split(".")
                        for i, part in enumerate(layer_parts):
                            if part.isdigit():
                                layer_number_idx = i
                                break
                        if found_digit:
                            break

                # get groups of target layers with same name except the layer number
                target_layer_groups = {}
                for layer_idx, layer in enumerate(layers):
                    if target_layer_flags[layer_idx]:
                        layer_parts = layer.split(".")
                        if (
                            layer_number_idx >= len(layer_parts)
                            or not layer_parts[layer_number_idx].isdigit()
                        ):
                            continue  # skip layers without number
                        base_name = ".".join(
                            layer_parts[:layer_number_idx]
                            + layer_parts[layer_number_idx + 1 :]
                        )
                        layer_number = int(layer_parts[layer_number_idx])
                        if base_name not in target_layer_groups:
                            target_layer_groups[base_name] = []
                        target_layer_groups[base_name].append(layer_number)

                # construct random shuffled mapping
                random_shuffle_mapping = {model: {} for model in models}
                for model_idx, model in enumerate(models):
                    for layer_idx, layer in enumerate(layers):
                        if target_layer_flags[layer_idx]:
                            layer_parts = layer.split(".")
                            if (
                                layer_number_idx >= len(layer_parts)
                                or not layer_parts[layer_number_idx].isdigit()
                            ):
                                continue  # skip layers without number
                            base_name = ".".join(
                                layer_parts[:layer_number_idx]
                                + layer_parts[layer_number_idx + 1 :]
                            )

                            if base_name not in random_shuffle_mapping[model]:
                                rng_state = random.getstate()
                                # Shuffle the layer numbers differently for each model
                                random.seed(self.config.random_seed + model_idx)
                                shuffled_layer_numbers = target_layer_groups[
                                    base_name
                                ].copy()
                                random.shuffle(shuffled_layer_numbers)
                                random_shuffle_mapping[model][base_name] = {
                                    orig: str(shuffled)
                                    for orig, shuffled in zip(
                                        target_layer_groups[base_name],
                                        shuffled_layer_numbers,
                                    )
                                }
                                random.setstate(rng_state)

                    for layer_idx, layer in enumerate(layers):
                        if target_layer_flags[layer_idx]:
                            layer_parts = layer.split(".")
                            if (
                                layer_number_idx >= len(layer_parts)
                                or not layer_parts[layer_number_idx].isdigit()
                            ):
                                continue  # skip layers without number
                            base_name = ".".join(
                                layer_parts[:layer_number_idx]
                                + layer_parts[layer_number_idx + 1 :]
                            )
                            layer_number = int(layer_parts[layer_number_idx])
                            new_layer_number = random_shuffle_mapping[model][base_name][
                                layer_number
                            ]
                            new_layer_name = ".".join(
                                layer_parts[:layer_number_idx]
                                + [new_layer_number]
                                + layer_parts[layer_number_idx + 1 :]
                            )
                            temp_state_dicts[model][new_layer_name] = state_dicts[
                                model
                            ][layer]
                            layer_mappings[model][new_layer_name] = layer
                state_dicts = temp_state_dicts
            else:
                layer_numbers = {}
                for layer_idx, layer in enumerate(layers):
                    if target_layer_flags[layer_idx]:
                        layer_parts = layer.split(".")
                        for part in layer_parts:
                            if part.isdigit():
                                layer_numbers[layer] = int(part)
                                break  # Only consider the first number for each layer
                if layer_numbers:
                    max_layer_number = max(layer_numbers.values())
                else:
                    max_layer_number = 0
                temp_state_dicts = deepcopy(state_dicts)
                # Wrap around and shift each model by a different amount
                for model_idx, model in enumerate(models):
                    for layer_idx, layer in enumerate(layers):
                        target_flag = target_layer_flags[layer_idx]
                        if not target_flag:
                            continue
                        layer_number = layer_numbers.get(layer)
                        if layer_number is None:
                            continue
                        new_layer_number = (
                            layer_number + model_idx * self.config.shift_layers
                        ) % (max_layer_number + 1)
                        new_layer_parts = []
                        replaced = False  # Only replace the first numeric part FIXME: make it more robust
                        for part in layer.split("."):
                            if part.isdigit() and not replaced:
                                new_layer_parts.append(str(new_layer_number))
                                replaced = True
                            else:
                                new_layer_parts.append(part)
                        new_layer = ".".join(new_layer_parts)
                        temp_state_dicts[model][new_layer] = state_dicts[model][layer]
                state_dicts = temp_state_dicts

        if self.debug >= 2:
            # for evaluating pairwise cosine similarity
            unmerged_task_vectors = deepcopy(state_dicts)

        # compress
        for layer_idx, layer in enumerate(layers):
            shape = state_dicts[models[0]][layer].shape
            compressed_layer = None
            target_flag = target_layer_flags[layer_idx]
            # self.verbose = 1
            if self.verbose >= 1:
                log.info(f"{layer} | {shape} | {target_flag}")
            if not target_flag:
                if absorber is not None:
                    compressed_layer = absorber[layer]
                else:
                    for model in models:
                        if compressed_layer is None:
                            compressed_layer = deepcopy(state_dicts[model][layer])
                        else:
                            compressed_layer += deepcopy(state_dicts[model][layer])
            else:
                if self.mode == "random_binary_diagonal_matrix":
                    for model_idx, model in enumerate(models):
                        if self.different_across_layers:
                            seed = self.random_seed + model_idx + hash(layer) % 1e6
                        else:
                            seed = self.random_seed + model_idx
                        numpy_state = np.random.get_state()
                        np.random.seed(int(seed))
                        context = (
                            np.random.binomial(p=0.5, n=1, size=(1, shape[-1])).astype(
                                np.float32
                            )
                            * 2
                            - 1
                        )
                        context = torch.from_numpy(context)
                        np.random.set_state(numpy_state)
                        compression_context[model][
                            layer
                        ] = context  # for analysis purposes
                        retrieval_context[model][layer] = context
                        if compressed_layer is None:
                            compressed_layer = state_dicts[model][layer] * context
                        else:
                            compressed_layer += state_dicts[model][layer] * context
                        if self.debug >= 2:
                            # hadamard product is not linear, convert it back to diagonal matrix and apply matrix multiplication
                            context_diag = torch.diag(context.squeeze())
                            unmerged_task_vectors[model][layer] = (
                                unmerged_task_vectors[model][layer] @ context_diag
                            )
                elif self.mode == "random_rotation_matrix":
                    for model_idx, model in enumerate(models):
                        if self.different_across_layers:
                            seed = self.random_seed + model_idx + hash(layer) % 1e6
                        else:
                            seed = self.random_seed + model_idx
                        context = torch.from_numpy(
                            ortho_group.rvs(shape[-1], random_state=seed).astype(
                                "float32"
                            )
                        )
                        compression_context[model][
                            layer
                        ] = context  # for analysis purposes
                        retrieval_context[model][layer] = context.t()
                        if compressed_layer is None:
                            compressed_layer = state_dicts[model][layer] @ context
                        else:
                            compressed_layer += state_dicts[model][layer] @ context
                        if self.debug >= 2:
                            unmerged_task_vectors[model][layer] = (
                                unmerged_task_vectors[model][layer] @ context
                            )
                elif self.mode == "random_dense_matrix":
                    for model_idx, model in enumerate(models):
                        if self.different_across_layers:
                            seed = self.random_seed + model_idx + hash(layer) % 1e6
                        else:
                            seed = self.random_seed + model_idx
                        numpy_state = np.random.get_state()
                        np.random.seed(int(seed))
                        context = torch.from_numpy(
                            np.random.randn(shape[-1], shape[-1]).astype(np.float32)
                        )
                        np.random.set_state(numpy_state)
                        compression_context[model][
                            layer
                        ] = context  # for analysis purposes
                        retrieval_context[model][layer] = torch.linalg.pinv(
                            context.to("cuda")
                        ).to("cpu")
                        if compressed_layer is None:
                            compressed_layer = state_dicts[model][layer] @ context
                        else:
                            compressed_layer += state_dicts[model][layer] @ context
                        if self.debug >= 2:
                            unmerged_task_vectors[model][layer] = (
                                unmerged_task_vectors[model][layer] @ context
                            )
                elif self.mode == "random_diagonal_matrix":
                    for model_idx, model in enumerate(models):
                        if self.different_across_layers:
                            seed = self.random_seed + model_idx + hash(layer) % 1e6
                        else:
                            seed = self.random_seed + model_idx
                        numpy_state = np.random.get_state()
                        np.random.seed(int(seed))
                        context = torch.from_numpy(
                            np.random.randn(1, shape[-1]).astype(np.float32)
                        )
                        np.random.set_state(numpy_state)
                        compression_context[model][
                            layer
                        ] = context  # for analysis purposes
                        retrieval_context[model][layer] = 1 / context
                        if compressed_layer is None:
                            compressed_layer = state_dicts[model][layer] * context
                        else:
                            compressed_layer += state_dicts[model][layer] * context
                        if self.debug >= 2:
                            unmerged_task_vectors[model][layer] = (
                                unmerged_task_vectors[model][layer] * context
                            )
                elif self.mode == "identity_matrix":
                    for model_idx, model in enumerate(models):
                        context = torch.eye(shape[-1])
                        compression_context[model][
                            layer
                        ] = context  # for analysis purposes
                        retrieval_context[model][layer] = context
                        if compressed_layer is None:
                            compressed_layer = state_dicts[model][layer] @ context
                        else:
                            compressed_layer += state_dicts[model][layer] @ context
                        if self.debug >= 2:
                            unmerged_task_vectors[model][layer] = (
                                unmerged_task_vectors[model][layer] @ context
                            )
                else:
                    raise ValueError(f"Unsupported mode: {self.mode}")

            compressed_layers[layer] = compressed_layer

        # retrieve: for purpose of benchmarking, retrieve all models at once. In practice, retrieval should be done per model request.
        nonzero_param_count = 0
        nonzero_param_count_context = 0
        total_bytes_retrieved = 0

        if self.debug >= 2:
            for model in test_models:
                tv_new[model] = deepcopy(unmerged_task_vectors)

        for layer_idx, layer in enumerate(layers):
            shape = state_dicts[models[0]][layer].shape
            target_flag = target_layer_flags[layer_idx]
            if not target_flag:
                if mode == "superposed_model_soup":
                    # we don't count non-target layers for superposed task arithmetic
                    # because they can be absorbed into the pretrained weights
                    param_count = torch.numel(compressed_layers[layer])
                    total_bytes_retrieved += (
                        param_count * compressed_layers[layer].element_size()
                    )
                    nonzero_param_count += param_count
                for model in models:
                    retrieval_models[model][layer] = compressed_layers[layer]
            else:
                if (
                    mode == "superposed_task_arithmetic"
                    and self.mode == "identity_matrix"
                    and self.shift_layers == 0
                ):
                    # we don't count target layers for task arithmetic
                    # because they can be absorbed into the pretrained weights
                    pass
                else:
                    param_count = torch.numel(compressed_layers[layer])
                    total_bytes_retrieved += (
                        param_count * compressed_layers[layer].element_size()
                    )
                    nonzero_param_count += torch.numel(compressed_layers[layer])

                if self.mode in [
                    "random_binary_diagonal_matrix",
                    "random_rotation_matrix",
                    "random_dense_matrix",
                    "random_diagonal_matrix",
                    "identity_matrix",
                ]:
                    for model in models:
                        if self.mode not in ["identity_matrix"]:
                            nonzero_count = torch.numel(retrieval_context[model][layer])
                            if self.mode == "random_binary_diagonal_matrix":
                                total_bytes_retrieved += (
                                    nonzero_count * 1
                                )  # 1 byte per element for binary
                            else:
                                total_bytes_retrieved += (
                                    nonzero_count
                                    * retrieval_context[model][layer].element_size()
                                )
                            nonzero_param_count += nonzero_count
                            nonzero_param_count_context += nonzero_count
                        if retrieval_context[model][layer].shape[0] == 1:
                            retrieval_models[model][layer] = (
                                compressed_layers[layer]
                                * retrieval_context[model][layer]
                            )
                        else:
                            retrieval_models[model][layer] = (
                                compressed_layers[layer]
                                @ retrieval_context[model][layer]
                            )
                        if self.debug >= 2 and model in test_models:
                            if retrieval_context[model][layer].shape[0] == 1:
                                retrieval_context_diag = torch.diag(
                                    retrieval_context[model][layer].squeeze()
                                )
                                for m in models:
                                    tv_new[model][m][layer] = (
                                        tv_new[model][m][layer] @ retrieval_context_diag
                                    )
                            else:
                                for m in models:
                                    tv_new[model][m][layer] = (
                                        tv_new[model][m][layer]
                                        @ retrieval_context[model][layer]
                                    )
                else:
                    raise ValueError(f"Unsupported mode: {self.mode}")
        # for model in test_models:
        #     # print(retrieval_context[model]['vision_model.encoder.layers.4.self_attn.q_proj.weight'])
        # #     print('a')
        #     print(tv_new[model][models[3]]['vision_model.encoder.layers.4.self_attn.q_proj.weight'])

        # Shift the layers back
        if self.shift_layers != 0:
            if self.shift_layers == -1:  # random shuffling
                if self.debug >= 2:
                    temp_tv_new = deepcopy(tv_new)
                temp_retrieval_models = deepcopy(retrieval_models)
                for model_idx, model in enumerate(models):
                    # reverse_layer_mapping = {shuffled: original for original, shuffled in layer_mappings[model].items()}
                    for shuffled_layer, original_layer in layer_mappings[model].items():
                        temp_retrieval_models[model][original_layer] = retrieval_models[
                            model
                        ][shuffled_layer]
                        if self.debug >= 2 and model in test_models:
                            for m in models:
                                temp_tv_new[model][m][original_layer] = tv_new[model][
                                    m
                                ][shuffled_layer]
                retrieval_models = temp_retrieval_models
                if self.debug >= 2:
                    tv_new = temp_tv_new
            else:  # TODO: check the correctness of this mode
                # raise NotImplementedError("Shift back mode not implemented yet. No tv_new support yet.")
                if self.debug >= 2:
                    temp_tv_new = deepcopy(tv_new)
                temp_retrieval_models = deepcopy(retrieval_models)
                for model_idx, model in enumerate(models):
                    for layer_idx, layer in enumerate(layers):
                        target_flag = target_layer_flags[layer_idx]
                        if not target_flag:
                            continue
                        layer_parts = layer.split(".")
                        layer_number = None
                        for part in layer_parts:
                            if part.isdigit():
                                layer_number = int(part)
                                break  # Only consider the first number
                        if layer_number is None:
                            continue
                        new_layer_number = (
                            layer_number - model_idx * self.shift_layers
                        ) % (max_layer_number + 1)
                        new_layer_parts = []
                        replaced = False
                        for part in layer_parts:
                            if part.isdigit() and not replaced:
                                new_layer_parts.append(str(new_layer_number))
                                replaced = True  # Only replace the first numeric part
                            else:
                                new_layer_parts.append(part)
                        new_layer = ".".join(new_layer_parts)
                        temp_retrieval_models[model][new_layer] = retrieval_models[
                            model
                        ][layer]
                        if self.debug >= 2 and model in test_models:
                            for m in models:
                                temp_tv_new[model][m][new_layer] = tv_new[model][m][
                                    layer
                                ]
                retrieval_models = temp_retrieval_models
                if self.debug >= 2:
                    tv_new = temp_tv_new

        # for model in test_models:
        #     # print(retrieval_context[model]['vision_model.encoder.layers.4.self_attn.q_proj.weight'])
        # #     print('a')
        #     print(tv_new[model][models[3]]['vision_model.encoder.layers.4.self_attn.q_proj.weight'])

        # metadata
        if self.debug >= 2:
            if self.mode in [
                "random_binary_diagonal_matrix",
                "random_rotation_matrix",
                "random_dense_matrix",
                "random_diagonal_matrix",
                "identity_matrix",
            ]:
                layers = list(layers_old[models[0]].keys())
                layers_old_flattened = [
                    torch.cat([layers_old[model][layer].flatten() for layer in layers])
                    for model in models
                ]
                metadata["pairwise_cosine_similarity_matrix_before"] = (
                    pairwise_cosine_similarity_matrix(layers_old_flattened).tolist()
                )
                metadata["task_vector_dim"] = layers_old_flattened[0].shape[0]
                # layers_new = deepcopy(retrieval_models)
                rms = []
                for retrieval_model in test_models:
                    print(f"Retrieval model: {retrieval_model}")
                    layers_new_flattened = [
                        torch.cat(
                            [
                                tv_new[retrieval_model][model][layer].flatten()
                                for layer in layers
                            ]
                        )
                        for model in models
                    ]
                    rms.append(layers_new_flattened)
                    # print(layers_new_flattened[0][:50])
                    # layers_new_flattened = [torch.cat([layers_new[retrieval_model][layer].flatten() for layer in layers]) for model in models]
                    pcsm = pairwise_cosine_similarity_matrix(
                        layers_new_flattened
                    ).tolist()
                    print(pcsm)
                    metadata[
                        f"pairwise_cosine_similarity_matrix_after_{retrieval_model}"
                    ] = pcsm
        if self.debug >= 0:
            metadata["nonzero_parameter_count"] = (
                nonzero_param_count.item()
                if isinstance(nonzero_param_count, torch.Tensor)
                else nonzero_param_count
            )
            metadata["nonzero_param_count_context"] = (
                nonzero_param_count_context.item()
                if isinstance(nonzero_param_count_context, torch.Tensor)
                else nonzero_param_count_context
            )
            gbs = total_bytes_retrieved / 1e9
            metadata["total_gb_retrieved"] = gbs

        return retrieval_models, metadata
