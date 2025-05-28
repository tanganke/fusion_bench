import copy
import logging
import os
import time
from copy import deepcopy
from re import U
from typing import TYPE_CHECKING, Dict, List, Tuple  # noqa: F401

import numpy as np
import torch
import torch.nn.functional as F
from joblib import Memory
from omegaconf import OmegaConf
from torch import Tensor, nn
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModelForCausalLM

from fusion_bench.compat.modelpool.base_pool import DictModelPool
from fusion_bench.method import BaseAlgorithm
from fusion_bench.method.s2_moe.utils import TSVC_utils, TSVM_utils
from fusion_bench.method.simple_average import simple_average
from fusion_bench.mixins.simple_profiler import SimpleProfilerMixin
from fusion_bench.modelpool import BaseModelPool, CausalLMPool
from fusion_bench.models.modeling_s2_moe_llama import (
    S2MoELlamaConfig,
    S2MoELlamaForCausalLM,
)
from fusion_bench.models.modeling_s2_moe_llama.modeling_s2_moe_llama import (
    S2MoELlamaDecoderLayer,
)
from fusion_bench.models.s2_moe.s2moelinear_from_hf_config import (
    S2MoELinear,
    upscale_to_s2moe_linear,
)
from fusion_bench.models.s2_moe.sparse_linear import SparseLinear
from fusion_bench.models.smile_moe.linear_from_module import ExpertNotTrainedError
from fusion_bench.models.smile_moe.utils import _is_all_zeros, svd
from fusion_bench.models.utils import get_attr, set_attr
from fusion_bench.utils.parameters import print_parameters

if TYPE_CHECKING:
    from transformers.models.llama.modeling_llama import LlamaForCausalLM

log = logging.getLogger(__name__)

memory = Memory("outputs/cache", verbose=0)


class S2MoEUpscalingAlgorithmForLlama(
    SimpleProfilerMixin,
    BaseAlgorithm,
):
    _linear_layer_cls = (nn.Linear,)
    _config_mapping = BaseAlgorithm._config_mapping | {
        "device": "device",
        "upscaling_accelerator": "upscaling_accelerator",
        "full_matrices": "full_matrices",
        "top_k": "top_k",
        "routing_use_diff": "routing_use_diff",
        "average_experts": "average_experts",
        "model_path": "model_path",
        "model_save_path": "model_save_path",
        "threshold": "threshold",  # 新增参数
        "use_aparse_expert": "use_aparse_expert",
        "sparsity_ratio": "sparsity_ratio",
    }

    def __init__(
        self,
        *,
        device: str = "cuda",
        upscaling_accelerator: str = None,
        full_matrices: bool = True,
        top_k: int = 2,
        threshold: float = 0.1,  # 新增参数
        routing_use_diff: bool = True,
        average_experts: bool = False,
        model_path: str = None,
        model_save_path: str = None,
        use_aparse_expert: bool = True,
        sparsity_ratio: float = 0.8,
        **kwargs,
    ):
        """
        初始化SmileUpscalingAlgorithm。

        Args:
            device (str): 用于计算的设备。
            upscaling_accelerator (str): 用于SVD计算的设备。
            full_matrices (bool): 是否计算完整大小的U和V矩阵。
            top_k (int): 选择的顶部专家数量。
            threshold (float): 激活任务子空间的阈值。
            routing_use_diff (bool): 是否使用权重差异进行路由。
            average_experts (bool): 是否平均专家。
            model_path (str): 保存/加载模型的路径。
            use_aparse_expert (bool): 是否使用稀疏专家。
            sparsity_ratio (float): 稀疏性比率。
            **kwargs: 额外参数。
        """
        super().__init__()
        self.device = device
        self.upscaling_accelerator = upscaling_accelerator
        self.full_matrices = full_matrices
        self.top_k = top_k
        self.threshold = threshold  # 新增参数
        self.routing_use_diff = routing_use_diff
        self.average_experts = average_experts
        self.model_path = model_path
        self.rout_svd_cache = None
        self.use_aparse_expert = use_aparse_expert
        self.sparsity_ratio = sparsity_ratio
        self.model_save_path = model_save_path
        
        for key, value in kwargs.items():
            log.warning(f"Unrecognized argument: {key}")
            setattr(self, key, value)

        # 打印配置
        print(f"=== Config for `{type(self).__name__}` ===")
        print(OmegaConf.to_yaml(self.config))
        print(f"=== Config for `{type(self).__name__}` ===")

    @torch.no_grad()
    def run(self, modelpool: CausalLMPool):
        """
        Executes the upscaling process.

        Args:
            modelpool (ModelPool): The pool of models to be used for upscaling.

        Returns:
            nn.Module: The upscaled model.
        """

        if not isinstance(modelpool, BaseModelPool):
            modelpool = BaseModelPool(modelpool)

        self.modelpool = modelpool

        if self.model_path is not None and os.path.exists(self.model_path):
            log.info(f"Loading model from {self.model_path}")
            model = AutoModelForCausalLM.from_pretrained(self.model_path)
            print_parameters(model)
            return model

        with self.profile("loading model"):
            # load models and move to GPU if available
            with self.profile("load pretrained model"):
                pretrained_model = modelpool.load_model("_pretrained_")
            with self.profile("load fine-tuned model"):
                finetuned_models = [
                    m
                    for m in tqdm(modelpool.models(), total=len(modelpool.model_names))
                ]

            if self.config.device == "cuda" and torch.cuda.is_available():
                pretrained_model = pretrained_model.cuda()
                finetuned_models = [m.cuda() for m in finetuned_models]

        # pretrained_model_orig = copy.deepcopy(pretrained_model)
        pretrained_model, orig_v = memory.cache(
            lambda: self.tsv_m(pretrained_model, finetuned_models)
        )()

        with self.profile("merge model"):
            model = self.merge(pretrained_model, finetuned_models, orig_v)

        self.print_profile_summary()
        if self.config.model_save_path is not None:
            os.makedirs(os.path.dirname(self.config.model_save_path), exist_ok=True)
            log.info(f"Saving model to {self.config.model_save_path}")
            model.save_pretrained(self.config.model_save_path)
        
        for name, module in model.named_modules():
            if isinstance(module, SparseLinear):
                module.apply_pruning_()
        print_parameters(model)
        return model

    def merge(
        self,
        pretrained_model: nn.Module,
        finetuned_models: List[nn.Module],
        orig_v,
        in_place: bool = True,
    ):
        """
        Merges the pretrained model with the fine-tuned models to create an upscaled model.

        Args:
            pretrained_model (nn.Module): The pretrained model.
            finetuned_models (List[nn.Module]): A list of fine-tuned models.
            in_place (bool): If True, modifies the pretrained model in place. Otherwise, creates a copy.

        Returns:
            nn.Module: The merged model.
        """

        if in_place:
            model = pretrained_model
        else:
            model = deepcopy(pretrained_model)
        print(
            "#############################create model with S2MoELlamaForCausalLM############################"
        )

        pretrained_model_config = self.modelpool.get_model_config("_pretrained_")
        if isinstance(pretrained_model_config, str):
            pretrained_path = pretrained_model_config
        else:
            pretrained_path = pretrained_model_config.get(
                "path", pretrained_model_config["pretrained_model_name_or_path"]
            )
        base_config = AutoConfig.from_pretrained(pretrained_path)
        # print("orig_v: ",orig_v)
        # print("orig_v [0] shape:",orig_v[0].shape)
        # import sys
        # sys.exit()
        model_config = S2MoELlamaConfig(
            num_experts_per_tok=self.top_k,
            num_local_experts=len(finetuned_models),
            use_sparse_expert=self.use_aparse_expert,
            sparsity_ratio=self.sparsity_ratio,
            **base_config.to_dict(),
        )
        model = S2MoELlamaForCausalLM(model_config)
        model.to(dtype=pretrained_model.dtype).to_empty(device="cpu")

        # copy pretrained model weights
        state_dict = model.state_dict()
        pretrained_state_dict = dict(pretrained_model.state_dict())
        for key in list(pretrained_state_dict.keys()):
            if key not in state_dict:
                pretrained_state_dict.pop(key)
        model.load_state_dict(pretrained_state_dict, strict=False)

        pretrained_model_linears = [
            (name, module)
            for name, module in list(pretrained_model.named_modules())[1:-1]
            if isinstance(module, self._linear_layer_cls)
        ]

        # upscale model
        for layer_idx, (name, module) in tqdm(
            enumerate(pretrained_model_linears),
            "Upscaling Modules (layer)",
            dynamic_ncols=True,
        ):
            name_list = name.split(".")
            pretrained_layer = get_attr(pretrained_model, name_list)
            finetuned_layers = [get_attr(m, name_list) for m in finetuned_models]
            target_layer = get_attr(model, name_list)

            try:
                upscale_to_s2moe_linear(
                    base=pretrained_layer,
                    experts=finetuned_layers,
                    target=target_layer,
                    orig_v=orig_v[layer_idx],
                    use_sparse_expert=self.use_aparse_expert,
                    sparsity_ratio=self.sparsity_ratio,
                )
            except ExpertNotTrainedError:
                print(
                    "ExpertNotTrainedError!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                )
                setattr(
                    target_layer.self_attn,
                    n,
                    getattr(pretrained_layer.self_attn, n),
                )

        # self._upscale_submodules(model, finetuned_models, orig_v)
        return model

    def _upscale_linear_layer(
        self,
        pretrained_model,
        finetuned_models,
        orig_v,
        name: str,
    ):
        """
        通过将其与微调模型中的相应层合并来升级线性层。

        Args:
            pretrained_model (nn.Module): 预训练模型。
            finetuned_models (List[nn.Module]): 微调模型列表。
            name (str): 要升级的线性层的名称。
        """
        config = self.config

        name_list = name.split(".")
        module = get_attr(pretrained_model, name_list)
        # module_orig = get_attr(pretrained_model_orig, name_list)
        experts = [get_attr(m, name_list) for m in finetuned_models]
        try:
            moe_linear = S2MoELinear(
                module,
                experts,
                top_k=config.top_k,
                threshold=config.threshold,  # 新增参数
                routing_use_diff=self.routing_use_diff,
                full_matrices=self.full_matrices,
                orig_v=orig_v,
                upscaling_accelerator=self.upscaling_accelerator,
            )
        except ExpertNotTrainedError:
            print(f"skip {name} because the experts are not trained.")
            return
        set_attr(pretrained_model, name_list, moe_linear)
        # 从微调模型中移除原始模块以节省内存
        for m in finetuned_models:
            set_attr(m, name_list, None)

    def _average_experts(self, pretarined_model, finetuned_models, name: str):
        """
        Average the experts for a given layer.

        Args:
            pretarined_model (nn.Module): The pretrained model.
            finetuned_models (List[nn.Module]): A list of fine-tuned models.
            name (str): The name of the layer to average.
        """
        name_list = name.split(".")
        experts = [get_attr(m, name_list) for m in finetuned_models]
        averaged_module = simple_average(experts)
        set_attr(pretarined_model, name_list, averaged_module)

    def _upscale_submodules(
        self,
        pretrained_model: nn.Module,
        finetuned_models: List[nn.Module],
        orig_v,
        tqdm_desc: str = "Upscaling Linear Modules",
    ):
        """
        Upscales the submodules of the pretrained model by merging them with the corresponding submodules from the fine-tuned models.

        Args:
            pretrained_model (nn.Module): The pretrained model.
            finetuned_models (List[nn.Module]): A list of fine-tuned models.
            tqdm_desc (str): Description for the tqdm progress bar.
        """
        config = self.config
        i = 0
        for name, module in tqdm(
            list(tuple(pretrained_model.named_modules()))[1:-1],
            tqdm_desc,
            leave=False,
            dynamic_ncols=True,
        ):
            if isinstance(module, self._linear_layer_cls):
                self._upscale_linear_layer(
                    pretrained_model=pretrained_model,
                    orig_v=orig_v[i],
                    finetuned_models=finetuned_models,
                    name=name,
                )
                i += 1
            elif config.average_experts and len(tuple(module.named_modules())) == 1:
                # if the module is a leaf module, we perform a parameter average
                self._average_experts(pretrained_model, finetuned_models, name)

    def tsv_m(self, pretrained_model, finetuned_models):
        """
        使用任务奇异向量合并(Task Singular Vector Merge)方法创建合并模型

        Args:
            pretrained_model: 预训练模型
            finetuned_models: 微调模型列表

        Returns:
            合并后的模型(在原预训练模型基础上修改)
        """

        ft_model_length = len(finetuned_models)
        sv_reduction = 1.0 / ft_model_length  # 根据模型数量确定压缩比例

        # 预先获取所有线性层模块，避免重复遍历
        linear_modules = [
            (name, module)
            for name, module in list(pretrained_model.named_modules())[1:-1]
            if isinstance(module, self._linear_layer_cls)
        ]

        # 如果需要平均专家，处理非线性层
        non_linear_modules = [
            (name, module)
            for name, module in list(pretrained_model.named_modules())
            if not isinstance(module, self._linear_layer_cls)
            and len(tuple(module.named_modules())) == 1
        ]
        for name, _ in tqdm(non_linear_modules, desc="处理非线性层"):
            if name == "model.embed_tokens":
                continue
            self._average_experts(pretrained_model, finetuned_models, name)

        # 使用tqdm显示进度
        orig_v = []
        for name, module in tqdm(linear_modules, desc="使用TSV合并线性层"):
            name_list = name.split(".")
            pretrained_module = get_attr(pretrained_model, name_list)
            expert_modules = [get_attr(m, name_list) for m in finetuned_models]

            # 计算权重差异
            weight_diffs = [
                expert.weight - pretrained_module.weight for expert in expert_modules
            ]

            # 检查是否所有差异都为零
            if _is_all_zeros(weight_diffs):
                continue

            with torch.no_grad():
                # 使用compute_svd_and_compress函数处理每个差异矩阵
                svd_results = []
                svd_orig = []
                for i, diff in enumerate(weight_diffs):
                    # 将差异矩阵移至指定设备以加速计算
                    device = self.upscaling_accelerator or diff.device
                    diff = diff.to(device)
                    # 使用TSVC_utils中的函数计算SVD并压缩
                    _, u, s, v, U, S, V = TSVC_utils.compute_svd_and_compress(
                        None, diff, sv_reduction
                    )
                    # 将结果存储在列表中
                    svd_results.append((u, s, v))
                    svd_orig.append((U, S, V))

                # 参考TSVM_utils中的方法合并SVD结果

                # svd_results -> num_experts x 3 x rank_of_router x in_features
                all_u = [result[0] for result in svd_results]
                all_s = [result[1] for result in svd_results]
                all_v = [result[2] for result in svd_results]

                # orig_v -> num of layers x num_experts x 3 x rank_of_router x in_features?????????
                orig_v.append(torch.stack(all_v, dim=0))
                # svd_results = [all_u,all_s,all_v]
                # 创建一个与第一个专家的U矩阵形状相同的全零张量
                concat_u = torch.zeros_like(U, device=all_u[0].device)

                reduced_index_u = int(svd_orig[0][0].shape[1] * sv_reduction)

                # 将每个专家的U矩阵放在特定的索引位置上
                for i, u_tensor in enumerate(all_u):
                    concat_u[:, i * reduced_index_u : (i + 1) * reduced_index_u] = (
                        u_tensor[:, :reduced_index_u]
                    )

                # 修改：使用TSVM_utils.py中的方式构建S矩阵
                # 创建一个与第一个专家的S矩阵形状相同的全零张量

                concat_s = torch.zeros_like(S, device=all_s[0].device)
                reduced_index_s = int(svd_orig[0][1].shape[0] * sv_reduction)

                # 将每个专家的S矩阵放在特定的索引位置上
                for i, s_tensor in enumerate(all_s):
                    concat_s[i * reduced_index_s : (i + 1) * reduced_index_s] = (
                        s_tensor[:reduced_index_s]
                    )

                # 修改：使用TSVM_utils.py中的方式构建V矩阵
                # 创建一个与第一个专家的V矩阵形状相同的全零张量
                concat_v = torch.zeros_like(V, device=all_v[0].device)
                reduced_index_v = int(svd_orig[0][2].shape[1] * sv_reduction)
                # 将每个专家的V矩阵放在特定的索引位置上
                for i, v_tensor in enumerate(all_v):
                    concat_v[:, i * reduced_index_v : (i + 1) * reduced_index_v] = (
                        v_tensor[:, :reduced_index_v]
                    )

                # 使用TSVM_utils中的方法重新计算SVD以获得更好的合并效果
                u_u, s_u, v_u = torch.linalg.svd(concat_u, full_matrices=False)
                u_v, s_v, v_v = torch.linalg.svd(concat_v.T, full_matrices=False)
                # 使用multi_dot构建最终权重
                reconstructed_weight = torch.linalg.multi_dot(
                    (
                        u_u,
                        v_u,
                        torch.diag(concat_s),
                        u_v,
                        v_v,
                    )
                )

                # 更新预训练模型权重
                reconstructed_weight = reconstructed_weight.to(
                    pretrained_module.weight.device
                )
                pretrained_module.weight.data.add_(reconstructed_weight)

                # 打印日志信息
                # if logging.getLogger().level <= logging.DEBUG:
                #     log.debug(f"更新模块 {name} 的权重，权重差异范数: {torch.norm(reconstructed_weight)}")
                # else:
                #     log.info(f"已更新模块 {name}")

        # 如果需要平均专家，处理非线性层
        if self.average_experts:
            non_linear_modules = [
                (name, module)
                for name, module in pretrained_model.named_modules()
                if not isinstance(module, self._linear_layer_cls)
                and len(tuple(module.named_modules())) == 1
            ]

            for name, _ in tqdm(non_linear_modules, desc="处理非线性层"):
                self._average_experts(pretrained_model, finetuned_models, name)
        return pretrained_model, orig_v
