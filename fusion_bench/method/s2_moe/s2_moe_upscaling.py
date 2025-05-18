import copy
import logging
import os
import time
from copy import deepcopy
from re import U
from typing import Dict, List, Tuple  # noqa: F401

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import Tensor, nn
from tqdm.auto import tqdm

from fusion_bench.method import BaseAlgorithm
from fusion_bench.method.s2_moe.utils import TSVC_utils, TSVM_utils
from fusion_bench.method.simple_average import simple_average
from fusion_bench.mixins.simple_profiler import SimpleProfilerMixin
from fusion_bench.modelpool import BaseModelPool
from fusion_bench.models.s2_moe.sparse_linear import SparseLinear
from fusion_bench.models.smile_moe.linear_from_module import ExpertNotTrainedError
from fusion_bench.models.smile_moe.utils import _is_all_zeros, svd
from fusion_bench.models.utils import get_attr, set_attr
from fusion_bench.utils.parameters import print_parameters

log = logging.getLogger(__name__)


class S2MoELinear(nn.Module):
    @torch.no_grad()
    def __init__(
        self,
        pretrained_model: nn.Linear,
        finetuned_models: List[nn.Linear],
        gate_k: int,
        k: int,
        top_k: int = 2,
        threshold: float = 0.1,
        full_matrices=True,
        orig_v=None,
        upscaling_accelerator=None,
        routing_use_diff=True,
    ):
        """
        初始化SmileMoELinear模块。

        Args:
            pretrained_model (nn.Linear): 预训练线性模型。
            finetuned_models (List[nn.Linear]): 微调线性模型列表。
            gate_k (int): 门控网络保留的奇异值数量。
            k (int): 专家保留的奇异值数量。
            top_k (int): 选择的顶部专家数量。
            threshold (float): 激活任务子空间的阈值。
            full_matrices (bool): 是否计算完整大小的U和V矩阵。
            upscaling_accelerator (str): 用于计算的设备。
            routing_use_diff (bool): 是否使用权重差异进行路由。
        """
        super().__init__()
        self.num_experts = len(finetuned_models)
        self.top_k = min(top_k, self.num_experts)
        self.k = k
        self.gate_k = gate_k
        self.in_features = pretrained_model.in_features
        self.out_features = pretrained_model.out_features
        self.threshold = threshold

        w_diff_list = [
            m.weight - pretrained_model.weight
            for m in finetuned_models
            if isinstance(m, (nn.Linear,))
        ]
        # w_diff_orig = [m.weight - pretrained_model_orig.weight for m in finetuned_models if isinstance(m, (nn.Linear,))]
        if _is_all_zeros(w_diff_list):
            # 所有微调模型与预训练模型相同
            raise ExpertNotTrainedError()

        if routing_use_diff or k > 0:
            svd_cache_list = [
                svd(w, full_matrices=full_matrices, accelerator=upscaling_accelerator)
                for w in w_diff_list
            ]  # SVD缓存列表，避免重复计算
        self.orig_v = orig_v
        # 构建门控网络
        if routing_use_diff:
            self.gate = ProjectionBasedGate(
                input_features=self.in_features,
                w_diff_list=w_diff_list,
                orig_v=orig_v,
                k=gate_k,
                threshold=threshold,
                top_k=top_k,
                upscaling_accelerator=upscaling_accelerator,
            )
        else:
            self.gate = ProjectionBasedGate(
                input_features=self.in_features,
                w_diff_list=[m.weight for m in finetuned_models],
                k=gate_k,
                threshold=threshold,
                top_k=top_k,
                upscaling_accelerator=upscaling_accelerator,
            )

        # 构建专家
        for m, w_diff in zip(finetuned_models, w_diff_list):
            m.weight.data = w_diff
        if k > 0:
            # TODO: 改成用 config 初始化 SparseLinear 或者 SparseLinear 加一个 static method `SparseLinear.from_module(m: nn.Linear, sparsity_ratio: float) -> SparseLinear` 来初始化
            if False:  # Example
                experts = []
                for m in finetuned_models:
                    experts.append(
                        SparseLinear(
                            in_features=m.in_features,
                            out_features=m.out_features,
                            sparsity_ratio=0.5,
                            bias=True if m.bias is not None else False,
                        )
                    )
                    experts[-1].weight.data = m.weight.data
                    if m.bias is not None:
                        experts[-1].bias.data = m.bias.data
                    experts[-1].apply_pruning_()
            experts = [SparseLinear(m, sparsity_ratio=0.5) for m in finetuned_models]
        else:
            # 如果k未设置（<0），我们使用完整的微调模型
            experts = finetuned_models
        self.experts = nn.ModuleList(experts)

        if pretrained_model.bias is not None:
            for m in experts:
                m.bias.data = m.bias.data - pretrained_model.bias
        # 分配预训练模型（共享部分）
        self.pretrained_model = pretrained_model

    def forward(self, hidden_states: Tensor):
        """
        SmileMoELinear模块的前向传播。

        Args:
            hidden_states (Tensor): 输入张量。

        Returns:
            Tensor: 输出张量。
        """
        pretrained_out = self.pretrained_model(hidden_states)
        input_shape = hidden_states.size()
        hidden_states = hidden_states.view(-1, self.in_features)

        # 使用基于投影的路由方法
        routing_weights = self.gate(hidden_states, pretrained_out)
        # 获取非零权重的索引（已经在gate中应用了阈值和top-k限制）
        non_zero_mask = routing_weights > 0
        final_hidden_states = torch.zeros(
            (hidden_states.size(0), self.out_features),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        # 批量处理专家计算，避免逐样本循环
        for expert_idx, expert in enumerate(self.experts):
            # 找出使用该专家的所有样本
            batch_indices = torch.nonzero(non_zero_mask[:, expert_idx]).squeeze()
            if batch_indices.numel() == 0:
                # 如果没有样本使用该专家，跳过
                continue

            # 确保 batch_indices 是一维向量
            if batch_indices.dim() == 0:
                batch_indices = batch_indices.unsqueeze(0)

            # 添加索引验证，确保索引在有效范围内
            valid_indices = (batch_indices >= 0) & (
                batch_indices < hidden_states.size(0)
            )
            if not torch.all(valid_indices):
                # 过滤无效索引
                batch_indices = batch_indices[valid_indices]
                if batch_indices.numel() == 0:
                    continue

            # 批量计算该专家的输出
            expert_inputs = hidden_states[batch_indices]
            expert_weights = routing_weights[batch_indices, expert_idx].unsqueeze(1)
            expert_outputs = expert(expert_inputs) * expert_weights

            # 将结果添加到最终输出
            final_hidden_states.index_add_(0, batch_indices, expert_outputs)

        final_hidden_states = final_hidden_states.reshape(
            *input_shape[:-1], self.out_features
        )
        final_hidden_states = pretrained_out + final_hidden_states

        return final_hidden_states

    @property
    def weight(self):
        """
        Mimic linear layer. Bacause in some cases, user might indicate the device (or dtype of parameters) of the linear layer using `linear_layer.weight.device`
        """
        return self.pretrained_model.weight

    @property
    def bias(self):
        return self.pretrained_model.bias

    def __repr__(self):
        return (
            f"SingularMoELinear("
            f"in_features={self.pretrained_model.in_features}, "
            f"out_features={self.pretrained_model.out_features}, "
            f"num_experts={self.num_experts}, "
            f"top_k={self.top_k}, "
            f"gate_k={self.gate_k}, "
            f"k={self.k}"
            f")"
        )


class S2MoEUpscalingAlgorithm(
    SimpleProfilerMixin,
    BaseAlgorithm,
):
    _linear_layer_cls = (nn.Linear,)
    _config_mapping = BaseAlgorithm._config_mapping | {
        "device": "device",
        "upscaling_accelerator": "upscaling_accelerator",
        "full_matrices": "full_matrices",
        "gate_k": "gate_k",
        "k": "k",
        "top_k": "top_k",
        "routing_use_diff": "routing_use_diff",
        "average_experts": "average_experts",
        "model_path": "model_path",
        "threshold": "threshold",  # 新增参数
    }

    def __init__(
        self,
        *,
        device: str = "cuda",
        upscaling_accelerator: str = None,
        full_matrices: bool = True,
        gate_k: int = 256,
        k: int = 256,
        top_k: int = 2,
        threshold: float = 0.1,  # 新增参数
        routing_use_diff: bool = True,
        average_experts: bool = False,
        model_path: str = None,
        **kwargs,
    ):
        """
        初始化SmileUpscalingAlgorithm。

        Args:
            device (str): 用于计算的设备。
            upscaling_accelerator (str): 用于SVD计算的设备。
            full_matrices (bool): 是否计算完整大小的U和V矩阵。
            gate_k (int): 门控网络保留的奇异值数量。
            k (int): 专家保留的奇异值数量。
            top_k (int): 选择的顶部专家数量。
            threshold (float): 激活任务子空间的阈值。
            routing_use_diff (bool): 是否使用权重差异进行路由。
            average_experts (bool): 是否平均专家。
            model_path (str): 保存/加载模型的路径。
            **kwargs: 额外参数。
        """
        super().__init__()
        self.device = device
        self.upscaling_accelerator = upscaling_accelerator
        self.full_matrices = full_matrices
        self.gate_k = gate_k
        self.k = k
        self.top_k = top_k
        self.threshold = threshold  # 新增参数
        self.routing_use_diff = routing_use_diff
        self.average_experts = average_experts
        self.model_path = model_path
        self.rout_svd_cache = None
        for key, value in kwargs.items():
            log.warning(f"Unrecognized argument: {key}")
            setattr(self, key, value)

        # 打印配置
        print(f"=== Config for `{type(self).__name__}` ===")
        print(OmegaConf.to_yaml(self.config))
        print(f"=== Config for `{type(self).__name__}` ===")

    @torch.no_grad()
    def run(self, modelpool: BaseModelPool):
        """
        Executes the upscaling process.

        Args:
            modelpool (ModelPool): The pool of models to be used for upscaling.

        Returns:
            nn.Module: The upscaled model.
        """
        if not isinstance(modelpool, BaseModelPool):
            modelpool = BaseModelPool(modelpool)

        if self.config.model_path is not None and os.path.exists(
            self.config.model_path
        ):
            log.info(f"Loading model from {self.config.model_path}")
            model = torch.load(self.config.model_path)
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
        pretrained_model, orig_v = self.tsv_m(pretrained_model, finetuned_models)

        with self.profile("merge model"):
            model = self.merge(pretrained_model, finetuned_models, orig_v)

        self.print_profile_summary()
        if self.config.model_path is not None:
            os.makedirs(os.path.dirname(self.config.model_path), exist_ok=True)
            log.info(f"Saving model to {self.config.model_path}")
            torch.save(model, self.config.model_path)
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

        self._upscale_submodules(model, finetuned_models, orig_v)
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
                gate_k=config.gate_k,
                k=config.k,
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

    def _average_experts(
        self,
        pretarined_model: nn.Module,
        finetuned_models: List[nn.Module],
        name: str,
    ):
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
            tuple(pretrained_model.named_modules()),
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

    def tsv_m(self, pretrained_model: nn.Module, finetuned_models: List[nn.Module]):
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
            for name, module in pretrained_model.named_modules()
            if isinstance(module, self._linear_layer_cls)
        ]

        # 如果需要平均专家，处理非线性层
        non_linear_modules = [
            (name, module)
            for name, module in pretrained_model.named_modules()
            if not isinstance(module, self._linear_layer_cls)
            and len(tuple(module.named_modules())) == 1
        ]
        for name, _ in tqdm(non_linear_modules, desc="处理非线性层"):
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
                all_u = [result[0] for result in svd_results]
                all_s = [result[1] for result in svd_results]
                all_v = [result[2] for result in svd_results]
                # orig_v.append(all_u)
                orig_v.append(all_v)
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


class ProjectionBasedGate(nn.Module):
    def __init__(
        self,
        input_features: int,
        w_diff_list: List[Tensor],
        orig_v,
        k: int,
        threshold: float = 0.1,
        top_k: int = 2,
        upscaling_accelerator=None,
    ):
        """
        基于投影的路由门控模块。

        Args:
            input_features (int): 输入特征的维度。
            w_diff_list (List[Tensor]): 权重差异张量列表。
            k (int): 保留的奇异值数量。
            threshold (float): 激活任务子空间的阈值。
            top_k (int): 最多选择的任务子空间数量。
            svd_list: 缓存的SVD结果。
            upscaling_accelerator: 用于计算的设备。
        """
        super().__init__()
        self.input_features = input_features
        self.num_experts = len(w_diff_list)
        # self.threshold = threshold
        self.threshold = 1 / self.num_experts
        self.top_k = min(top_k, self.num_experts)
        # # 构建任务子空间
        # self.task_subspaces_V = nn.ParameterList()
        # self.task_subspaces_U = nn.ParameterList()
        # self.task_subspaces_S = nn.ParameterList()

        self.orig_v = orig_v
        # for i, w_diff in enumerate(w_diff_orig):
        #     u, s, v = svd(w_diff.T, accelerator=upscaling_accelerator)
        #     split_k = int(1/self.num_experts * s.shape[0])

        # for i, w_diff in enumerate(w_diff_list):
        #     u, s, v = svd(w_diff, accelerator=upscaling_accelerator)

        #     # 截断到秩k
        #     #v_truncated = v[:, :k]
        #     # 存储右奇异向量作为子空间的基
        #     self.task_subspaces_U.append(nn.Parameter(u, requires_grad=False))
        #     self.task_subspaces_S.append(nn.Parameter(s, requires_grad=False))
        #     self.task_subspaces_V.append(nn.Parameter(v, requires_grad=False))

    def forward(self, x: Tensor, x_l: Tensor):
        """
        前向传播，计算路由权重。

        Args:
            x (Tensor): 输入张量。

        Returns:
            Tensor: 路由权重。
        """
        batch_size = x.size(0)
        if self.num_experts == 1:
            return torch.ones(batch_size, 1, device=x.device, dtype=x.dtype)

        # 计算每个任务子空间的投影残差
        residuals = []

        for i in range(self.num_experts):
            # U = self.task_subspaces_U[i]
            # S = self.task_subspaces_S[i]
            # V = self.task_subspaces_V[i]

            v_0 = self.orig_v[i]

            # 判断x_l的维度为3则进行视图变换
            if len(x.shape) == 3:
                # 将三维张量重塑为二维张量，合并前两个维度
                x = x.view(x.shape[0] * x.shape[1], -1)

            projection = torch.matmul(
                v_0, torch.matmul(v_0.T, x.T)
            ).T  # 768 96 96 768 768 6400  ——》 6400 768
            # 计算残差: r = ||x - proj_V(x)||_2
            # projection = torch.nn.functional.normalize(projection, p=2, dim=-1)
            # x_l = torch.nn.functional.normalize(x_l, p=2, dim=-1)
            residual = torch.norm(x - projection, p=2, dim=-1)
            residuals.append(residual)
        # 将残差堆叠为形状 [batch_size, num_experts] 的张量
        residuals = torch.stack(residuals, dim=1)

        # 计算残差的加性逆（取负数并加上最大值，确保非负）
        # inverse_residuals = -residuals + torch.max(residuals, dim=1, keepdim=True)[0]

        # 通过softmax归一化得到路由权重
        routing_weights = F.softmax(-residuals, dim=1)

        # 应用阈值过滤
        mask = routing_weights > self.threshold

        # 如果没有超过阈值的权重，选择最大的一个
        if not torch.any(mask):
            top_values, top_indices = torch.topk(routing_weights, 1, dim=1)
            mask = torch.zeros_like(routing_weights, dtype=torch.bool)
            mask.scatter_(1, top_indices, True)
        # 限制选择前top_k个
        if self.top_k < self.num_experts:
            top_values, top_indices = torch.topk(routing_weights, self.top_k, dim=1)
            top_k_mask = torch.zeros_like(routing_weights, dtype=torch.bool)
            top_k_mask.scatter_(1, top_indices, True)
            mask = mask & top_k_mask

        # 将未选中的权重置为0，并重新归一化
        filtered_weights = routing_weights * mask.float()
        sum_weights = filtered_weights.sum(dim=1, keepdim=True)
        sum_weights = torch.where(
            sum_weights == 0, torch.ones_like(sum_weights), sum_weights
        )
        normalized_weights = filtered_weights / sum_weights

        torch.set_printoptions(threshold=np.inf)
        # print("routing weights",routing_weights)
        # print("mask : ", mask)
        # print("self.top_k: ", self.top_k)
        # import sys
        # sys.exit()

        return normalized_weights
