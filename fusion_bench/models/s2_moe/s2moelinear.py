import torch
from torch import nn, Tensor
from typing import List
from fusion_bench.models.smile_moe.utils import _is_all_zeros, svd
from fusion_bench.models.s2_moe.sparse_linear import SparseLinear
import torch.nn.functional as F


class ExpertNotTrainedError(Exception):
    pass


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
            svd_cache_list = []
            for w in w_diff_list:
                # 保存原始数据类型
                original_dtype = w.dtype

                # 如果是BFloat16类型，转换为float32
                if w.dtype == torch.bfloat16:
                    w = w.to(torch.float32)

                # 执行SVD操作
                u, s, v = svd(
                    w, full_matrices=full_matrices, accelerator=upscaling_accelerator
                )

                # 如果原始数据是BFloat16，将结果转换回BFloat16
                if original_dtype == torch.bfloat16:
                    u = u.to(torch.bfloat16)
                    s = s.to(torch.bfloat16)
                    v = v.to(torch.bfloat16)

                svd_cache_list.append((u, s, v))
            # SVD缓存列表，避免重复计算
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
            experts = [
                #! use SparseLinear instead of SmileCompressedLinear
                SparseLinear(
                    m.in_features,
                    m.out_features,
                    bias=m.bias is not None,
                    sparsity_ratio=0.8,
                )
                for m in finetuned_models
            ]
            for m, expert in zip(finetuned_models, experts):
                expert.set_parameters(m)
                expert.apply_pruning_()
        else:
            # 如果k未设置（<0），我们使用完整的微调模型
            experts = finetuned_models
        self.experts = nn.ModuleList(experts)

        if pretrained_model.bias is not None:
            for m in experts:
                m.bias.data = m.bias.data - pretrained_model.bias
        # 分配预训练模型（共享部分）
        self.shared_linear = pretrained_model

    def forward(self, hidden_states: Tensor):
        """
        SmileMoELinear模块的前向传播。

        Args:
            hidden_states (Tensor): 输入张量。

        Returns:
            Tensor: 输出张量。
        """
        pretrained_out = self.shared_linear(hidden_states)
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
        return self.shared_linear.weight

    @property
    def bias(self):
        return self.shared_linear.bias

    def __repr__(self):
        return (
            f"SingularMoELinear("
            f"in_features={self.shared_linear.in_features}, "
            f"out_features={self.shared_linear.out_features}, "
            f"num_experts={self.num_experts}, "
            f"top_k={self.top_k}, "
            f"gate_k={self.gate_k}, "
            f"k={self.k}"
            f")"
        )


class ProjectionBasedGate(nn.Module):
    def __init__(
        self,
        input_features: int,
        w_diff_list: List[Tensor],
        orig_v: List[Tensor],
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

        # TODO: use torch.stack to convet the parameter list into a single parameter
        self.orig_v = nn.ParameterList(
            [nn.Parameter(v, requires_grad=False) for v in orig_v]
        )
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
        # print("top_values: ", top_values)
        # print("top_indices: ", top_indices)
        # print("#######################################################")
        # 将未选中的权重置为0，并重新归一化
        filtered_weights = routing_weights * mask.float()
        sum_weights = filtered_weights.sum(dim=1, keepdim=True)
        sum_weights = torch.where(
            sum_weights == 0, torch.ones_like(sum_weights), sum_weights
        )
        normalized_weights = filtered_weights / sum_weights

        return normalized_weights
