from typing import List

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from fusion_bench.models.s2_moe.sparse_linear import SparseLinear
from fusion_bench.models.smile_moe.utils import _is_all_zeros, svd
from fusion_bench.utils.state_dict_arithmetic import state_dict_sub


class ExpertNotTrainedError(Exception):
    pass


class S2MoEConfig:
    num_experts_per_tok: int
    rank_of_router: int
    num_local_experts: int
    use_sparse_expert: bool
    sparsity_ratio: float


class ProjectionBasedGate(nn.Module):
    def __init__(
        self,
        num_local_experts: int,
        num_experts_per_tok: int,
        rank_of_router: int,
        in_features: int,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.num_local_experts = num_local_experts
        self.rank_of_router = rank_of_router
        self.in_features = in_features
        self.num_experts_per_tok = num_experts_per_tok  # top_k
        self.threshold = 1 / num_local_experts

        factory_kwargs = {"device": device, "dtype": dtype}

        
        self.weight = nn.Parameter(
            torch.empty(
                self.num_local_experts,
                self.in_features,
                self.rank_of_router,
                **factory_kwargs,
            ),
        )

    def forward(self, x: Tensor, x_l: Tensor):
        """
        前向传播，计算路由权重。

        Args:
            x (Tensor): 输入张量。

        Returns:
            Tensor: 路由权重。
        """
        batch_size = x.size(0)
        if self.num_local_experts == 1:
            return torch.ones(batch_size, 1, device=x.device, dtype=x.dtype)

        # 计算每个任务子空间的投影残差
        residuals = []

        for i in range(self.num_local_experts):


            v_0 = self.weight[i]

            # 判断x_l的维度为3则进行视图变换
            if len(x.shape) == 3:
                # 将三维张量重塑为二维张量，合并前两个维度
                x = x.view(x.shape[0] * x.shape[1], -1)

            projection = torch.matmul(
                v_0, torch.matmul(v_0.T, x.T)
            ).T  # 768 96 96 768 768 6400  ——》 6400 768
            residual = torch.norm(x - projection, p=2, dim=-1)
            residuals.append(residual)
        # 将残差堆叠为形状 [batch_size, num_experts] 的张量
        residuals = torch.stack(residuals, dim=1)

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
        if self.num_experts_per_tok < self.num_local_experts:
            top_values, top_indices = torch.topk(
                routing_weights, self.num_experts_per_tok, dim=1
            )
            top_k_mask = torch.zeros_like(routing_weights, dtype=torch.bool)
            top_k_mask.scatter_(1, top_indices, True)
            mask = mask & top_k_mask

        # 将未选中的权重置为0，并重新归一化
        filtered_weights = routing_weights * mask.to(dtype=routing_weights.dtype)
        sum_weights = filtered_weights.sum(dim=1, keepdim=True)
        sum_weights = torch.where(
            sum_weights == 0, torch.ones_like(sum_weights), sum_weights
        )
        normalized_weights = filtered_weights / sum_weights

        return normalized_weights


class S2MoELinear(nn.Module):
    def __init__(
        self,
        config: S2MoEConfig,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.num_experts_per_tok = config.num_experts_per_tok
        self.rank_of_router = min(in_features, out_features)//config.num_local_experts
        self.num_local_experts = config.num_local_experts
        self.use_sparse_expert = config.use_sparse_expert
        self.in_features = in_features
        self.out_features = out_features

        factory_kwargs = {"device": device, "dtype": dtype}

        # construct the gate network
        self.gate = ProjectionBasedGate(
            num_local_experts=self.num_local_experts,
            num_experts_per_tok=self.num_experts_per_tok,
            rank_of_router=self.rank_of_router,
            in_features=self.in_features,
            **factory_kwargs,
        )

        # construct the expert network
        self.experts = nn.ModuleList(
            [
                SparseLinear(
                    in_features=self.in_features,
                    out_features=self.out_features,
                    bias=bias,
                    sparsity_ratio=config.sparsity_ratio,
                    **factory_kwargs,
                ) for _ in range(self.num_local_experts)
            ]
        )

        self.pretrained_model = nn.Linear(
            in_features=self.in_features,
            out_features=self.out_features,
            bias=bias,
            **factory_kwargs,
        )

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
            f"num_experts={self.num_local_experts}, "
            f"top_k={self.num_experts_per_tok}, "
            f"rank_of_router={self.rank_of_router}, "
            f")"
        )


@torch.no_grad()
def upscale_to_s2moe_linear(
    base: nn.Linear, experts: List[nn.Linear], target: S2MoELinear, orig_v, use_sparse_expert, sparsity_ratio
):
    """
    Upscale a base linear layer to a SmileLinear layer using expert models.

    Args:
        base (nn.Linear): The base linear layer.
        experts (List[nn.Linear]): A list of expert linear layers.
        target (SmileLinear): The target SmileLinear layer.
        orig_v: The original v of the gate network.
        use_sparse_expert: Whether to use sparse expert.
        sparsity_ratio: The sparsity ratio of the sparse expert.
    Returns:
        SmileLinear: The upscaled SmileLinear layer.
    """
    w = base.weight
    w_ft_list = [e.weight for e in experts]
    dw_list = [w_ft - w for w_ft in w_ft_list]

    if _is_all_zeros(dw_list):
        raise ExpertNotTrainedError("Expert models are not trained")

    num_local_experts = target.num_local_experts
    gate_weight=torch.stack([v for v in orig_v], dim=0 )

    target.gate.load_state_dict({"weight": gate_weight})

    # shared linear
    target.pretrained_model.load_state_dict(base.state_dict())

    # experts
    for expert_idx, target_expert in enumerate(target.experts):
        target_expert.load_state_dict(
            state_dict_sub(experts[expert_idx].state_dict(), base.state_dict()),
            strict=False  
        )
    # if use_sparse_expert:
    #     for expert_idx, target_expert in enumerate(target.experts):
    #         target_expert.sparsity_ratio = sparsity_ratio
    #         target_expert.apply_pruning_()

    return target