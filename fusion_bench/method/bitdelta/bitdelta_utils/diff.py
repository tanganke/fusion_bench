import gc

import torch
from torch import Tensor, nn

from .binary_gemm_kernel import binary_bmm, pack, unpack


class BinaryDiff(nn.Module):
    def __init__(self, base: Tensor, finetune: Tensor):
        super().__init__()
        diff = finetune - base
        quantile = diff.float().abs().mean()

        mask = torch.ones_like(diff)
        mask[diff < 0] = 0
        mask = pack(mask.bool().T)

        self.register_buffer("mask", mask)
        self.register_buffer("base", base.T)
        self.register_parameter(
            "coeff",
            nn.Parameter(
                torch.tensor(
                    quantile,
                    dtype=torch.float32,
                    requires_grad=True,
                    device=base.device,
                )
            ),
        )
        del base, finetune, diff

    def forward(self, x):
        # print(x.shape, self.base.shape, self.coeff.shape, self.mask.shape)
        # [B, seq, in] @ [in, out] + [B, seq, in] @ [B, in/32, out]

        # TODO: This can be faster
        repeated_mask = self.mask.unsqueeze(0).repeat(x.size(0), 1, 1)
        return x @ self.base + self.coeff * binary_bmm(x, repeated_mask)


def compress_diff(base_model, finetuned_model, finetuned_compressed_model):
    def compress_submodule(name, subname, module, submodule):
        target_device = submodule.weight.device

        base_weight = (
            base_model.get_submodule(f"{name}.{subname}")
            .weight.detach()
            .to(target_device)
        )
        finetuned_weight = (
            finetuned_model.get_submodule(f"{name}.{subname}")
            .weight.detach()
            .to(target_device)
        )

        compressed = BinaryDiff(
            base=base_weight,
            finetune=finetuned_weight,
        ).to(target_device)

        del submodule, base_weight
        setattr(module, subname, None)
        gc.collect()
        torch.cuda.empty_cache()
        setattr(module, subname, compressed)

    # TODO: this can be parallelized
    for name, module in finetuned_compressed_model.named_modules():
        if "mlp" in name or "self_attn" in name:
            for subname, submodule in module.named_children():
                if "proj" in subname:
                    compress_submodule(name, subname, module, submodule)


def save_diff(finetuned_compressed_model, save_dir):
    diff_dict = {}

    for name, module in finetuned_compressed_model.named_modules():
        if isinstance(module, BinaryDiff):
            # diff_dict[name + ".mask"] = (module.mask == 1).bool().cpu()
            diff_dict[name + ".mask"] = module.mask.cpu()
            diff_dict[name + ".coeff"] = module.coeff.cpu()

    for name, param in finetuned_compressed_model.named_parameters():
        if param.requires_grad:
            diff_dict[name] = param.cpu()

    torch.save(diff_dict, save_dir)


@torch.no_grad()
def load_diff(model, diff_dir):
    device = model.device
    diff_dict = torch.load(diff_dir)

    for name, module in model.named_modules():
        if name + ".mask" in diff_dict:
            coeff = diff_dict[name + ".coeff"].to(device)
            mask = diff_dict[name + ".mask"].to(device)

            # setattr(module, "mask", mask)
            # setattr(module, "coeff", coeff)
            weight = (unpack(mask) * 2 - 1) * coeff

            module.weight.add_(weight.T.to(module.weight.dtype))
        elif name + ".weight" in diff_dict:
            module.weight = nn.Parameter(
                diff_dict[name + ".weight"].to(device).to(module.weight.dtype)
            )

        elif name + ".A" in diff_dict:
            A = diff_dict[name + ".A"].to(device)
            B = diff_dict[name + ".B"].to(device)

            mask = (A @ B).T
            module.weight.add_(mask.to(module.weight.dtype))

    model.config.vocab_size = model.lm_head.weight.size(0)


def save_full_model(base_model, tokenizer, diff_dir, save_dir):
    load_diff(base_model, diff_dir)

    base_model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    del base_model
