import math
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """
    LoRA wrapper for a frozen linear layer.
    Forward: y = xW^T + scale * xA^T B^T
    """
    def __init__(self, linear_layer: nn.Linear, r: int = 8, lora_alpha: int = 16):
        super().__init__()
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.r = r
        self.scale = lora_alpha / r if r > 0 else 1.0

        self.weight = nn.Parameter(linear_layer.weight.data.clone(), requires_grad=False)

        if linear_layer.bias is not None:
            self.bias = nn.Parameter(linear_layer.bias.data.clone(), requires_grad=False)
        else:
            self.bias = None

        self.lora_A = nn.Parameter(torch.empty(r, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))

        nn.init.normal_(self.lora_A, mean=0.0, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = x @ self.weight.T
        lora = (x @ self.lora_A.T) @ self.lora_B.T
        out = base + self.scale * lora
        if self.bias is not None:
            out = out + self.bias
        return out

    def merge(self):
        delta_w = self.scale * (self.lora_B @ self.lora_A)
        self.weight.data += delta_w.data


def _get_parent_module(model: nn.Module, module_name: str):
    parts = module_name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def inject_lora(model: nn.Module, r: int = 8, lora_alpha: int = 16,
                target_keywords=("query", "value")) -> nn.Module:
    """
    Replace attention query/value linear layers with LoRALinear.
    Matches the report plan: adapt Q and V projections only.
    """
    replace_names = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(k in name for k in target_keywords):
            replace_names.append(name)

    for name in replace_names:
        parent, child_name = _get_parent_module(model, name)
        old_layer = getattr(parent, child_name)
        setattr(parent, child_name, LoRALinear(old_layer, r=r, lora_alpha=lora_alpha))

    return model


def freeze_non_lora_params(model: nn.Module):
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


def merge_lora_weights(model: nn.Module):
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge()


def count_parameters(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "trainable_percent": 100.0 * trainable / total if total > 0 else 0.0,
    }