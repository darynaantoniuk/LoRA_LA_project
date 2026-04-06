import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    """
    Replaces nn.Linear with a LoRA-augmented version.
    Forward: y = x @ W.T + (x @ A.T) @ B.T * scale
    """
    def __init__(self, in_features, out_features, r=8, lora_alpha=16, bias=True):
        super().__init__()
        self.r = r
        self.scale = lora_alpha / r

        self.weight = nn.Parameter(
            torch.empty(out_features, in_features), requires_grad=False
        )
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))  # B=0 at init

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        base_out = x @ self.weight.T
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T * self.scale
        out = base_out + lora_out
        return out + self.bias if self.bias is not None else out

    def merge_weights(self):
        """Merge LoRA into base weight for zero-latency inference."""
        self.weight.data += self.scale * (self.lora_B @ self.lora_A)


def inject_lora(model, r=8, lora_alpha=16, target_modules=("query", "value")):
    """
    Replace target Linear layers in a RoBERTa model with LoRALinear.
    Copies pretrained weights into the frozen weight parameter.
    """
    for name, module in model.named_modules():
        for target in target_modules:
            if target in name and isinstance(module, nn.Linear):
                parent_name, child_name = name.rsplit(".", 1)
                parent = dict(model.named_modules())[parent_name]

                lora_layer = LoRALinear(
                    module.in_features, module.out_features,
                    r=r, lora_alpha=lora_alpha,
                    bias=module.bias is not None
                )

                lora_layer.weight.data = module.weight.data.clone()
                if module.bias is not None:
                    lora_layer.bias.data = module.bias.data.clone()

                setattr(parent, child_name, lora_layer)
    return model


def count_trainable_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable: {trainable:,} / Total: {total:,} ({100*trainable/total:.2f}%)")
    return trainable, total
