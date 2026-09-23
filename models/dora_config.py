"""Attach DoRA to the fused QKV projection of every DINOv3 block."""

from __future__ import annotations

from torch import nn
from peft import LoraConfig, inject_adapter_in_model

DORA_RANK = 128
DORA_ALPHA = 256


def fused_qkv_dora_config() -> LoraConfig:
    """Paper setting: rank 128, alpha 256, scale alpha/r = 2."""
    return LoraConfig(
        r=DORA_RANK,
        lora_alpha=DORA_ALPHA,
        target_modules=["qkv"],
        lora_dropout=0.0,
        bias="none",
        use_dora=True,
    )


def attach_fused_qkv_dora(backbone: nn.Module) -> nn.Module:
    """Add DoRA adapters while freezing the pretrained backbone weights."""
    qkv_count = sum(name.endswith(".attn.qkv") for name, _ in backbone.named_modules())
    if qkv_count != 24:
        raise ValueError(f"Expected one fused QKV per DINOv3 ViT-L block; found {qkv_count}")
    backbone = inject_adapter_in_model(fused_qkv_dora_config(), backbone)
    for name, parameter in backbone.named_parameters():
        parameter.requires_grad = "lora_" in name or "dora_" in name or "magnitude" in name
    return backbone
