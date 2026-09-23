"""Four independent residual projections and patch-token-only injection."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch
from torch import nn

INJECTION_BLOCKS = (6, 12, 18, 24)


class GatedProjection(nn.Module):
    """256 -> 487 -> 1024 with an independent, zero-initialized scalar gate."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(256, 487),
            nn.GELU(),
            nn.Linear(487, 1024),
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, side_tokens: torch.Tensor) -> torch.Tensor:
        return self.gamma * self.net(side_tokens)


class ResidualInjector(nn.Module):
    """Inject side tokens after DINOv3 blocks 6, 12, 18, and 24.

    Call ``register(backbone)`` once and wrap each backbone forward pass in
    ``use(side_tokens)``. The first token and register tokens are untouched.
    """

    def __init__(self) -> None:
        super().__init__()
        self.adapters = nn.ModuleDict(
            {str(block): GatedProjection() for block in INJECTION_BLOCKS}
        )
        self._side_tokens: torch.Tensor | None = None
        self._prefix_tokens: int | None = None
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def register(self, backbone: nn.Module) -> None:
        if self._handles:
            raise RuntimeError("ResidualInjector is already registered")
        if len(backbone.blocks) < max(INJECTION_BLOCKS):
            raise ValueError("Backbone does not have 24 Transformer blocks")
        n_register = getattr(backbone, "num_register_tokens", None)
        if n_register is None:
            n_register = getattr(backbone, "n_storage_tokens", 4)
        self._prefix_tokens = 1 + int(n_register)
        for block in INJECTION_BLOCKS:
            self._handles.append(
                backbone.blocks[block - 1].register_forward_hook(self._make_hook(block))
            )

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._side_tokens = None
        self._prefix_tokens = None

    @contextmanager
    def use(self, side_tokens: torch.Tensor) -> Iterator[None]:
        if not self._handles:
            raise RuntimeError("Call register(backbone) before use(...)")
        if self._side_tokens is not None:
            raise RuntimeError("Nested use(...) is not supported")
        if side_tokens.ndim != 3 or side_tokens.shape[1:] != (1024, 256):
            raise ValueError("Expected side tokens [B,1024,256]")
        self._side_tokens = side_tokens
        try:
            yield
        finally:
            self._side_tokens = None

    def _make_hook(self, block: int):
        def hook(_module, _inputs, output):
            side = self._side_tokens
            if side is None:
                return output
            if isinstance(output, torch.Tensor):
                hidden, tail, wrapper = output, (), torch.Tensor
            elif isinstance(output, (tuple, list)) and output:
                hidden, tail, wrapper = output[0], output[1:], type(output)
            else:
                raise TypeError(f"Unexpected block output type: {type(output)}")
            if hidden.ndim != 3 or self._prefix_tokens is None:
                raise ValueError("Expected token sequence [B,N,1024]")
            prefix = self._prefix_tokens
            if hidden.shape[0] != side.shape[0] or hidden.shape[1] - prefix != side.shape[1]:
                raise ValueError("Side and patch token grids do not match")
            with torch.amp.autocast("cuda", enabled=False):
                delta = self.adapters[str(block)](side.float())
            patches = hidden[:, prefix:, :]
            if delta.shape != patches.shape:
                raise ValueError("Projected residual shape does not match patch tokens")
            updated = torch.cat((hidden[:, :prefix, :].clone(), patches + delta), dim=1)
            if wrapper is tuple:
                return (updated, *tail)
            if wrapper is list:
                return [updated, *tail]
            return updated

        return hook
