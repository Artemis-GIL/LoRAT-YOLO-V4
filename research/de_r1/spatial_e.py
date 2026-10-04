"""A zero-output spatial residual around the frozen native LoRAT MLP head."""
from pathlib import Path

import torch
from torch import nn


class SpatialResidual(nn.Module):
    """Return a token residual; the last projection starts at exactly zero."""

    def __init__(self, dim=1024, hidden=64, map_size=(27, 27)):
        super().__init__()
        self.dim = dim
        self.map_size = tuple(map_size)  # Native LoRAT order is (width, height).
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, hidden)
        self.spatial = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.act = nn.GELU()
        self.up = nn.Linear(hidden, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, tokens):
        width, height = self.map_size
        if tokens.ndim != 3 or tokens.shape[1:] != (width * height, self.dim):
            raise ValueError(f'Unexpected search token shape: {tuple(tokens.shape)}')
        hidden = self.act(self.down(self.norm(tokens)))
        hidden = hidden.transpose(1, 2).reshape(tokens.shape[0], -1, height, width)
        hidden = self.act(self.spatial(hidden)).flatten(2).transpose(1, 2)
        return self.up(hidden).to(dtype=tokens.dtype)


class SpatialResidualHead(nn.Module):
    """Keep the existing head and all its decoding conventions unchanged."""

    def __init__(self, base_head, dim=1024, hidden=64, map_size=(27, 27)):
        super().__init__()
        self.base_head = base_head.requires_grad_(False).eval()
        self.spatial = SpatialResidual(dim, hidden, map_size)

    def train(self, mode=True):
        super().train(mode)
        self.base_head.eval()
        return self

    def forward(self, tokens):
        # The frozen head remains in the autograd graph so E can receive gradients.
        return self.base_head(tokens + self.spatial(tokens))


def install_spatial(adapter, checkpoint=None):
    """Install E once, returning its trainable branch; load branch-only weights."""
    model = adapter.model
    if isinstance(model.head, SpatialResidualHead):
        raise ValueError('Spatial residual is already installed')
    model.requires_grad_(False)
    model.eval()
    reference = next(model.head.parameters())
    wrapper = SpatialResidualHead(
        model.head, dim=model.embed_dim, hidden=64, map_size=model.x_size
    ).to(device=reference.device, dtype=reference.dtype)
    wrapper.eval()
    model.head = wrapper
    if checkpoint is not None:
        from safetensors.torch import load_file
        state = load_file(str(Path(checkpoint)), device='cpu')
        wrapper.spatial.load_state_dict(state, strict=True)
    # Only newly introduced residual parameters may be optimized.
    wrapper.spatial.requires_grad_(True)
    return wrapper.spatial
