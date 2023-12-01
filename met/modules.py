import torch
from torch import nn as nn


class MaskGenerator(nn.Module):
    """Module for generating Bernoulli mask."""

    def __init__(self, p: float):
        super().__init__()
        self.p = p

    def forward(self, x: torch.tensor):
        """Generate Bernoulli mask."""
        p_mat = torch.ones_like(x) * self.p
        return torch.bernoulli(p_mat)
