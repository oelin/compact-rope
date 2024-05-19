import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RoPE(nn.Module):
    def __init__(self, embedding_dimension: int) -> None:
        super().__init__()

        self.theta = torch.linspace(
            start=math.log(0.5 * math.pi),
            end=math.log(1000. * math.pi),
            steps=embedding_dimension // 2,
        ).exp().repeat_interleave(2, dim=-1)

    def forward(self, x: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        """Forward the module.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor (..., T, E).
        position : torch.Tensor
            The position tensor (..., T).
        """
        
        cos = torch.cos(position[..., None] * self.theta)
        sin = torch.sin(position[..., None] * self.theta)

        x_even, x_odd = x[..., :: 2], x[..., 1 :: 2]
        x_right = torch.stack((-x_odd, x_even), dim=-1).view(x.shape)
        x = x*cos + x_right*sin

        return x
