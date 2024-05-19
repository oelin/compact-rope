# Compact RoPE

A more compact implementation of RoPE. Unlike most open-source implementations, this one is entirely self-contained.

```python
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
```

## Usage

```python
>>> from compact_rope import RoPE
>>>
>>> module = RoPE(embedding_dimension=256)
>>> x = torch.randn((1, 10, 256))
>>> position = torch.rand((1, 10))
>>> x = module(x, position)
```
