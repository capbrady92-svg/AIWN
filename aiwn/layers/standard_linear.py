import torch.nn as nn


class StandardLinear(nn.Module):
    """Vanilla nn.Linear wrapper with a .flops() helper for fair comparison."""

    def __init__(self, in_d: int, out_d: int):
        super().__init__()
        self.lin = nn.Linear(in_d, out_d, bias=True)

    def forward(self, x):
        return self.lin(x)

    def flops(self) -> int:
        return 2 * self.lin.in_features * self.lin.out_features