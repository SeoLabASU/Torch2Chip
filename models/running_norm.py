"""
Layernorm with
"""

import torch
import torch.nn as nn
from torch import Tensor 

class LayerRunNorm(nn.LayerNorm):
    def __init__(self, normalized_shape: int, eps: float = 0.00001, elementwise_affine: bool = True, device=None, dtype=None, batch_size:int=128, momentum:float=0.9):
        super().__init__(normalized_shape, eps, elementwise_affine, device, dtype)
        self.register_buffer("running_mean", torch.zeros(batch_size))
        self.register_buffer("running_var", torch.ones(batch_size))
        self.momentum = momentum

    def forward(self, x:Tensor):
        assert len(x.size()) == 3, "Incorrect input size"

        if self.training:
            mean = torch.mean(x, dim=[1,2])
            var = x.var([1, 2], unbiased=False)
            n = x.numel() / x.size(1)

            with torch.no_grad():
                self.running_mean = self.momentum * mean\
                    + (1 - self.momentum) * self.running_mean
                # update running_var with unbiased var
                self.running_var = self.momentum * var * n / (n - 1)\
                    + (1 - self.momentum) * self.running_var

        else:
            mean = self.running_mean
            var = self.running_var
        
        x = (x - mean[:, None, None]) / (torch.sqrt(var[:, None, None] + self.eps))
        x = x * self.weight + self.bias
        return x

    def extra_repr(self) -> str:
        return super().extra_repr() + ", momentum={}".format(self.momentum)

# if __name__ == "__main__":
#     norm = LayerRunNorm(384)
#     x = torch.randn(128, 65, 384)
#     y = norm(x)