import torch as T
import torch.nn as nn
import torch.nn.functional as F


class mapping(nn.Module):
    def __init__(
        self,
        x_len,
        y_len,
        d_m,
        layers=[5, 5, 5],
        padding_idx=0
    ):
        super(mapping, self).__init__()

        self.d_m = d_m
        self.layers = layers
        self.W = nn.ParameterList(
            [nn.Parameter(T.randn(1, x_len, 1, layers[0])*0.001)] +
            [nn.Parameter(T.randn(1, x_len, h, h)*0.01)
             for h in layers] +
            [nn.Parameter(T.randn(1, x_len, layers[-1], d_m)*0.1)]
        )
        self.b = nn.ParameterList(
            [nn.Parameter(T.randn(1, x_len, 1, layers[0]))] +
            [nn.Parameter(T.randn(1, x_len, 1, h))
             for h in layers] +
            [nn.Parameter(T.randn(1, x_len, 1, d_m))]
        )

    def forward(self, xv):
        if len(xv.shape) == 2:
            xv = xv.unsqueeze(2)  # (b,x,1)
        z = self.linear(xv)
        return z

    def linear(self, xv):
        if len(xv.shape) == 2:
            x = xv.unsqueeze(2).unsqueeze(2)  # (b, x, 1, 1)
        else:
            x = xv.unsqueeze(2)
        # map
        x.requires_grad = True
        for i in range(len(self.layers)+2):
            W = self.W[i]  # (1, x, 1, h)
            b = self.b[i]  # (1, x, 1, h)
            x = x @ W + b  # (batch, x, 1, d_m)

        x = x.squeeze(2)  # (batch, x, d_m)
        return x
