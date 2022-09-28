import torch as T
import torch.nn as nn
import torch.nn.functional as F

# outcome dependent decoders
class decoder(nn.Module):
    def __init__(
        self,
        seq_len,
        d_m,
        layers=[5, 5],
        out_dim=2,
        padding_idx=0,
        activation=None,
    ):
        super(decoder, self).__init__()
        self.d_m = d_m
        self.layers = layers
        self.W = nn.ParameterList(
            [nn.Parameter(
                T.randn(1, seq_len, d_m, layers[0]))] +
            [nn.Parameter(T.randn(1, seq_len, h, h))
             for h in layers] +
            [nn.Parameter(T.randn(1, seq_len, layers[-1], out_dim))]
        )
        self.b = nn.ParameterList(
            [nn.Parameter(T.randn(1, seq_len, 1, layers[0]))] +
            [nn.Parameter(T.randn(1, seq_len, 1, h))
             for h in layers] +
            [nn.Parameter(T.randn(1, seq_len, 1, out_dim))]
        )
        self.activation = activation

    def forward(self, z, idx=[2]):
        for i in idx:
            z = z.unsqueeze(i)  # (b, y, 1, d_m) (b, 1, x, d_m)

        for i in range(len(self.layers) + 2):
            w = self.W[i]  # (1, y, d_m, h)
            b = self.b[i]  # (1, y, 1, h)
            if self.activation is not None:
                z = self.activation(z @ w + b)  # (batch, y, x, d_m)
            else:
                z = z @ w + b
        return z  # (batch, y, x, out_dim)

    def embedding(self, xc, weights):
        xs = xc.shape
        ws = weights.shape[1:]
        w = weights.index_select(0, xc.flatten())\
            .reshape(xs[0], xs[1], *ws)  # (batch, x, 1, embed)
        return w

