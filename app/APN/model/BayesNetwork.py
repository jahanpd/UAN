# import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
# import torch.distributions as D
# from .Optimizer import RAdam
from adabelief_pytorch import AdaBelief
from app.APN.model.aux.attention import attention_bayes
from app.APN.model.aux.mapping import mapping


class bayes_network(nn.Module):

    def __init__(
        self,
        features,  # number features
        outcomes,  # number outcomes
        d_m,
        N=None,
        sample=True,
        lr=1e-3,
        scheduler=None,
        regr=1e-10,
        loss=nn.BCEWithLogitsLoss(),
    ):
        super(bayes_network, self).__init__()

        self.features = features
        self.outcomes = outcomes
        self.d_m = d_m

        if N is not None:
            self.N = nn.Parameter(
                T.tensor(N),
                requires_grad=False
                )
        else:
            self.N = nn.Parameter(
                T.randint(100, (8, 2)),
                requires_grad=False
                )

        self.xmap = mapping(
            self.features,
            self.outcomes,
            d_m,
            layers=[10, 10, 10]
            )
        self.x_attn = nn.Parameter(T.randn(1, self.features, self.d_m*2))

        self.xattn = attention_bayes(
            d_q=d_m,
            d_k=d_m,
            d_v=d_m,
            d_m=d_m*2
        )
        self.attn = attention_bayes(
            d_q=d_m,
            d_k=d_m,
            d_v=d_m,
            d_m=d_m)

        self.ffn = nn.Sequential(
            nn.Linear(d_m, d_m),
            nn.Softplus(),
            nn.Linear(d_m, d_m),
        )

        self.y_attn = nn.Parameter(T.randn(1, self.outcomes, self.d_m))

        self.decoder = nn.Sequential(
            nn.Linear(d_m, 10),
            nn.Softplus(),
            nn.Linear(10, 10),
            nn.Softplus(),
            nn.Linear(10, 1),
        )

        self.sample = sample
        self.loss_fn = loss
        # self.optim = RAdam(self.parameters(), lr=lr)
        self.optim = AdaBelief(
            self.parameters(), 
            lr=lr, 
            eps=1e-16, 
            betas=(0.9,0.999),
            weight_decouple=True,
            weight_decay=1e-4,
            rectify=True,
            fixed_decay=False,
            amsgrad=False
            )
        self.regr = regr
        self.scheduler1 = None
        self.kld = 0
        self.reg = 0
        self.alt = [1, 1]
        self.loss_1 = 0
        self.loss_2 = 0
        self.scaler_max = 1

    def forward(self, xv, yv=None):
        # find mask, reduce y_attn, run attn
        x_mask = (xv != -1).unsqueeze(1)  # (batch, 1, x)
        xs = xv.shape
        if yv is not None:
            y_mask = (yv != -1)
            y_mask_ = y_mask.unsqueeze(2).repeat(1, 1, xs[1]) & x_mask
        # (batch, y, x)

        # map each feature to latent space
        xm = self.xmap(xv)  # (b,x,embed)

        # pass input data through bayesian attn network
        # this determines the x_n|X  ?makes each feature independent
        x_attn = self.x_attn.repeat(xs[0], 1, 1)
        xshift = x_attn[:, :, :self.d_m] + xm*x_attn[:, :, self.d_m:]
        x_X, attn_x, valuesx = self.xattn(
            xshift, xshift, x_mask,
            sample=self.sample,
            apply_v=True,
            )  # (batch, x, d_m)

        h = x_X[:, :, :self.d_m] + xshift*x_X[:, :, self.d_m:]
        h = self.ffn(h)  # (b, x, d_m)

        # encode outcome dependent latent space with bayesian attn network
        y_attn = self.y_attn.repeat(xs[0], 1, 1)
        z, attn, values = self.attn(
            y_attn, h, x_mask,
            sample=self.sample,
            v_alt=xm,
            apply_v=True,
            )  # (batch, y, d_m), (batch, y, x), (batch, x, d_m)
        # posterior probability parameters
        # (batch, y, 2)
        z = z + y_attn  # residual connection to y (b, y, d_m)
        logits = self.decoder(z)  # (b, y, 2)

        # univariate distributions
        values_ = values.unsqueeze(1) + y_attn.unsqueeze(2)  # (b, y, x, d_m)
        logits_uni = self.decoder(values_)

        if yv is not None:
            self.loss_1 = self.loss_fn(
                logits.squeeze(-1).masked_select(y_mask),
                yv.masked_select(y_mask)
            )

            yv_ = yv.unsqueeze(2).repeat(1, 1, xs[1])  # (batch, y, x)
            self.loss_2 = self.loss_fn(
                logits_uni.squeeze(-1).masked_select(y_mask_),
                yv_.masked_select(y_mask_)
            )

            # regularisation
            self.reg = (
                xm.pow(2).sum(-1).mean()
                + values.pow(2).sum(-1).mean()
                + valuesx.pow(2).sum(-1).mean()
                + y_attn.pow(2).sum(-1).mean()
                + x_attn.pow(2).sum(-1).mean()
                )

            self.kld = (
                1 * self.attn.kld
                + 1 * self.xattn.kld
                )

            self.grad_loss =\
                self.loss_1\
                + self.kld\
                + self.regr * self.reg\

        return logits, logits_uni, 0, (attn, attn_x, 0)

    def step(self, step):
        self.optim.zero_grad()
        self.grad_loss.backward()
        # T.nn.utils.clip_grad_norm_(self.parameters(), 1, 2)
        self.optim.step()

    def loss(self):
        return {
            'Loss_1': float(self.loss_1),
            'Loss_2': float(self.loss_2),
            'Loss': (
                self.alt[0] * float(self.loss_1) +
                self.alt[1] * float(self.loss_2)),
            'KLD': float(self.kld),
            "REG": float(self.reg)
        }