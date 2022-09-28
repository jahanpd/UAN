import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
# import torch.distributions as D
# from .Optimizer import RAdam
from adabelief_pytorch import AdaBelief
from app.APN.model.aux.attention import multihead_attention as attention
from app.APN.model.aux.mapping import mapping


class mha_network(nn.Module):

    def __init__(
        self,
        features,  # number features
        outcomes,  # number outcomes
        d_m,
        heads,
        N=None,
        sample=True,
        lr=1e-3,
        scheduler=None,
        regr=1e-10,
        loss=nn.BCEWithLogitsLoss(),
        bayes_attn=False,

    ):
        super(mha_network, self).__init__()

        self.features = features
        self.outcomes = outcomes
        self.d_m = d_m
        self.heads = heads

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

        self.xattn = attention(
            d_q=d_m,
            d_k=d_m,
            d_v=d_m,
            d_m=d_m*heads*2,
            heads=heads
        )

        self.ffn = nn.Sequential(
            nn.Linear(d_m, d_m),
            nn.Softplus(),
            nn.Linear(d_m, d_m),
        )

        self.attn = attention(
            d_q=d_m,
            d_k=d_m,
            d_v=d_m,
            d_m=d_m*heads,
            heads=heads
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
            betas=(0.9, 0.999),
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
        self.eps = 1  # for fast gradient sign method

    def forward(self, xv, yv=None):
        x_mask = (xv != -1).unsqueeze(1)  # (batch, 1, x)
        xs = xv.shape
        if yv is not None:
            y_mask = (yv != -1)
            y_mask_ = y_mask.unsqueeze(2).repeat(1, 1, xs[1]) & x_mask

        logits, univariate_logits, N, (attn, attn_x),\
            (xm, values, valuesx, y_attn, x_attn)\
            = self._forward(xv=xv)

        if yv is not None:
            # select random head for training purposes
            h = np.random.randint(self.heads)
            lg = logits[:, h, :]  # (b, y)

            # lg = logits.mean(1)  # (b, y)
            self.loss_1 = self.loss_fn(
                lg.masked_select(y_mask),
                yv.masked_select(y_mask)
            )

            # find adversarial example wrt to loss_1
            # xm.retain_grad()
            # self.optim.zero_grad()
            # self.loss_1.backward()
            # x_grad = T.sign(xm.grad.data)
            # xm_adv = (xm.data + self.eps * x_grad).detach()

            # # combine data
            # xv_all = T.cat([xv, xv], 0)
            # xm = self.xmap(xv)  # (b,x,embed)
            # xm_all = T.cat([xm, xm_adv], 0)
            # lgt, uni_lgt, _, (_, _), (xm, values, valuesx, y_attn, x_attn)\
            #     = self._forward(xv=xv_all, xm=xm_all)
            lgt, uni_lgt = logits, univariate_logits

            # yv = T.cat([yv, yv], 0)
            # y_mask = T.cat([y_mask, y_mask], 0)
            lg = lgt[:, h, :]  # (b, h, y)
            # lg = lgt.mean(1)  # (b, y)
            self.loss_1 = self.loss_fn(
                lg.masked_select(y_mask),
                yv.masked_select(y_mask)
            )

            # ul = uni_lgt.mean(1)
            ul = uni_lgt[:, h, :, :]  # (b, h, y, x)
            yv_ = yv.unsqueeze(2).repeat(1, 1, xs[1])  # (batch, y, x)
            # y_mask_ = T.cat([y_mask_, y_mask_], 0)
            self.loss_2 = self.loss_fn(
                ul.masked_select(y_mask_),
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

            self.grad_loss = self.loss_1\
                + self.regr * self.reg\
                # + self.alt[1] * self.loss_2\

        return logits, univariate_logits, N, (attn, attn_x)

    def _forward(self, xv, xm=None, yv=None):
        # find mask, reduce y_attn, run attn
        x_mask = (xv != -1).unsqueeze(1)  # (batch, 1, x)
        xs = xv.shape
        # (batch, y, x)

        # find count values for beta prior
        N = self.N.unsqueeze(0)
        N = N.repeat(xs[0], 1, 1)

        if xm is None:
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
            y_add=False
            )  # (batch, x, d_m)
        x_X = x_X.mean(1)
        h = x_X[:, :, :self.d_m] + xshift*x_X[:, :, self.d_m:]
        h = self.ffn(h)  # (b, x, d_m)

        # encode outcome dependent latent space with bayesian attn network
        y_attn = self.y_attn.repeat(xs[0], 1, 1)  # (b, y, d_m)
        z, attn, values = self.attn(
            y_attn, h, x_mask,
            sample=self.sample,
            v_alt=xm,
            apply_v=True,
            )  # (b, h, y, d_m), (b, h, y, x), (b, h, x, d_m)

        logits = self.decoder(z).squeeze(-1)  # (b, h, y)

        # univariate distributions
        values_ = values.unsqueeze(2) + y_attn.unsqueeze(1).unsqueeze(3)
        # (b, h, y, x, d_m)
        univariate_logits = self.decoder(values_).squeeze(-1)  # (b, h, y, x)

        return logits, univariate_logits, N, (attn, attn_x),\
            (xm, values, valuesx, y_attn, x_attn)

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

    def UCE(self, yv, post_params, mask):
        # prep indicators of outcomes
        yv_ = F.one_hot(yv + 1, 3)[..., 1:]
        alpha_0 = post_params.sum(-1, keepdims=True)
        label = (yv_ * -1.0) + 1.0
        loss = (label * (T.digamma(alpha_0) - T.digamma(post_params))).sum(-1)
        return loss.masked_select(mask).mean()
