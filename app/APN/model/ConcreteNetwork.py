# import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
# import torch.distributions as D
# from .Optimizer import RAdam
from adabelief_pytorch import AdaBelief
from torch.optim.lr_scheduler import MultiStepLR
from app.APN.model.aux.attention import attention
from app.APN.model.aux.mapping import mapping


class network(nn.Module):

    def __init__(
        self,
        features,  # number features
        outcomes,  # number outcomes
        d_m,
        N=None,
        d_l=4,
        sample=True,
        lr=1e-3,
        scheduler=None,
        regr=1e-10,
        loss=nn.BCELoss(),
        init_min=0.5,
        init_max=0.5
    ):
        super(network, self).__init__()

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

        # set up for concrete distribution training mask
        self.weight_regulariser = 1e-6
        self.dropout_regulariser = 1e-7
        self.p_logit = nn.Parameter(
            T.empty(outcomes, features, 1).uniform_(init_min, init_max))
        self.p = T.sigmoid(self.p_logit)
        self.conc_reg = 0.0

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
            d_v=d_m*2,
            d_m=d_m*2
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
            d_m=d_m)

        self.y_attn = nn.Parameter(T.randn(1, self.outcomes, self.d_m))

        self.scaling = nn.Parameter(T.randn(self.outcomes, self.features))

        self.decoder_uni = nn.Sequential(
            nn.Linear(d_m, 10),
            nn.Softplus(),
            nn.Linear(10, 10),
            nn.Softplus(),
            nn.Linear(10, 2),
            nn.Softplus(),
        )
        self.decoder_full = nn.Sequential(
            nn.Linear(d_m, 10),
            nn.Softplus(),
            nn.Linear(10, 10),
            nn.Softplus(),
            nn.Linear(10, 2),
            nn.Softplus(),
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
            weight_decay=1e-10,
            rectify=True,
            fixed_decay=False,
            amsgrad=False
            )
        self.regr = regr
        self.scheduler1 = MultiStepLR(self.optim, milestones=[15], gamma=0.1)
        self.kld = 0
        self.reg = 0
        self.alt = [1, 1]
        self.loss_1 = 0
        self.loss_1_alt = 0
        self.loss_2 = 0
        self.scaler_max = 1

    def forward(self, xv, yv=None, training=False):
        # find mask of missing vars, reduce y_attn, run attn
        x_mask = (xv != -1).unsqueeze(1)  # (batch, 1, x)

        xs = xv.shape
        if yv is not None:
            y_mask = (yv != -1)
            y_mask_ = y_mask.unsqueeze(2).repeat(1, 1, xs[1]) & x_mask
        # (batch, y, x)

        # find count values for beta prior
        N = self.N.unsqueeze(0)
        N = N.repeat(xs[0], 1, 1)

        # initialize beta prior with baseline
        prior_params = (N / N[:, :, 1:2])  # (b, y, 2)

        # map each feature to latent space
        xm_ = self.xmap(xv)  # (b,x,embed)
        xm = xm_.detach()

        x_attn = self.x_attn.repeat(xs[0], 1, 1)
        xshift = x_attn[:, :, :self.d_m] + xm*x_attn[:, :, self.d_m:]
        # concrete distribution training mask on mapped values
        if training:
            _, conc_mask = self._concrete_mask(xm_, y_mask)  # (b, f, 1)
            # xm = xm_.detach()
            x_mask_ = x_mask.to(T.long) * conc_mask.permute(0, 2, 1).to(T.long)

            # concrete distribution regularisation terms
            sum_of_squares = xm_.pow(2).sum(-1).unsqueeze(1)  # (b, 1, x)
            weights_reg = (self.weight_regulariser * sum_of_squares
                           / (1.0 - self.p.permute(2, 0, 1)))\
                .sum(2).sum(1).mean()
            dropout_reg = self.p * T.log(self.p)
            dropout_reg += (1.0 - self.p) * T.log(1.0 - self.p)
            dropout_reg *= self.dropout_regulariser * self.d_m
            self.regularisation = weights_reg + dropout_reg.sum()  # (1, d_m)
            xshift = xshift * conc_mask
        else:
            x_mask_ = x_mask

        # pass input data through attn network

        x_X, attn_x, _ = self.xattn(
            xshift, xshift, x_mask_,
            apply_v=True,
            v_alt=x_attn,
            )  # (batch, x, d_m)

        h = x_X[:, :, :self.d_m] + xshift*x_X[:, :, self.d_m:]
        h = self.ffn(h)  # (b, x, d_m)

        y_attn = self.y_attn.repeat(xs[0], 1, 1)

        # encode outcome dependent latent space with bayesian attn network
        if training:
            h = h * conc_mask

        z, attn, _ = self.attn(
            y_attn, h, x_mask_,
            v_alt=None,
            apply_v=True,
            )  # (batch, y, d_m), (batch, y, x), (batch, x, d_m)
        # posterior probability parameters
        # (batch, y, 2)

        multiplier = ((F.softplus(self.scaling)) * x_mask_).sum(
            -1, keepdims=True)  # (b, y, 1)
        z = (z * multiplier) + y_attn  # residual connection to y (b, y, d_m)
        beta_params = (1 + self.decoder_full(z))  # (b, y, 2)
        beta = T.distributions.Beta(
            beta_params[:, :, 0], beta_params[:, :, 1]
        )

        # use the commented out prior if not a rebalanced dataset
        # beta_prior = T.distributions.Beta(
        #     prior_params[:, :, 0], prior_params[:, :, 1]
        # )
        beta_prior = T.distributions.Beta(
            T.ones_like(prior_params[:, :, 0]),
            T.ones_like(prior_params[:, :, 1])
        )

        single_prob = beta.mean

        # univariate distributions
        values_ = xm_.unsqueeze(1) + y_attn.unsqueeze(2)  # (b, y, x, d_m)
        univariate_params = 1 + self.decoder_uni(values_)  # (b, y, x, d_m)
        univariate_beta = T.distributions.Beta(
            univariate_params[:, :, :, 0], univariate_params[:, :, :, 1]
        )
        uni_prior = T.distributions.Beta(
            T.ones_like(univariate_params[:, :, :, 0]),
            T.ones_like(univariate_params[:, :, :, 1])
        )
        probs = univariate_beta.mean  # (b, y, x)

        if yv is not None:
            self.loss_1 = self.UCE(
                yv.to(T.long), beta_params, y_mask
            )

            yv_ = yv.unsqueeze(2).repeat(1, 1, xs[1])  # (batch, y, x)
            self.loss_2 = self.UCE(
                yv_.to(T.long),
                univariate_params,
                y_mask_
            )

            # regularisation
            self.reg = (
                # self.alt[1] * xm.pow(2).sum(-1).mean()
                self.regularisation
                + 1e-5 * self.alt[0] * self.scaling.pow(2).sum()
                + self.regr * self.alt[1] * y_attn.pow(2).sum(-1).mean()
                + self.regr * self.alt[1] * x_attn.pow(2).sum(-1).mean()
                )

            self.beta_kld = T.distributions.kl_divergence(
                beta_prior, beta
            ).masked_select(y_mask).abs().mean()

            self.uni_kld = T.distributions.kl_divergence(
                uni_prior, univariate_beta
            ).masked_select(y_mask_).abs().mean()

            self.kld = (
                self.alt[0] * 1e-3 * self.beta_kld
                + self.alt[1] * 1e-3 * self.uni_kld
                )

            self.grad_loss =\
                self.alt[0] * self.loss_1\
                + self.alt[1] * self.loss_2\
                + self.kld\
                + self.reg

        return single_prob, probs, N, (attn, attn_x, beta_params)

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

    def _concrete_mask(self, x, ymask):
        eps = 1e-7
        tmp = 0.1

        self.p = T.sigmoid(self.p_logit)  # (outcomes, features, 1)

        # build p|y matrix for each sample in batch
        indices = ymask.nonzero()[:, 1]  # (batch)

        p = T.index_select(self.p, 0, indices)  # (b, f, 1)

        u_noise = T.rand(x.shape[0], self.features, 1)

        drop_prob = (T.log(p + eps) -
                     T.log(1 - p + eps) +
                     T.log(u_noise + eps) -
                     T.log(1 - u_noise + eps))

        drop_prob = T.sigmoid(drop_prob / tmp)

        random_tensor = 1 - drop_prob
        retain_prob = 1 - p

        x = (x * random_tensor) / retain_prob

        return x, drop_prob  # (b, features, 1)
