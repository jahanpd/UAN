import numpy as np
import torch as T
from torch import nn
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence
from torch.optim import Adam


class BayesianLR(nn.Module):
    def __init__(self, in_features):
        super(BayesianLR, self).__init__()
        self.W_mu = nn.Parameter(T.randn(1, 1, in_features))
        self.W_logvar = nn.Parameter(T.randn(1, 1 ,in_features))
        self.b_mu = nn.Parameter(T.randn(1, 1))
        self.b_logvar = nn.Parameter(T.randn(1, 1))
        self.optim = Adam(self.parameters(), lr=1e-3)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.ce = 0
        self.kld = 0
        self.total = 0
        self.sample=True
        self.scheduler1 = None
    def forward(self, xv, yv=None):
        x = xv.unsqueeze(2)  # (b, x, 1)
        e_w = T.randn(x.shape[0], 1, x.shape[1])  # (b, 1, x)
        W = self.W_mu + e_w * T.exp(self.W_logvar)**0.5 # (b, 1, x)
        e_b = T.randn(x.shape[0], 1)
        b = self.b_mu + e_b * T.exp(self.b_logvar)**0.5 # (b, 1)
        logit = (W @ x).squeeze(-1) + b # (b, 1)

        if yv is not None:
            self.ce = self.loss_fn(
                logit, yv
            )
            self.kld = self.kullback(self.W_mu, self.W_logvar) +\
                self.kullback(self.b_mu, self.b_logvar)
            self.grad_loss = self.ce + 1e-3*self.kld

        return logit

    def step(self, step):
        self.optim.zero_grad()
        self.grad_loss.backward()
        self.optim.step()
    
    def loss(self):
        return {
            'Loss': float(self.ce),
            'KLD': float(self.kld)
        }
    
    def kullback(self, mu, logvar):
        kld = - 0.5 * (1 + logvar - T.exp(logvar) - mu**2)
        return kld.abs().mean()