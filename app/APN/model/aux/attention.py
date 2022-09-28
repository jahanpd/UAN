import math
import torch as T
import torch.nn as nn
import torch.nn.functional as F


class attention_bayes(nn.Module):
    def __init__(
        self,
        d_q,
        d_k,
        d_v,
        d_m
    ):
        super(attention_bayes, self).__init__()

        self.q = nn.Sequential(
            nn.Linear(d_q, d_m),
            nn.Softplus(),
            nn.Linear(d_m, d_m),
            nn.Softplus(),
            nn.Linear(d_m, d_m),
            nn.Softplus()
        )
        self.k = nn.Sequential(
            nn.Linear(d_k, d_m),
            nn.Softplus(),
            nn.Linear(d_m, d_m),
            nn.Softplus(),
            nn.Linear(d_m, d_m),
            nn.Softplus()
        )

        self.v = nn.Sequential(
            nn.Linear(d_v, d_m),
            nn.Softplus(),
            nn.Linear(d_m, d_m),
            nn.Softplus(),
            nn.Linear(d_m, d_m),
        )
        self.kld = 0

    def forward(self, y, x, mask, v_alt=None, sample=True, apply_v=True):
        q = self.q(y)  # (batch, y, d_m)
        k = self.k(x)  # (batch, x, d_m)
        if apply_v:
            v = self.v(x)  # (batch, x, d_m)
            if v_alt is not None:
                va = self.v(v_alt)
            else:
                va = v
        else:
            v = x
            if v_alt is not None:
                va = self.v(v_alt)
            else:
                va = v

        qk = q @ k.permute(0, 2, 1)  # (batch, y, x)
        # mask needs to be 0 for masking and 1 for including
        alpha = qk + 1.0
        mu = self._to_mu(alpha, mask)
        std = self._to_std(alpha, mask)

        self.kld = self.dirichlet_kld(alpha, mask).mean()

        if sample:
            attn = self.sample(mu, std, mask)
            return attn @ va, attn, va  # (batch, y, d_m)
        else:
            mask = ((mask * -1) + 1) * -1e9
            attn = F.softmax(mu + mask, dim=-1)
            return attn @ va, attn, va  # (batch, y, d_m)

    def dirichlet_kld(self, alpha, mask):
        a0 = (alpha * mask).sum(-1, keepdims=True)
        kl = (
            T.lgamma(a0)
            - T.sum(T.lgamma(alpha) * mask, dim=-1,  keepdims=True)
            + T.sum((alpha * T.digamma(alpha)) * mask, dim=-1,  keepdims=True)
            - T.sum((alpha * T.digamma(a0)) * mask, dim=-1,  keepdims=True)
            - T.mean(
                (T.digamma(alpha) - T.digamma(a0)) * mask,
                dim=-1, keepdims=True)
        )
        return T.abs(kl)

    def _to_mu(self, alpha, mask):
        log_alpha = T.log(alpha)
        mu = log_alpha - (log_alpha * mask).mean(-1, keepdims=True)
        return mu * mask

    def _to_std(self, alpha, mask):
        latent_dims = mask.sum(-1, keepdims=True)
        k1 = 1.0 - (2.0 / latent_dims)
        k2 = 1 / (latent_dims ** 2)
        sigma = k1 * (1 / alpha) + k2 * ((1 / alpha) * mask).sum(
            -1, keepdims=True)
        return sigma * mask

    def sample(self, mu, sigma, mask):
        e = T.randn_like(sigma)
        mask = ((mask * -1) + 1) * -1e9
        return F.softmax(mu + sigma*e + mask, dim=-1)


class attention(nn.Module):
    def __init__(
        self,
        d_q,
        d_k,
        d_v,
        d_m
    ):
        super(attention, self).__init__()

        self.q = nn.Sequential(
            nn.Linear(d_q, d_m),
            nn.Softplus(),
            nn.Linear(d_m, d_m),
            nn.Softplus(),
            nn.Linear(d_m, d_m),
        )
        self.k = nn.Sequential(
            nn.Linear(d_k, d_m),
            nn.Softplus(),
            nn.Linear(d_m, d_m),
            nn.Softplus(),
            nn.Linear(d_m, d_m),
        )

        self.v = nn.Sequential(
            nn.Linear(d_v, d_m),
            nn.Softplus(),
            nn.Linear(d_m, d_m),
            nn.Softplus(),
            nn.Linear(d_m, d_m),
        )

        self.reg = 0

    def forward(
        self, y, x, mask, v_alt=None, sample=None, apply_v=True, y_add=False
    ):
        q = self.q(y)  # (batch, y, d_m)
        k = self.k(x)  # (batch, x, d_m)
        if apply_v:
            if v_alt is not None:
                va = self.v(v_alt)
            else:
                va = self.v(x)  # (batch, x, d_m)
        else:
            if v_alt is not None:
                va = v_alt
            else:
                va = x

        qk = q @ k.permute(0, 2, 1)  # (batch, y, x)
        # mask needs to be 0 for masking and 1 for including

        mask = ((mask * -1) + 1) * -1e9
        attn = F.softmax(qk + mask, dim=-1)

        if y_add:
            out = (
                (attn.unsqueeze(-1) * va.unsqueeze(1)) + y.unsqueeze(2)
                ).sum(2)

            return out, attn, va  # (batch, y, d_m)

        else:
            return (attn @ va), attn, va  # (batch, y, d_m)


class multihead_attention(nn.Module):
    def __init__(
        self,
        d_q,
        d_k,
        d_v,
        d_m,
        heads
    ):
        super(multihead_attention, self).__init__()
        if d_m % heads != 0:
            raise ValueError(
                '`d_m`({}) should be divisible by `heads`({})'.format(
                    d_m, heads))

        self.q = nn.Sequential(
            nn.Linear(d_q, d_m),
            nn.Softplus(),
            nn.Linear(d_m, d_m),
            nn.Softplus(),
            nn.Linear(d_m, d_m),
        )
        self.k = nn.Sequential(
            nn.Linear(d_k, d_m),
            nn.Softplus(),
            nn.Linear(d_m, d_m),
            nn.Softplus(),
            nn.Linear(d_m, d_m),
        )

        self.v = nn.Sequential(
            nn.Linear(d_v, d_m),
            nn.Softplus(),
            nn.Linear(d_m, d_m),
            nn.Softplus(),
            nn.Linear(d_m, d_m),
        )
        self.heads = heads
        self.d_k_sqrt = math.sqrt(d_k)
        self.bias

    def forward(
        self,
        y,
        x,
        mask,
        v_alt=None,
        sample=None,
        apply_v=True,
        y_add=True
    ):
        q = self.q(y)  # (batch, y, d_m)
        k = self.k(x)  # (batch, x, d_m)
        if apply_v:
            if v_alt is not None:
                va = self.v(v_alt)
            else:
                va = self.v(x)  # (batch, x, d_m)
        else:
            if v_alt is not None:
                va = self.v(v_alt)
            else:
                va = x

        q = self._reshape_to_batches(q)  # (b*h, x, d_m/h)
        k = self._reshape_to_batches(k)
        v = self._reshape_to_batches(va)

        qk = (q @ k.permute(0, 2, 1))/self.d_k_sqrt  # (batch*heads, y, x)
        # mask needs to be -1e9 for masking and 1 for including
        mask = ((mask * -1) + 1) * -1e9  # (b, y, x)
        mask = self._reshape_mask(mask)  # (b*h, y, x)
        attn = F.softmax(qk + mask, dim=-1)

        v = self._reshape_from_batches(v)  # (b, h, x, d_m/h)
        attn = self._reshape_from_batches(attn)  # (b, h, y, x)

        if y_add:
            out = ((attn.unsqueeze(-1) * v.unsqueeze(2))  # (b, h, y, x, d_m/h)
                   + y.unsqueeze(1).unsqueeze(3)  # (b, 1, y, 1, d_m/h)
                   ).sum(3)  # (b, h, y, d_m/h)

            return out, attn, v  # (batch, h, y, d_m/h)

        else:
            out = attn @ v  # (b, h, y, d_m/h)
            return out, attn, v  # (batch, h, y, d_m/h)

    def _reshape_to_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        sub_dim = in_feature // self.heads
        return x.reshape(batch_size, seq_len, self.heads, sub_dim)\
                .permute(0, 2, 1, 3)\
                .reshape(batch_size * self.heads, seq_len, sub_dim)

    def _reshape_mask(self, x):
        batch_size, queries, seq_len = x.size()
        return x.unsqueeze(1).repeat(1, self.heads, 1, 1)\
                .reshape(batch_size * self.heads, queries, seq_len)

    def _reshape_from_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        batch_size //= self.heads
        # out_dim = in_feature * self.heads
        return x.reshape(batch_size, self.heads, seq_len, in_feature)
