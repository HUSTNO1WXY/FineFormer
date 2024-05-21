import torch
from torch.nn import Module
import torch.nn as nn
from itertools import product
from torch.nn import functional as F


class layernorm2d(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.affine = nn.parameter.Parameter(torch.ones(dim), requires_grad=True)
        self.bias = nn.parameter.Parameter(torch.zeros(dim), requires_grad=True)

    def forward(self, x):
        # x: B*C*H*W
        mean, std = x.mean(dim=1, keepdim=True), x.std(dim=1, keepdim=True)
        return self.affine[None, :, None, None] * (x - mean) / (std + 1e-6) + self.bias[None, :, None, None]


class FullAttention(Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead

    def forward(self, q, k, v, mask0=None, mask1=None, temp=1):
        """ Multi-head scaled dot-product attention, a.k.a full attention.
        Args:
            q,k,v: [N, D, L]
            mask: [N, L]
        Returns:
            msg: [N,L]
        """
        bs = q.shape[0]
        q, k, v = q.view(bs, self.nhead, self.d_model // self.nhead, -1), k.view(bs, self.nhead,
                                                                                 self.d_model // self.nhead,
                                                                                 -1), v.view(bs, self.nhead,
                                                                                             self.d_model // self.nhead,
                                                                                             -1)
        # Compute the unnormalized attention and apply the masks
        QK = torch.einsum("nhdl,nhds->nhls", q, k)
        if mask0 is not None:
            QK.masked_fill_(~(mask0[:, None, :, None] * mask1[:, None, None]).bool(), float(-1e8))
        # Compute the attention and the weighted average
        softmax_temp = temp / q.size(2) ** .5  # sqrt(D)
        A = torch.softmax(softmax_temp * QK, dim=-1)
        queried_values = torch.einsum("nhls,nhds->nhdl", A, v).contiguous().view(bs, self.d_model, -1)
        return queried_values


def elu_feature_map(x):
    return F.elu(x) + 1


class LinearAttention(Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.feature_map = elu_feature_map
        self.eps = eps

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        """ Multi-Head linear attention proposed in "Transformers are RNNs"
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """
        Q = self.feature_map(queries)
        K = self.feature_map(keys)

        # set padded position to zero
        if q_mask is not None:
            Q = Q * q_mask[:, :, None, None]
        if kv_mask is not None:
            K = K * kv_mask[:, :, None, None]
            values = values * kv_mask[:, :, None, None]

        v_length = values.size(1)
        values = values / v_length  # prevent fp16 overflow
        KV = torch.einsum("nshd,nshv->nhdv", K, values)  # (S,D)' @ S,V
        Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps)
        queried_values = torch.einsum("nlhd,nhdv,nlh->nlhv", Q, KV, Z) * v_length

        return queried_values.contiguous()