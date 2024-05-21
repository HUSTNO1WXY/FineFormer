import copy

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt

from src.loftr.loftr_module.attention import LinearAttention, FullAttention
from einops.einops import rearrange



def ones(tensor):
    if tensor is not None:
        tensor.data.fill_(0.5)

def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0.0)


class LoFTREncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 attention='linear'):
        super(LoFTREncoderLayer, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = LinearAttention() if attention == 'linear' else FullAttention() # 改
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model*2, d_model*2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model*2, d_model, bias=False),
        )

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.down_proj = nn.Linear(d_model * 2, 64, bias=False)
        self.act = nn.ReLU()
        self.up_proj = nn.Linear(64, d_model, bias=False)
        self.dropout = nn.Dropout(0.0)
        nn.init.kaiming_uniform_(self.down_proj.weight)
        nn.init.zeros_(self.up_proj.weight)

        #ACmix
        self.fc = nn.Conv2d(3 * self.nhead, 9, kernel_size=1, bias=True)
        self.dep_conv = nn.Conv2d(9 * d_model // self.nhead, d_model, kernel_size=3, bias=True,
                                  groups=d_model // self.nhead, padding=1)
        self.rate = torch.nn.Parameter(torch.Tensor(1))
        self.reset_parameters()

    def reset_parameters(self):
        ones(self.rate)
        # shift initialization for group convolution
        kernel = torch.zeros(9, 3, 3)
        for i in range(9):
            kernel[i, i // 3, i % 3] = 1.
        kernel = kernel.squeeze(0).repeat(self.dim * self.nhead, 1, 1, 1)
        self.dep_conv.weight = nn.Parameter(data=kernel, requires_grad=True)
        self.dep_conv.bias = zeros(self.dep_conv.bias)


    def forward(self, x, source, H, W, name, x_mask=None, source_mask=None):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
        bs = x.size(0)
        B0, T0, C0 = x.size()
        B1, T1, C1 = source.size()

        query, key, value = x, source, source
        query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)

        # TCmix
        if name == 'self':
            f_all = torch.cat((query, key, value), dim=-1) # [N, L, (H, 3D)]
            f_all = f_all.reshape(bs, T0, 3*self.nhead, -1).permute(0, 2, 1, 3) # B, 3*nhead, H*W, C//nhead
            f_conv = self.fc(f_all).permute(0, 3, 1, 2).reshape(bs, 9 * C0 // self.nhead, H,
                                                                W)  # B, 9*C//nhead, H, W
            out_conv = self.dep_conv(f_conv).permute(0, 2, 3, 1)  # B, H, W, C
            out_conv = rearrange(out_conv, 'n h w c -> n (h w) c')
        # TCmix

        message = self.attention(query, key, value, q_mask=x_mask, kv_mask=source_mask)  # [N, L, (H, D)]
        message = self.merge(message.view(bs, -1, self.nhead*self.dim))  # [N, L, C]

        message = self.norm1(message)
        # feed-forward network
        message = self.mlp(torch.cat([x, message], dim=2))  # 拼接
        message = self.norm2(message)

        down = self.down_proj(torch.cat([x, message], dim=2))
        down = self.act(down)
        down = self.dropout(down)
        up = self.up_proj(down)
        message = message + up * 0.1


        #TCmix
        if name == 'self':
            message = message + self.rate * out_conv

        return x + message


class FeatureTransformer(nn.Module):
    """A Local Feature Transformer (LoFTR) module."""

    def __init__(self, config):
        super(FeatureTransformer, self).__init__()

        self.config = config
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.layer_names = config['layer_names']
        encoder_layer = LoFTREncoderLayer(config['d_model'], config['nhead'], config['attention'])
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names))])  # 堆叠encoder layer
        self.conv = nn.Conv2d(128, 128, kernel_size=4, stride=4)  # 1/2 --- 1/8
        self.merge_layer = nn.Sequential(
            nn.Conv2d(256 + 128, 256 + 128, kernel_size=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(256 + 128, 256, kernel_size=1, bias=False),
        )
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat0, feat1, feat_f0, feat_f1, mask0=None, mask1=None):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, S, C]
            mask0 (torch.Tensor): [N, L] (optional)
            mask1 (torch.Tensor): [N, S] (optional)
        """

        bs = feat0.size(0)
        c, h, w = feat0.shape[1], feat0.shape[2], feat0.shape[3]
        feat0 = rearrange(feat0, 'n c h w -> n (h w) c').contiguous()
        feat1 = rearrange(feat1, 'n c h w -> n (h w) c').contiguous()

        feat_f0_conv, feat_f1_conv = self.conv(feat_f0.view(bs, c // 2, h * 4, w * 4)), \
                                         self.conv(feat_f1.view(bs, c // 2, h * 4, w * 4))  ###1/8

        for layer, name in zip(self.layers, self.layer_names):

            if name == 'self':
                feat0 = layer(feat0, feat0, h, w, name, mask0, mask0)
                feat0 = feat0.view(bs, c, h, w)
                feat0 = self.merge_layer(torch.cat((feat0, feat_f0_conv), dim=1))
                feat0 = feat0.view(bs, -1, c)

                feat1 = layer(feat1, feat1, h, w, name, mask0, mask1)
                feat1 = feat1.view(bs, c, h, w)
                feat1 = self.merge_layer(torch.cat((feat1, feat_f1_conv), dim=1))
                feat1 = feat1.view(bs, -1, c)

            elif name == 'cross':
                feat0 = layer(feat0, feat1, h, w, name, mask0, mask1)
                feat0 = feat0.view(bs, c, h, w)
                feat0 = self.merge_layer(torch.cat((feat0, feat_f0_conv), dim=1))
                feat0 = feat0.view(bs, -1, c)

                feat1 = layer(feat1, feat0, h, w, name, mask1, mask0)
                feat1 = feat1.view(bs, c, h, w)
                feat1 = self.merge_layer(torch.cat((feat1, feat_f1_conv), dim=1))
                feat1 = feat1.view(bs, -1, c)

            else:
                raise KeyError


        return feat0, feat1
