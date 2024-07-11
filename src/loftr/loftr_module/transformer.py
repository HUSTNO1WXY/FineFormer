import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from matplotlib import pyplot as plt

from src.loftr.loftr_module.attention import layernorm2d, LinearAttention, FullAttention

def ones(tensor):
    if tensor is not None:
        tensor.data.fill_(0.5)

def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0.0)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*4, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

class LoFTREncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 attention='linear'):
        super(LoFTREncoderLayer, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead
        self.d_model =d_model

        self.q_proj = nn.Conv1d(d_model, d_model, kernel_size=1, bias=False)
        #self.q_proj_re1 = nn.Conv1d(d_model + 128, d_model, kernel_size=1, bias=False)
        self.k_proj = nn.Conv1d(d_model, d_model, kernel_size=1, bias=False)
        self.v_proj = nn.Conv1d(d_model + 128, d_model, kernel_size=1, bias=False)
        #self.v_proj_re1 = nn.Conv1d(d_model, d_model, kernel_size=1, bias=False)
        self.attention = LinearAttention()
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


    def forward(self, x0, x1, pos0, pos1, x_mask=None, source_mask=None):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
        bs = x0.size(0)
        #x0_pos = torch.cat([x1, pos0], dim=1)
        #query, key = self.q_proj_re1(x0_pos.view(bs, self.d_model + 128, -1)), self.k_proj(x1.view(bs, self.d_model, -1))
        query, key = self.q_proj(x0.view(bs, self.d_model, -1)), self.k_proj(x1.view(bs, self.d_model, -1))
        x1_pos = torch.cat([x1, pos1], dim=1)
        # x1_pos = x1
        value = self.v_proj(x1_pos.view(bs, self.d_model + 128, -1))
        # value = self.v_proj_re1(x1_pos.view(bs, self.d_model, -1))
        value = value.view(bs, self.nhead, self.d_model // self.nhead, -1)
        query, key = query.transpose(-2, -1).contiguous(), key.transpose(-2, -1).contiguous()
        query, key = query.view(bs, -1, self.nhead, self.dim), key.view(bs, -1, self.nhead, self.dim)
        value = value.permute(0, -1, 1, 2)

        message = self.attention(query, key, value, q_mask=x_mask, kv_mask=source_mask)  # [N, L, (H, D)]
        message = self.merge(message.view(bs, -1, self.nhead*self.dim))  # [N, L, C]

        message = self.norm1(message)
        # feed-forward network
        x0 = x0.view(bs, -1, self.d_model)
        message = self.mlp(torch.cat([x0, message], dim=2))  # 拼接
        message = self.norm2(message)

        down = self.down_proj(torch.cat([x0, message], dim=2))
        down = self.act(down)
        down = self.dropout(down)
        up = self.up_proj(down)
        message = message + up * 0.1

        return x0 + message



class EncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 attention='linear'):
        super(EncoderLayer, self).__init__()

        self.dim = d_model // nhead
        self.d_model = d_model
        self.nhead = nhead
        self.d_value = d_model + 128

        self.q_proj = nn.Conv1d(d_model, d_model, kernel_size=1, bias=False)
        # self.q_proj_re1 = nn.Conv1d(d_model + 128, d_model, kernel_size=1, bias=False)
        self.k_proj = nn.Conv1d(d_model, d_model, kernel_size=1, bias=False)
        self.v_proj = nn.Conv1d(d_model + 128, d_model, kernel_size=1, bias=False)
        #self.v_proj_re1 = nn.Conv1d(d_model, d_model, kernel_size=1, bias=False)
        self.attention = FullAttention(d_model, nhead)
        self.merge_head = nn.Conv1d(d_model, d_model, kernel_size=1, bias=False)

        # feed-forward network
        self.merge_f = nn.Sequential(
            nn.Conv2d(d_model * 2, d_model * 2, kernel_size=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(d_model * 2, d_model, kernel_size=1, bias=False),
        )

        self.norm1 = layernorm2d(d_model)
        self.norm2 = layernorm2d(d_model)


    def forward(self, x0, x1, pos0, pos1, x_mask=None, source_mask=None):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
        bs, h, w = x0.shape[0], x0.shape[2], x0.shape[3]
        x0_pos = torch.cat([x0, pos0], dim=1)
        #queries, keys = self.q_proj_re1(x0_pos.view(bs, self.d_value, -1)), self.k_proj(x1.view(bs, self.d_model, -1))
        queries, keys = self.q_proj(x0.view(bs, self.d_model, -1)), self.k_proj(x1.view(bs, self.d_model, -1))

        x1_pos = torch.cat([x1, pos1], dim=1)
        # x1_pos = x1
        # values = self.v_proj_re1(x1_pos.view(bs, self.d_model, -1))
        values = self.v_proj(x1_pos.view(bs, self.d_value, -1))
        values = values.view(bs,self.nhead,self.d_model//self.nhead,-1)
        message = self.attention(queries, keys, values, mask0=x_mask, mask1=source_mask)
        msg = self.merge_head(message).view(bs, -1, h, w)
        msg = self.norm2(self.merge_f(torch.cat([x0, self.norm1(msg)], dim=1)))

        return x0 + msg



class LocalFeatureTransformer_UNet(nn.Module):
    """A Local Feature Transformer (LoFTR) module."""

    def __init__(self, config):
        super(LocalFeatureTransformer_UNet, self).__init__()

        self.config = config
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.d_pos = 128
        self.conv1 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=3//2)  # 1/16
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=3//2)  # 1/32
        self.conv3 = nn.Conv2d(128, 64, kernel_size=8, stride=8)  # 1/2 --- 1/16
        self.conv4 = nn.Conv2d(128, 32, kernel_size=4, stride=4)  # 1/2 --- 1/8
        self.merge_layer2 = nn.Sequential(
            nn.Conv2d(256*2 + 64, 256 * 2, kernel_size=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(256 * 2, 256, kernel_size=1, bias=False),
        )

        self.merge_layer1 = nn.Sequential(
            nn.Conv2d(256 * 2 + 32, 256 * 2, kernel_size=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(256 * 2, 256, kernel_size=1, bias=False),
        )


        # #####Ablation#####
        # self.merge_layer3 = nn.Sequential(
        #     nn.Conv2d(256 + 64, 256 * 2, kernel_size=1, bias=False),
        #     nn.ReLU(True),
        #     nn.Conv2d(256 * 2, 256, kernel_size=1, bias=False),
        # )
        # self.merge_layer4 = nn.Sequential(
        #     nn.Conv2d(256 + 32, 256 * 2, kernel_size=1, bias=False),
        #     nn.ReLU(True),
        #     nn.Conv2d(256 * 2, 256, kernel_size=1, bias=False),
        # )

        self.up3_2 = Upsample(int(256))
        self.up2_1 = Upsample(int(256))


        self.pos_transform = nn.Conv2d(config['d_model'], self.d_pos, kernel_size=1, bias=False)
        self.encoder_layer1 = LoFTREncoderLayer(config['d_model'], config['nhead'], config['attention'])
        self.encoder_layer2 = LoFTREncoderLayer(config['d_model'], config['nhead'], config['attention'])
        self.encoder_layer3 = EncoderLayer(config['d_model'], config['nhead'], config['attention'])
        self.decoder_layer1 = LoFTREncoderLayer(config['d_model'], config['nhead'], config['attention'])
        self.decoder_layer2 = LoFTREncoderLayer(config['d_model'], config['nhead'], config['attention'])
        self.decoder_layer3 = EncoderLayer(config['d_model'], config['nhead'], config['attention'])
        self._reset_parameters()


        # self.ablation1 = nn.Sequential(
        #     nn.Conv2d(256*2 , 256 * 2, kernel_size=1, bias=False),
        #     nn.ReLU(True),
        #     nn.Conv2d(256 * 2, 256, kernel_size=1, bias=False),
        # )
        # self.ablation2 = nn.Sequential(
        #     nn.Conv2d(256 * 2, 256 * 2, kernel_size=1, bias=False),
        #     nn.ReLU(True),
        #     nn.Conv2d(256 * 2, 256, kernel_size=1, bias=False),
        # )

    def _reset_parameters(self):
        for name, p in self.named_parameters():
            if 'temp' in name or 'sample_offset' in name:
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat0, feat1, feat_f0, feat_f1, pos0, pos1, mask0=None, mask1=None):

        bs = feat0.size(0)
        c, h, w = feat0.shape[1], feat0.shape[2], feat0.shape[3]

        pos0, pos1 = self.pos_transform(pos0), self.pos_transform(pos1)  # 把位置编码降低维度
        pos0, pos1 = pos0.expand(bs, -1, -1, -1), pos1.expand(bs, -1, -1, -1)
        # pos0, pos1 = torch.zeros_like(pos0), torch.zeros_like(pos1)

        feat_f0_layer1, feat_f1_layer1 = self.conv4(feat_f0.view(bs, c // 2, h * 4, w * 4)), \
                                         self.conv4(feat_f1.view(bs, c // 2, h * 4, w * 4))  ###1/8
        feat_f0_layer2, feat_f1_layer2 = self.conv3(feat_f0.view(bs, c // 2, h * 4, w * 4)), \
                                         self.conv3(feat_f1.view(bs, c // 2, h * 4, w * 4))  ###1/16


        # ####Ablation merge####
        # feat0, feat1 = self.merge_layer4(torch.cat((feat_f0_layer1, feat0), dim=1)), \
        #                self.merge_layer4(torch.cat((feat_f1_layer1, feat1), dim=1))  ####1/8


        feat0, feat1 = self.encoder_layer1(feat0, feat0, pos0, pos0, mask0, mask0), \
                       self.encoder_layer1(feat1, feat1, pos1, pos1, mask1, mask1)  ### 1/8

        feat0_layer1, feat1_layer1 = self.conv1(feat0.view(bs, c, h, w)), self.conv1(feat1.view(bs, c, h, w))  ### 1/16

        # ####Ablation merge####
        # feat0_layer1, feat1_layer1 = self.merge_layer3(torch.cat((feat_f0_layer2, feat0_layer1), dim=1)), \
        #                self.merge_layer3(torch.cat((feat_f1_layer2, feat1_layer1), dim=1))  ####1/16

        pos0_layer1, pos1_layer1 = F.avg_pool2d(pos0, 2, stride=2), \
                             F.avg_pool2d(pos1, 2, stride=2)  # 2个位置信息分割成一组

        feat0_layer1, feat1_layer1 = self.encoder_layer2(feat0_layer1, feat1_layer1, pos0_layer1, pos1_layer1, x_mask=None, source_mask=None), \
                                     self.encoder_layer2(feat1_layer1, feat0_layer1, pos1_layer1, pos0_layer1, x_mask=None, source_mask=None)  ### 1/16

        feat0_layer2, feat1_layer2 = self.conv2(feat0_layer1.view(bs, c, h//2, w//2)), self.conv2(feat1_layer1.view(bs, c, h//2, w//2))  ##1/32
        pos0_layer2, pos1_layer2 = F.avg_pool2d(pos0_layer1, 2, stride=2), \
                                   F.avg_pool2d(pos1_layer1, 2, stride=2)  # 2个位置信息分割成一组

        feat0_layer2, feat1_layer2 = self.encoder_layer3(feat0_layer2, feat0_layer2, pos0_layer2, pos0_layer2, x_mask=None, source_mask=None), \
                                     self.encoder_layer3(feat1_layer2, feat1_layer2, pos1_layer2, pos1_layer2, x_mask=None, source_mask=None)
        feat0_layer2, feat1_layer2 = self.decoder_layer3(feat0_layer2, feat1_layer2, pos0_layer2, pos1_layer2, x_mask=None, source_mask=None), \
                                     self.decoder_layer3(feat1_layer2, feat0_layer2, pos1_layer2, pos0_layer2, x_mask=None, source_mask=None)

        feat0_up, feat1_up = self.up3_2(feat0_layer2), self.up3_2(feat1_layer2) ###1/16
        feat0_layer1, feat1_layer1 = self.merge_layer2(torch.cat((feat0_up, feat_f0_layer2, feat0_layer1.view(bs, c, h//2, w//2)), dim=1)), \
                                     self.merge_layer2(torch.cat((feat1_up, feat_f1_layer2, feat1_layer1.view(bs, c, h//2, w//2)), dim=1)) ####1/16

        feat0_layer1, feat1_layer1 = self.decoder_layer2(feat0_layer1, feat0_layer1, pos0_layer1, pos0_layer1, x_mask=None, source_mask=None), \
                                     self.decoder_layer2(feat1_layer1, feat1_layer1, pos1_layer1, pos1_layer1, x_mask=None, source_mask=None)  ### 1/16

        feat0_up, feat1_up = self.up2_1(feat0_layer1.view(bs, c, h//2, w//2)), self.up2_1(feat1_layer1.view(bs, c, h//2, w//2))  ###1/8
        feat0, feat1 = self.merge_layer1(torch.cat((feat0_up, feat_f0_layer1, feat0.view(bs, c, h, w)), dim=1)), \
                                     self.merge_layer1(torch.cat((feat1_up, feat_f1_layer1, feat1.view(bs, c, h, w)), dim=1))  ####1/8


        feat0, feat1 = self.decoder_layer1(feat0, feat1, pos0, pos1, mask0, mask1), \
                       self.decoder_layer1(feat1, feat0, pos1, pos0, mask1, mask0)  ### 1/8

        return feat0, feat1
