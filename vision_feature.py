import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from nystrom_attention import NystromAttention
from diff_attention import DiffAttention
from einops import rearrange, repeat

def fourier_encode(x, max_freq, num_bands):

    device, dtype, orig_x = x.device, x.dtype, x

    scales = torch.linspace(1., max_freq / 2, num_bands, device=device, dtype=dtype)
    scales = scales[(None,) * (len(x.shape) - 1) + (Ellipsis,)]

    x = x * scales * torch.pi
    x = torch.cat([x.sin(), x.cos()], dim=-1)
    x = torch.cat((x, orig_x), dim=-1)
    return x

class GateASPP(nn.Module):
    def __init__(self, dim_in, dim_out, group_size, H, W, rate=1, bn_mom=0.1):
        super(GateASPP, self).__init__()
        self.group_size = group_size
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=6 * rate, dilation=6 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=12 * rate, dilation=12 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=18 * rate, dilation=18 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim_out * 4, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.alpha = nn.Parameter(torch.ones(4, requires_grad=True))

    def forward(self, x):
        alpha = torch.sigmoid(self.alpha)
        [b, c, row, col] = x.size()
        conv1x1 = self.branch1(x) * alpha[0]
        conv3x3_1 = self.branch2(x) * alpha[1]
        conv3x3_2 = self.branch3(x) * alpha[2]
        conv3x3_3 = self.branch4(x) * alpha[3]

        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3], dim=1)
        result = self.conv_cat(feature_cat)
        return result


class ASPP(nn.Module):
    def __init__(self, dim_in, dim_out, group_size, H, W, rate=1, bn_mom=0.1):
        super(ASPP, self).__init__()
        self.group_size = group_size
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=6 * rate, dilation=6 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=12 * rate, dilation=12 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=18 * rate, dilation=18 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim_out * 4, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):

        [b, c, row, col] = x.size()
        conv1x1 = self.branch1(x) 
        conv3x3_1 = self.branch2(x) 
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)

        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3], dim=1)
        result = self.conv_cat(feature_cat)
        return result


class SpatialAttention(nn.Module):
    def __init__(self, ):
        super(SpatialAttention, self).__init__()

    def forward(self, x):
        ####################################
        # The code is being organized
        ####################################


        return x

class VisionText(nn.Module):
    def __init__(self, vision_dim,  vision_embed_dim,text_dim, text_embed_dim, out_dim = 128, head = 8, group_size = 2,vision_dropout = 0.2):
        super(VisionText, self).__init__()
        ####################################
        # The code is being organized
        ####################################

    def init_position_embedding(self, x):
        b, c, h, w = x.shape
        device, dtype = x.device, x.dtype
        x = rearrange(x, 'b c h w -> b h w c')
        axis_pos = [
            torch.linspace(-1., 1., steps=h, device=device, dtype=dtype),
            torch.linspace(-1., 1., steps=w, device=device, dtype=dtype)
        ]
        pos = torch.stack(torch.meshgrid(*axis_pos, indexing='ij'), dim=-1)
        enc_pos = fourier_encode(pos, max_freq=10, num_bands=1)
        enc_pos = repeat(enc_pos, 'h w d -> b h w d', b=b)
        return enc_pos


    def forward(self, x, y):
        ####################################
        # The code is being organized
        ####################################
        


class Vision(nn.Module):
    def __init__(self, vision_dim,  vision_embed_dim, out_dim = 128, head = 8, group_size = 2,vision_dropout = 0.2):
        super(Vision, self).__init__()
        ####################################
        # The code is being organized
        ####################################

    def init_position_embedding(self, x):
        b, c, h, w = x.shape
        device, dtype = x.device, x.dtype
        x = rearrange(x, 'b c h w -> b h w c')
        axis_pos = [
            torch.linspace(-1., 1., steps=h, device=device, dtype=dtype),
            torch.linspace(-1., 1., steps=w, device=device, dtype=dtype)
        ]
        pos = torch.stack(torch.meshgrid(*axis_pos, indexing='ij'), dim=-1)
        enc_pos = fourier_encode(pos, max_freq=10, num_bands=1)
        enc_pos = repeat(enc_pos, 'h w d -> b h w d', b=b)
        return enc_pos


    def forward(self, x):
        ####################################
        # The code is being organized
        ####################################

        return vision_feature
