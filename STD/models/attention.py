"""
Enhanced Information Association Mechanism Modules
Includes various attention mechanisms to improve feature association
and small target detection in complex backgrounds.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EMA(nn.Module):
    """Efficient Multi-Scale Attention (EMA) Module.
    
    Reference: https://arxiv.org/abs/2205.13563
    Efficient multi-scale attention with cross-spatial learning.
    """
    
    def __init__(self, channels, factor=8):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // factor > 0
        
        self.softmax = nn.Softmax(dim=-1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(channels // factor, channels // factor, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels // factor),
            nn.SiLU(inplace=True)
        )
        
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(channels // factor, channels // factor, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels // factor),
            nn.SiLU(inplace=True)
        )
        
    def forward(self, x):
        b, c, h, w = x.shape
        group_x = x.reshape(b, self.groups, -1, h, w)
        
        # Multi-scale processing for each group
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 2, 4, 3)
        hw = h * w
        
        # Cross-spatial attention
        x_h = self.softmax(self.conv1x1(x_h.reshape(b * self.groups, -1, h, 1)).reshape(b, self.groups, -1, h))
        x_w = self.softmax(self.conv1x1(x_w.reshape(b * self.groups, -1, w, 1)).reshape(b, self.groups, -1, w))
        
        # Apply attention
        out = group_x * x_h.unsqueeze(-1) * x_w.unsqueeze(-2)
        out = out.reshape(b, c, h, w)
        
        return out


class SimAM(nn.Module):
    """Simple Attention Module (SimAM).
    
    Reference: https://proceedings.mlr.press/v139/yang21o.html
    Energy-based attention without parameter overhead.
    """
    
    def __init__(self, channels=None, e_lambda=1e-4):
        super(SimAM, self).__init__()
        self.activation = nn.Sigmoid()
        self.e_lambda = e_lambda
    
    def forward(self, x):
        b, c, h, w = x.shape
        
        n = h * w - 1
        x_minus_mean = x - x.mean(dim=[2, 3], keepdim=True)
        y = (x_minus_mean ** 2).sum(dim=[2, 3], keepdim=True) / n
        
        energy = 4 * (y + self.e_lambda) / (2 + y + self.e_lambda + 1e-6)
        return x * self.activation(energy)


class CBAM(nn.Module):
    """Convolutional Block Attention Module.
    
    Reference: https://arxiv.org/abs/1807.06521
    Combines channel attention and spatial attention.
    """
    
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channels, reduction)
        self.spatial_attention = SpatialAttentionModule()
    
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class ChannelAttentionModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)


class SpatialAttentionModule(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttentionModule, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return x * self.sigmoid(out)
