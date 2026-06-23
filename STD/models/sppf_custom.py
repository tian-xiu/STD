"""
Modified SPPF (Spatial Pyramid Pooling - Fast) module for STD-YOLO.
Enhanced with additional pooling scales and channel attention
to better capture multi-scale features of small ship targets.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """Channel Attention Module for emphasizing informative channels."""
    
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
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
        return self.sigmoid(out) * x


class SPPFCustom(nn.Module):
    """Modified SPPF with enhanced multi-scale pooling.
    
    Features:
    - Multiple pooling scales for better multi-scale feature extraction
    - Channel attention for feature refinement
    - Improved for small target detection
    """
    
    def __init__(self, c1, c2, k=5, n=4):
        """
        Args:
            c1: Input channels
            c2: Output channels
            k: Maxpool kernel size
            n: Number of pooling layers (increased from default 3 for more scales)
        """
        super(SPPFCustom, self).__init__()
        c_ = c1 // 2  # hidden channels
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(c1, c_, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c_),
            nn.SiLU(inplace=True)
        )
        
        # Multi-scale maxpooling layers
        self.m = nn.ModuleList([
            nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
            for _ in range(n)
        ])
        
        # Channel attention for feature refinement
        self.ca = ChannelAttention(c_ * (n + 1))
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(c_ * (n + 1), c2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        
        # Multi-scale pooling
        features = [x]
        for pooling in self.m:
            features.append(pooling(features[-1]))
        
        # Concatenate multi-scale features
        x = torch.cat(features, dim=1)
        
        # Apply channel attention
        x = self.ca(x)
        
        x = self.conv2(x)
        return x
