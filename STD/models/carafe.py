"""
CARAFE: Content-Aware ReAssembly of FEatures
Reference: https://arxiv.org/abs/1905.02188

CARAFE is a lightweight, general, and end-to-end upsampling operator
that reassembles features in a content-aware manner, which helps
improve feature representation especially for small targets.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CARAFE(nn.Module):
    """CARAFE Upsampling Module.
    
    Args:
        in_channels: Number of input channels
        up_factor: Upsampling factor
        kernel_size: Kernel size for reassembly
        group_size: Group size for channel compression
        scale_factor: Scale factor for the kernel prediction module
    """
    
    def __init__(self, in_channels, up_factor=2, kernel_size=5, group_size=1, scale_factor=2):
        super(CARAFE, self).__init__()
        self.kernel_size = kernel_size
        self.up_factor = up_factor
        self.group_size = group_size
        self.scale_factor = scale_factor
        
        # Channel compression ratio
        self.compressed_channels = max(in_channels // 4, 4)
        
        # 1x1 Conv for channel compression
        self.channel_compressor = nn.Conv2d(
            in_channels, self.compressed_channels, kernel_size=1, stride=1, padding=0
        )
        self.channel_compressor.apply(self._init_weights)
        
        # Kernel prediction module
        self.kernel_predictor = nn.Conv2d(
            self.compressed_channels,
            (self.kernel_size * self.group_size) * (self.up_factor ** 2),
            kernel_size=1, stride=1, padding=0
        )
        self.kernel_predictor.apply(self._init_weights)
        
        # Feature reassembly module
        self.reassembly = nn.Conv2d(
            self.compressed_channels,
            self.compressed_channels * (self.up_factor ** 2),
            kernel_size=1, stride=1, padding=0
        )
        self.reassembly.apply(self._init_weights)
        
        # Output projection
        self.out_proj = nn.Conv2d(
            self.compressed_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.out_proj.apply(self._init_weights)
        
        self.pixel_shuffle = nn.PixelShuffle(self.up_factor)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        batch_size, channels, height, width = x.shape
        
        # Channel compression
        compressed_feat = self.channel_compressor(x)
        
        # Kernel prediction
        kernel_raw = self.kernel_predictor(compressed_feat)
        
        # Reshape kernel for reassembly
        kernel = kernel_raw.view(
            batch_size, self.group_size, 
            self.kernel_size * self.kernel_size,
            self.up_factor ** 2, height, width
        )
        kernel = kernel.permute(0, 1, 3, 4, 5, 2).contiguous()
        kernel = F.softmax(kernel, dim=-1)
        
        # Feature reassembly
        reassembly_feat = self.reassembly(compressed_feat)
        reassembly_feat = self.pixel_shuffle(reassembly_feat)
        
        # Output projection
        out = self.out_proj(reassembly_feat)
        
        return out


class CARAFE_UP(nn.Module):
    """CARAFE upsampling layer compatible with YOLO architecture."""
    
    def __init__(self, c1, c2, up_factor=2, kernel_size=5):
        super(CARAFE_UP, self).__init__()
        self.c1 = c1
        self.c2 = c2
        self.up_factor = up_factor
        
        # Reduce channels if needed
        if c1 != c2:
            self.reduce = nn.Conv2d(c1, c2, kernel_size=1, stride=1, padding=0)
        else:
            self.reduce = nn.Identity()
        
        # CARAFE upsampling
        self.carafe = CARAFE(c2, up_factor=up_factor, kernel_size=kernel_size)
    
    def forward(self, x):
        x = self.reduce(x)
        x = self.carafe(x)
        return x
