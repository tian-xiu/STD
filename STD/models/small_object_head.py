"""
Small Object Detection Head for STD-YOLO.
Adds a dedicated P2 detection layer (160x160 feature map) 
specifically designed for small ship target detection.
"""

import torch
import torch.nn as nn


class SmallObjectHead(nn.Module):
    """Small Object Detection Head Module.
    
    Adds a P2 detection head operating on high-resolution feature maps
    (160x160 for 640 input) to improve small target detection.
    """
    
    def __init__(self, in_channels=128, num_classes=1):
        super(SmallObjectHead, self).__init__()
        self.num_classes = num_classes
        
        # Feature refinement for small objects
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True)
        )
        
        # Detection outputs
        self.cls_pred = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        self.reg_pred = nn.Conv2d(in_channels, 4, kernel_size=1)  # bbox regression
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        
        cls_out = self.cls_pred(x)
        reg_out = self.reg_pred(x)
        
        return cls_out, reg_out


class P2DetectionLayer(nn.Module):
    """P2 Detection Layer for small targets.
    
    This module processes high-resolution feature maps from early
    backbone stages to detect small objects that would otherwise
    be lost in deeper layers.
    """
    
    def __init__(self, c1, c2, num_classes=1):
        """
        Args:
            c1: Input channels from backbone
            c2: Output channels for detection
            num_classes: Number of target classes
        """
        super(P2DetectionLayer, self).__init__()
        
        # Feature alignment
        self.align = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
            nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )
        
        # Small object detection head
        self.small_head = SmallObjectHead(c2, num_classes)
    
    def forward(self, x):
        x = self.align(x)
        cls_out, reg_out = self.small_head(x)
        return cls_out, reg_out
