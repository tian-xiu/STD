"""
STD-YOLO: A Small Ship Target Detection Model with Enhanced Information Association Mechanism
Main model definition integrating all custom modules.

Based on YOLOv8 architecture with:
1. P2 small object detection layer
2. CARAFE upsampling operator
3. Enhanced SPPF module
4. Attention-based feature association
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.carafe import CARAFE_UP
from models.sppf_custom import SPPFCustom
from models.attention import EMA, SimAM, CBAM
from models.small_object_head import P2DetectionLayer


class Conv(nn.Module):
    """Standard convolution block with BatchNorm and SiLU activation."""
    
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
    
    def forward_fuse(self, x):
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    """Standard bottleneck block."""
    
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k=1, s=1)
        self.cv2 = Conv(c_, c2, k=3, s=1, g=g)
        self.add = shortcut and c1 == c2
    
    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """C2f (Cross Stage Partial with 2 convolutions and f bottlenecks)."""
    
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super(C2f, self).__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, k=1, s=1)
        self.cv2 = Conv((2 + n) * self.c, c2, k=1, s=1)
        self.m = nn.ModuleList([
            Bottleneck(self.c, self.c, shortcut, g, e=1.0) for _ in range(n)
        ])
    
    def forward(self, x):
        y = list(self.cv1(x).chunk(2, dim=1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, dim=1))


class Concat(nn.Module):
    """Concatenate tensors along a specified dimension."""
    
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension
    
    def forward(self, x):
        return torch.cat(x, self.d)


class Detect(nn.Module):
    """YOLOv8 Detection Head."""
    
    def __init__(self, nc=1, ch=()):
        super(Detect, self).__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = 16
        
        self.no = nc + self.reg_max * 4
        self.stride = torch.zeros(self.nl)
        
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], self.nc)
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c2, k=3),
                Conv(c2, c2, k=3),
                nn.Conv2d(c2, 4 * self.reg_max, k=1)
            ) for x in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c3, k=3),
                Conv(c3, c3, k=3),
                nn.Conv2d(c3, self.nc, k=1)
            ) for x in ch
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()
    
    def forward(self, x):
        shape = x[0].shape
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), dim=1)
        
        if self.training:
            return x
        
        # Inference path
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], dim=2)
        box, cls = x_cat.split((self.reg_max * 4, self.nc), dim=1)
        dbox = self.dfl(box)
        return dbox, cls.sigmoid()


class DFL(nn.Module):
    """Distribution Focal Loss module."""
    
    def __init__(self, c1=16):
        super(DFL, self).__init__()
        self.conv = nn.Conv2d(c1, 1, kernel_size=1, bias=False)
        self.c1 = c1
    
    def forward(self, x):
        b, c, a = x.shape
        x = x.view(b, 4, self.c1, a).softmax(2)
        return self.conv(x).view(b, 4, a)


def autopad(k, p=None, d=1):
    """Calculate automatic padding."""
    if d > 1:
        k = d * (k - 1) + 1
    if p is None:
        p = k // 2
    return p


class STDYOLO(nn.Module):
    """STD-YOLO: Main model class.
    
    Integrates all improvements:
    - P2 small object detection head
    - CARAFE upsampling
    - Enhanced SPPF
    - Attention mechanisms
    """
    
    def __init__(self, cfg='yolov8-std-yolo.yaml', nc=1):
        super(STDYOLO, self).__init__()
        self.nc = nc
        self.cfg = cfg
        self.model, self.save = self._build_model()
        self.stride = torch.tensor([8, 16, 32, 64])  # P2-P5 strides
    
    def _build_model(self):
        """Build STD-YOLO model architecture."""
        
        # Backbone
        backbone = nn.ModuleList([
            Conv(3, 16, k=3, s=2),           # 0: P1/2  - 320x320
            Conv(16, 32, k=3, s=2),           # 1: P2/4  - 160x160
            C2f(32, 64, n=3, shortcut=True),  # 2
            Conv(64, 64, k=3, s=2),           # 3: P3/8  - 80x80
            C2f(64, 128, n=6, shortcut=True), # 4
            Conv(128, 128, k=3, s=2),         # 5: P4/16 - 40x40
            C2f(128, 256, n=6, shortcut=True),# 6
            Conv(256, 256, k=3, s=2),         # 7: P5/32 - 20x20
            C2f(256, 512, n=3, shortcut=True),# 8
            SPPFCustom(512, 512, k=5, n=4),   # 9: Enhanced SPPF
        ])
        
        # Head (Neck + Detection)
        # Use CARAFE instead of bilinear upsampling
        head = nn.ModuleList([
            # Top-down FPN with CARAFE
            CARAFE_UP(512, 256),              # 10: Upsample P5 -> P4
            Concat(),                          # 11: Concat with P4
            C2f(512, 256, n=3, shortcut=False),# 12: Neck P4
            CARAFE_UP(256, 128),              # 13: Upsample P4 -> P3
            Concat(),                          # 14: Concat with P3
            C2f(256, 128, n=3, shortcut=False),# 15: Neck P3
            CARAFE_UP(128, 64),               # 16: Upsample P3 -> P2 (for small objects)
            Concat(),                          # 17: Concat with P2
            C2f(128, 64, n=3, shortcut=False), # 18: Neck P2 (small object branch)
            
            # Bottom-up path
            Conv(64, 64, k=3, s=2),           # 19: Downsample P2 -> P3
            Concat(),                          # 20: Concat with P3 neck
            C2f(192, 128, n=3, shortcut=False),# 21: Neck P3
            Conv(128, 128, k=3, s=2),          # 22: Downsample P3 -> P4
            Concat(),                          # 23: Concat with P4 neck
            C2f(384, 256, n=3, shortcut=False),# 24: Neck P4
            Conv(256, 256, k=3, s=2),          # 25: Downsample P4 -> P5
            Concat(),                          # 26: Concat with P5
            C2f(768, 512, n=3, shortcut=False),# 27: Neck P5
        ])
        
        # Detection layers (P2, P3, P4, P5)
        detect = Detect(self.nc, ch=(64, 128, 256, 512))
        
        return nn.ModuleList([backbone, head, detect]), [2, 4, 6, 8, 12, 15, 18, 21, 24, 27]
    
    def forward(self, x):
        """Forward pass through STD-YOLO."""
        backbone, head, detect = self.model
        
        # Store intermediate features for concatenation
        features = {}
        
        # Backbone forward
        for i, layer in enumerate(backbone):
            x = layer(x)
            if i in [2, 4, 6, 8]:  # P2, P3, P4, P5
                features[f'P{i//2+2}'] = x
        
        # Neck forward - need to track indexes for concat
        p5 = features['P5']  # After SPPF
        p4 = features['P4']
        p3 = features['P3']
        p2 = features['P2']
        
        # Top-down
        x = head[0](p5)      # CARAFE upsample P5 -> P4 size
        x = head[1]([x, p4]) # Concat
        x = head[2](x)       # C2f
        neck_p4 = x
        
        x = head[3](x)       # CARAFE upsample P4 -> P3 size
        x = head[4]([x, p3]) # Concat
        x = head[5](x)       # C2f
        neck_p3 = x
        
        x = head[6](x)       # CARAFE upsample P3 -> P2 size
        x = head[7]([x, p2]) # Concat
        x = head[8](x)       # C2f
        neck_p2 = x
        
        # Bottom-up
        x = head[9](neck_p2) # Downsample
        x = head[10]([x, neck_p3]) # Concat
        x = head[11](x)      # C2f
        neck_p3_up = x
        
        x = head[12](neck_p3_up) # Downsample
        x = head[13]([x, neck_p4]) # Concat
        x = head[14](x)      # C2f
        neck_p4_up = x
        
        x = head[15](neck_p4_up) # Downsample
        x = head[16]([x, p5]) # Concat with backbone P5
        x = head[17](x)      # C2f
        neck_p5 = x
        
        # Detection heads
        det_out = detect([neck_p2, neck_p3_up, neck_p4_up, neck_p5])
        return det_out
    
    def _apply(self, fn):
        """Override _apply for proper device transfer."""
        return super()._apply(fn)
