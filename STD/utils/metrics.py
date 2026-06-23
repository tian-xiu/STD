"""
Utility functions for STD-YOLO evaluation and post-processing.
"""

import torch
import torchvision
import numpy as np


def compute_iou(box1, box2):
    """Compute IoU between two sets of boxes.
    
    Args:
        box1: Tensor of shape (N, 4) [x1, y1, x2, y2]
        box2: Tensor of shape (M, 4) [x1, y1, x2, y2]
    
    Returns:
        iou: Tensor of shape (N, M)
    """
    # Expand dimensions for broadcasting
    box1 = box1.unsqueeze(1)  # (N, 1, 4)
    box2 = box2.unsqueeze(0)  # (1, M, 4)
    
    # Get intersection coordinates
    inter_x1 = torch.max(box1[..., 0], box2[..., 0])
    inter_y1 = torch.max(box1[..., 1], box2[..., 1])
    inter_x2 = torch.min(box1[..., 2], box2[..., 2])
    inter_y2 = torch.min(box1[..., 3], box2[..., 3])
    
    # Intersection area
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    
    # Union area
    area1 = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
    area2 = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])
    union_area = area1 + area2 - inter_area
    
    # IoU
    iou = inter_area / (union_area + 1e-6)
    
    return iou


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, max_det=300):
    """Non-Maximum Suppression.
    
    Args:
        prediction: Tensor of shape (batch, num_boxes, 6) [x1, y1, x2, y2, conf, cls]
        conf_thres: Confidence threshold
        iou_thres: IoU threshold for NMS
        max_det: Maximum number of detections per image
    
    Returns:
        List of tensors, one per image, each of shape (num_detections, 6)
    """
    if isinstance(prediction, list):
        prediction = prediction[0] if len(prediction) == 1 else torch.cat(prediction, dim=1)
    
    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates
    
    output = [torch.zeros((0, 6), device=prediction.device)] * bs
    
    for xi, x in enumerate(prediction):
        x = x[xc[xi]]  # filter by confidence
        
        if not x.shape[0]:
            continue
        
        # Compute class scores
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
        box = x[:, :4]
        conf, j = x[:, 5:].max(dim=1, keepdim=True)
        
        x = torch.cat((box, conf, j.float()), dim=1)[conf.view(-1) > conf_thres]
        
        if not x.shape[0]:
            continue
        
        # Sort by confidence
        x = x[x[:, 4].argsort(descending=True)[:max_det]]
        
        # NMS
        boxes = x[:, :4]
        scores = x[:, 4]
        
        i = torchvision.ops.nms(boxes, scores, iou_thres)
        i = i[:max_det]
        
        output[xi] = x[i]
    
    return output
