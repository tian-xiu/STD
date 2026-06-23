"""
STD-YOLO Training Script
A Small Ship Target Detection Model with Enhanced Information Association Mechanism

Usage:
    python train.py --model ultralytics/cfg/models/v8/yolov8-std-yolo.yaml \\
                    --data data/ship_dataset.yaml \\
                    --epochs 200 --batch 16 --img 640 --device 0
"""

import argparse
import os
import sys
import yaml
from pathlib import Path

import torch
from ultralytics import YOLO


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='STD-YOLO: Small Ship Target Detection Training'
    )
    
    # Model configuration
    parser.add_argument(
        '--model', type=str,
        default='ultralytics/cfg/models/v8/yolov8-std-yolo.yaml',
        help='Model configuration file path (YAML)'
    )
    parser.add_argument(
        '--pretrained', type=str, default=None,
        help='Pretrained weights path (e.g., yolov8s.pt)'
    )
    
    # Dataset
    parser.add_argument(
        '--data', type=str, default='data/ship_dataset.yaml',
        help='Dataset configuration file path (YAML)'
    )
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--img', type=int, default=640, help='Input image size (pixels)')
    parser.add_argument('--device', type=str, default='0', help='Device to use (e.g., 0, cpu)')
    parser.add_argument('--workers', type=int, default=8, help='Number of data loading workers')
    
    # Optimization
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate')
    parser.add_argument('--lrf', type=float, default=0.01, help='Final learning rate factor')
    parser.add_argument('--momentum', type=float, default=0.937, help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Optimizer weight decay')
    parser.add_argument('--warmup_epochs', type=int, default=3, help='Number of warmup epochs')
    parser.add_argument('--warmup_momentum', type=float, default=0.8, help='Warmup initial momentum')
    parser.add_argument('--warmup_bias_lr', type=float, default=0.1, help='Warmup initial bias lr')
    
    # Data augmentation
    parser.add_argument('--hsv_h', type=float, default=0.015, help='HSV-Hue augmentation')
    parser.add_argument('--hsv_s', type=float, default=0.7, help='HSV-Saturation augmentation')
    parser.add_argument('--hsv_v', type=float, default=0.4, help='HSV-Value augmentation')
    parser.add_argument('--degrees', type=float, default=0.0, help='Rotation augmentation')
    parser.add_argument('--translate', type=float, default=0.1, help='Translation augmentation')
    parser.add_argument('--scale', type=float, default=0.5, help='Scale augmentation')
    parser.add_argument('--shear', type=float, default=0.0, help='Shear augmentation')
    parser.add_argument('--perspective', type=float, default=0.0, help='Perspective augmentation')
    parser.add_argument('--flipud', type=float, default=0.0, help='Flip up-down augmentation')
    parser.add_argument('--fliplr', type=float, default=0.5, help='Flip left-right augmentation')
    parser.add_argument('--mosaic', type=float, default=1.0, help='Mosaic augmentation')
    parser.add_argument('--mixup', type=float, default=0.0, help='MixUp augmentation')
    parser.add_argument('--copy_paste', type=float, default=0.0, help='Copy-Paste augmentation')
    
    # Validation
    parser.add_argument('--val', action='store_true', default=True, help='Validate during training')
    parser.add_argument('--val_period', type=int, default=1, help='Validation frequency (epochs)')
    
    # Save and logging
    parser.add_argument('--project', type=str, default='runs/detect', help='Project name')
    parser.add_argument('--name', type=str, default='train', help='Experiment name')
    parser.add_argument('--exist_ok', action='store_true', help='Allow overwriting existing project')
    parser.add_argument('--save_dir', type=str, default=None, help='Save directory')
    parser.add_argument('--save_period', type=int, default=10, help='Save checkpoint frequency')
    
    # Resume training
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint path')
    
    # Other
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--patience', type=int, default=50, help='Early stopping patience')
    parser.add_argument('--cos_lr', action='store_true', default=True, help='Use cosine learning rate')
    parser.add_argument('--label_smoothing', type=float, default=0.0, help='Label smoothing')
    parser.add_argument('--freeze', type=int, default=0, help='Freeze first N layers')
    
    args = parser.parse_args()
    return args


def main():
    """Main training function."""
    args = parse_args()
    
    print('=' * 60)
    print('STD-YOLO: Small Ship Target Detection Training')
    print('=' * 60)
    print(f'Model config: {args.model}')
    print(f'Dataset: {args.data}')
    print(f'Image size: {args.img}')
    print(f'Batch size: {args.batch}')
    print(f'Epochs: {args.epochs}')
    print(f'Device: {args.device}')
    print('=' * 60)
    
    # Build training kwargs
    train_kwargs = {
        'data': args.data,
        'epochs': args.epochs,
        'batch': args.batch,
        'imgsz': args.img,
        'device': args.device,
        'workers': args.workers,
        'lr0': args.lr,
        'lrf': args.lrf,
        'momentum': args.momentum,
        'weight_decay': args.weight_decay,
        'warmup_epochs': args.warmup_epochs,
        'warmup_momentum': args.warmup_momentum,
        'warmup_bias_lr': args.warmup_bias_lr,
        'hsv_h': args.hsv_h,
        'hsv_s': args.hsv_s,
        'hsv_v': args.hsv_v,
        'degrees': args.degrees,
        'translate': args.translate,
        'scale': args.scale,
        'shear': args.shear,
        'perspective': args.perspective,
        'flipud': args.flipud,
        'fliplr': args.fliplr,
        'mosaic': args.mosaic,
        'mixup': args.mixup,
        'copy_paste': args.copy_paste,
        'project': args.project,
        'name': args.name,
        'exist_ok': args.exist_ok,
        'save_period': args.save_period,
        'seed': args.seed,
        'patience': args.patience,
        'cos_lr': args.cos_lr,
        'label_smoothing': args.label_smoothing,
        'freeze': args.freeze,
    }
    
    if args.resume:
        train_kwargs['resume'] = args.resume
    
    # Initialize model
    if args.pretrained:
        print(f'Loading pretrained weights: {args.pretrained}')
        model = YOLO(args.pretrained)
    else:
        print(f'Building model from config: {args.model}')
        model = YOLO(args.model)
    
    # Start training
    print('\nStarting training...')
    results = model.train(**train_kwargs)
    
    print('\nTraining completed!')
    print(f'Best model saved at: {results.save_dir}/weights/best.pt')
    
    # Run final validation
    print('\nRunning final validation...')
    val_results = model.val()
    print(f'Validation results: mAP@0.5 = {val_results.box.map50:.4f}, '
          f'mAP@0.5:0.95 = {val_results.box.map:.4f}')


if __name__ == '__main__':
    main()
