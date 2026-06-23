"""
STD-YOLO Validation Script
Evaluates model performance (mAP@0.5, mAP@0.5:0.95, FPS, Precision, Recall)

Usage:
    python val.py --weights runs/detect/train/weights/best.pt \\
                  --data data/ship_dataset.yaml \\
                  --img 640 --device 0 --batch 16
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import numpy as np
from ultralytics import YOLO


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='STD-YOLO: Model Validation'
    )
    
    parser.add_argument('--weights', type=str, required=True,
                        help='Model weights path (e.g., runs/detect/train/weights/best.pt)')
    parser.add_argument('--data', type=str, default='data/ship_dataset.yaml',
                        help='Dataset configuration file path')
    parser.add_argument('--img', type=int, default=640,
                        help='Input image size (pixels)')
    parser.add_argument('--batch', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--device', type=str, default='0',
                        help='Device to use (e.g., 0, 1, cpu)')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of data loading workers')
    parser.add_argument('--conf', type=float, default=0.001,
                        help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.6,
                        help='NMS IoU threshold')
    parser.add_argument('--half', action='store_true',
                        help='Use half precision (FP16)')
    parser.add_argument('--project', type=str, default='runs/detect',
                        help='Project name')
    parser.add_argument('--name', type=str, default='val',
                        help='Experiment name')
    parser.add_argument('--exist_ok', action='store_true',
                        help='Allow overwriting existing project')
    parser.add_argument('--plots', action='store_true', default=True,
                        help='Generate validation plots')
    parser.add_argument('--save_json', action='store_true',
                        help='Save results to JSON')
    parser.add_argument('--save_hybrid', action='store_true',
                        help='Save hybrid labels')
    parser.add_argument('--max_det', type=int, default=300,
                        help='Maximum number of detections per image')
    
    args = parser.parse_args()
    return args


def benchmark_fps(model, img_size=640, num_warmup=50, num_iter=200, device='0'):
    """Benchmark model inference speed (FPS)."""
    print(f'\nBenchmarking FPS (image size: {img_size}x{img_size})...')
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, img_size, img_size).to(device)
    
    # Warmup
    for _ in range(num_warmup):
        _ = model(dummy_input, verbose=False)
    
    # Benchmark
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(num_iter):
        _ = model(dummy_input, verbose=False)
    torch.cuda.synchronize()
    
    elapsed = time.time() - start_time
    fps = num_iter / elapsed
    
    print(f'FPS: {fps:.2f} (avg over {num_iter} iterations)')
    print(f'Average inference time: {elapsed / num_iter * 1000:.2f} ms')
    
    return fps


def main():
    """Main validation function."""
    args = parse_args()
    
    print('=' * 60)
    print('STD-YOLO: Model Validation')
    print('=' * 60)
    print(f'Weights: {args.weights}')
    print(f'Dataset: {args.data}')
    print(f'Image size: {args.img}')
    print(f'Batch size: {args.batch}')
    print(f'Device: {args.device}')
    print(f'Confidence: {args.conf}')
    print(f'IoU threshold: {args.iou}')
    print('=' * 60)
    
    # Load model
    print('\nLoading model...')
    model = YOLO(args.weights)
    
    # Move to device
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() and args.device != 'cpu' else 'cpu')
    print(f'Using device: {device}')
    
    # Run validation
    print('\nRunning validation...')
    val_kwargs = {
        'data': args.data,
        'imgsz': args.img,
        'batch': args.batch,
        'device': args.device,
        'workers': args.workers,
        'conf': args.conf,
        'iou': args.iou,
        'half': args.half,
        'project': args.project,
        'name': args.name,
        'exist_ok': args.exist_ok,
        'plots': args.plots,
        'save_json': args.save_json,
        'save_hybrid': args.save_hybrid,
        'max_det': args.max_det,
    }
    
    results = model.val(**val_kwargs)
    
    # Display results
    print('\n' + '=' * 60)
    print('Validation Results')
    print('=' * 60)
    
    if hasattr(results, 'box') and results.box is not None:
        print(f'Precision:     {results.box.p:.4f}')
        print(f'Recall:        {results.box.r:.4f}')
        print(f'mAP@0.5:       {results.box.map50:.4f}')
        print(f'mAP@0.5:0.95:  {results.box.map:.4f}')
        
        # Per-class results
        if hasattr(results.box, 'ap_class_index') and hasattr(results.box, 'ap'):
            print('\nPer-class AP:')
            for i, (cls_idx, ap) in enumerate(zip(results.box.ap_class_index, results.box.ap)):
                print(f'  Class {cls_idx}: AP@0.5:0.95 = {ap:.4f}')
    
    # Benchmark FPS
    if torch.cuda.is_available() and args.device != 'cpu':
        fps = benchmark_fps(model.model, args.img, device=device)
    else:
        fps = 0.0
        print('\nFPS benchmark skipped (requires CUDA)')
    
    print('\n' + '=' * 60)
    print('Summary')
    print('=' * 60)
    print(f'Weights:         {args.weights}')
    print(f'mAP@0.5:         {results.box.map50:.4f}')
    print(f'mAP@0.5:0.95:    {results.box.map:.4f}')
    if fps > 0:
        print(f'FPS:             {fps:.2f}')
    print(f'Results saved to: {results.save_dir}')
    
    return results


if __name__ == '__main__':
    main()
