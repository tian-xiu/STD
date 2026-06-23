"""
STD-YOLO Inference / Prediction Script
Detect small ship targets in remote sensing images

Usage:
    python predict.py --weights runs/detect/train/weights/best.pt \\
                      --source path/to/test/images/ \\
                      --img 640 --device 0 --conf 0.25
"""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import torch
import numpy as np
from ultralytics import YOLO


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='STD-YOLO: Small Ship Target Detection Inference'
    )
    
    parser.add_argument('--weights', type=str, required=True,
                        help='Model weights path (e.g., runs/detect/train/weights/best.pt)')
    parser.add_argument('--source', type=str, required=True,
                        help='Source for inference (image path, directory, video, or 0 for webcam)')
    parser.add_argument('--img', type=int, default=640,
                        help='Input image size (pixels)')
    parser.add_argument('--device', type=str, default='0',
                        help='Device to use (e.g., 0, 1, cpu)')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='NMS IoU threshold')
    parser.add_argument('--max_det', type=int, default=300,
                        help='Maximum number of detections per image')
    parser.add_argument('--half', action='store_true',
                        help='Use half precision (FP16)')
    parser.add_argument('--project', type=str, default='runs/detect',
                        help='Save results to project/name')
    parser.add_argument('--name', type=str, default='predict',
                        help='Experiment name')
    parser.add_argument('--exist_ok', action='store_true',
                        help='Allow overwriting existing project')
    parser.add_argument('--save_txt', action='store_true', default=True,
                        help='Save results as txt labels')
    parser.add_argument('--save_img', action='store_true', default=True,
                        help='Save results as images')
    parser.add_argument('--save_conf', action='store_true', default=True,
                        help='Save confidence scores in label files')
    parser.add_argument('--save_crop', action='store_true',
                        help='Save cropped detection boxes')
    parser.add_argument('--show', action='store_true',
                        help='Display results in real-time')
    parser.add_argument('--vid_stride', type=int, default=1,
                        help='Video frame stride')
    parser.add_argument('--line_width', type=int, default=2,
                        help='Bounding box line width')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize model features')
    parser.add_argument('--augment', action='store_true',
                        help='Use test-time augmentation (TTA)')
    parser.add_argument('--agnostic_nms', action='store_true',
                        help='Class-agnostic NMS')
    parser.add_argument('--classes', type=int, nargs='+', default=None,
                        help='Filter by class (e.g., --classes 0)')
    parser.add_argument('--retina_masks', action='store_true',
                        help='Use retina masks for segmentation')
    
    args = parser.parse_args()
    return args


def main():
    """Main inference function."""
    args = parse_args()
    
    print('=' * 60)
    print('STD-YOLO: Small Ship Target Detection Inference')
    print('=' * 60)
    print(f'Weights:       {args.weights}')
    print(f'Source:        {args.source}')
    print(f'Image size:    {args.img}')
    print(f'Device:        {args.device}')
    print(f'Confidence:    {args.conf}')
    print(f'IoU threshold: {args.iou}')
    print('=' * 60)
    
    # Load model
    print('\nLoading model...')
    model = YOLO(args.weights)
    
    # Run inference
    print('\nRunning inference...')
    start_time = time.time()
    
    results = model.predict(
        source=args.source,
        imgsz=args.img,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        half=args.half,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        save_crop=args.save_crop,
        show=args.show,
        vid_stride=args.vid_stride,
        line_width=args.line_width,
        visualize=args.visualize,
        augment=args.augment,
        agnostic_nms=args.agnostic_nms,
        classes=args.classes,
        retina_masks=args.retina_masks,
    )
    
    elapsed = time.time() - start_time
    
    # Display results summary
    num_images = len(results)
    print(f'\nInference completed in {elapsed:.2f}s')
    print(f'Processed {num_images} image(s)')
    
    if num_images > 0:
        total_detections = sum(len(r.boxes) for r in results if r.boxes is not None)
        print(f'Total detections: {total_detections}')
        
        # Per-image statistics
        for i, result in enumerate(results):
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes
                confs = boxes.conf.cpu().numpy()
                cls_ids = boxes.cls.cpu().numpy().astype(int)
                
                print(f'\nImage {i+1}: {result.path}')
                print(f'  Detections: {len(boxes)}')
                for j in range(min(len(boxes), 5)):  # Show first 5 detections
                    cls_name = result.names[cls_ids[j]] if cls_ids[j] in result.names else f'class_{cls_ids[j]}'
                    print(f'    [{j+1}] {cls_name}: {confs[j]:.4f}')
                if len(boxes) > 5:
                    print(f'    ... and {len(boxes) - 5} more')
            else:
                print(f'\nImage {i+1}: {result.path} - No detections')
    
    # Save path
    save_dir = Path(args.project) / args.name
    if save_dir.exists():
        print(f'\nResults saved to: {save_dir}')
    
    return results


if __name__ == '__main__':
    main()
