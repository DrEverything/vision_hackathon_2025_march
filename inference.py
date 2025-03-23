import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
import torch
from torch.utils.data import DataLoader
from skimage import morphology

from models import get_unet_efficientnet, get_fpn, get_deeplabv3plus
from dataset import TestDataset, get_test_loader
from utils import mask2rle, set_seed

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Generate predictions for test data')
    
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model')
    parser.add_argument('--model_type', type=str, default='unet', 
                        choices=['unet', 'fpn', 'deeplabv3plus'],
                        help='Model architecture used for training')
    parser.add_argument('--encoder', type=str, default='efficientnet-b4',
                        help='Encoder used for training')
    parser.add_argument('--test_dir', type=str, default='test/test/images',
                        help='Directory containing test images')
    parser.add_argument('--output', type=str, default='submission.csv',
                        help='Output path for submission file')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for inference')
    parser.add_argument('--img_size', type=int, default=256,
                        help='Size of input images')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for binary segmentation')
    parser.add_argument('--tta', action='store_true',
                        help='Use test-time augmentation')
    parser.add_argument('--advanced_tta', action='store_true',
                        help='Use advanced test-time augmentation with more transforms')
    parser.add_argument('--postprocess', action='store_true',
                        help='Apply post-processing to predictions')
    parser.add_argument('--min_size', type=int, default=50,
                        help='Minimum size for connected components in post-processing')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    return parser.parse_args()

def apply_tta(model, image, threshold=0.5, device='cuda'):
    """
    Apply test-time augmentation
    
    Args:
        model: trained model
        image: input image (PyTorch tensor)
        threshold: threshold for binary segmentation
        device: device to use for inference
        
    Returns:
        Binary mask after TTA
    """
    # Original prediction
    image = image.to(device)
    pred_orig = model(image.unsqueeze(0))
    pred_orig = torch.sigmoid(pred_orig).cpu().numpy()[0, 0]
    
    # Flip horizontally
    img_h_flip = torch.flip(image, dims=[2])
    img_h_flip = img_h_flip.to(device)
    pred_h_flip = model(img_h_flip.unsqueeze(0))
    pred_h_flip = torch.sigmoid(pred_h_flip).cpu().numpy()[0, 0]
    pred_h_flip = np.flip(pred_h_flip, axis=1)
    
    # Flip vertically
    img_v_flip = torch.flip(image, dims=[1])
    img_v_flip = img_v_flip.to(device)
    pred_v_flip = model(img_v_flip.unsqueeze(0))
    pred_v_flip = torch.sigmoid(pred_v_flip).cpu().numpy()[0, 0]
    pred_v_flip = np.flip(pred_v_flip, axis=0)
    
    # Average predictions
    pred_mean = (pred_orig + pred_h_flip + pred_v_flip) / 3.0
    
    # Apply threshold
    return (pred_mean > threshold).astype(np.uint8)

def apply_advanced_tta(model, image, threshold=0.5, device='cuda'):
    """
    Apply advanced test-time augmentation with more transforms
    
    Args:
        model: trained model
        image: input image (PyTorch tensor)
        threshold: threshold for binary segmentation
        device: device to use for inference
        
    Returns:
        Binary mask after advanced TTA
    """
    # Move image to device
    image = image.to(device)
    
    # List to store all predictions
    all_preds = []
    
    # Original prediction
    pred_orig = torch.sigmoid(model(image.unsqueeze(0))).cpu().numpy()[0, 0]
    all_preds.append(pred_orig)
    
    # Horizontal flip
    img_h_flip = torch.flip(image, dims=[2])
    pred_h_flip = torch.sigmoid(model(img_h_flip.unsqueeze(0).to(device))).cpu().numpy()[0, 0]
    pred_h_flip = np.flip(pred_h_flip, axis=1)
    all_preds.append(pred_h_flip)
    
    # Vertical flip
    img_v_flip = torch.flip(image, dims=[1])
    pred_v_flip = torch.sigmoid(model(img_v_flip.unsqueeze(0).to(device))).cpu().numpy()[0, 0]
    pred_v_flip = np.flip(pred_v_flip, axis=0)
    all_preds.append(pred_v_flip)
    
    # 90 degree rotation
    img_rot90 = torch.rot90(image, k=1, dims=[1, 2])
    pred_rot90 = torch.sigmoid(model(img_rot90.unsqueeze(0).to(device))).cpu().numpy()[0, 0]
    pred_rot90 = np.rot90(pred_rot90, k=3)
    all_preds.append(pred_rot90)
    
    # 180 degree rotation
    img_rot180 = torch.rot90(image, k=2, dims=[1, 2])
    pred_rot180 = torch.sigmoid(model(img_rot180.unsqueeze(0).to(device))).cpu().numpy()[0, 0]
    pred_rot180 = np.rot90(pred_rot180, k=2)
    all_preds.append(pred_rot180)
    
    # 270 degree rotation
    img_rot270 = torch.rot90(image, k=3, dims=[1, 2])
    pred_rot270 = torch.sigmoid(model(img_rot270.unsqueeze(0).to(device))).cpu().numpy()[0, 0]
    pred_rot270 = np.rot90(pred_rot270, k=1)
    all_preds.append(pred_rot270)
    
    # Average all predictions
    pred_mean = np.mean(all_preds, axis=0)
    
    # Apply threshold
    return (pred_mean > threshold).astype(np.uint8)

def postprocess_prediction(pred_mask, min_size=50):
    """
    Apply post-processing to clean up the prediction
    
    Args:
        pred_mask: binary prediction mask
        min_size: minimum size for connected components
        
    Returns:
        Cleaned binary mask
    """
    # Remove small objects from foreground
    cleaned_foreground = morphology.remove_small_objects(
        pred_mask.astype(bool), 
        min_size=min_size
    )
    
    # Remove small holes from background
    cleaned_mask = morphology.remove_small_holes(
        cleaned_foreground, 
        area_threshold=min_size
    )
    
    return cleaned_mask.astype(np.uint8)

def main():
    """Main inference function"""
    # Parse arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    set_seed(args.random_seed)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Get test image IDs
    test_img_dir = args.test_dir
    test_img_ids = [os.path.splitext(f)[0] for f in os.listdir(test_img_dir) if f.endswith('.png')]
    print(f'Found {len(test_img_ids)} test images')
    
    # Create model
    print(f'Creating {args.model_type} model with {args.encoder} encoder')
    if args.model_type == 'unet':
        model = get_unet_efficientnet(input_channels=3, encoder=args.encoder)
    elif args.model_type == 'fpn':
        model = get_fpn(input_channels=3, encoder=args.encoder)
    elif args.model_type == 'deeplabv3plus':
        model = get_deeplabv3plus(input_channels=3, encoder=args.encoder)
    
    # Load model weights
    print(f'Loading model weights from {args.model_path}')
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Create test data loader
    test_loader = get_test_loader(
        data_dir=test_img_dir,
        test_ids=test_img_ids,
        batch_size=args.batch_size if not (args.tta or args.advanced_tta) else 1,  # Batch size 1 for TTA
        img_size=(args.img_size, args.img_size),
        num_workers=args.num_workers
    )
    
    # Generate predictions
    print('Generating predictions...')
    predictions = []
    img_ids = []
    
    with torch.no_grad():
        for inputs, batch_ids in tqdm(test_loader):
            # Apply different prediction strategies based on args
            if args.advanced_tta:
                # Advanced TTA (must be batch size 1)
                mask = apply_advanced_tta(model, inputs[0], threshold=args.threshold, device=device)
                if args.postprocess:
                    mask = postprocess_prediction(mask, min_size=args.min_size)
                predictions.append(mask)
                img_ids.append(batch_ids[0])
            elif args.tta:
                # Standard TTA (must be batch size 1)
                mask = apply_tta(model, inputs[0], threshold=args.threshold, device=device)
                if args.postprocess:
                    mask = postprocess_prediction(mask, min_size=args.min_size)
                predictions.append(mask)
                img_ids.append(batch_ids[0])
            else:
                # Standard batch prediction
                inputs = inputs.to(device)
                outputs = model(inputs)
                batch_preds = torch.sigmoid(outputs).cpu().numpy()
                
                # Apply threshold and optional post-processing
                batch_preds = (batch_preds > args.threshold).astype(np.uint8)
                
                # Add to lists
                for i, img_id in enumerate(batch_ids):
                    mask = batch_preds[i].squeeze()
                    if args.postprocess:
                        mask = postprocess_prediction(mask, min_size=args.min_size)
                    predictions.append(mask)
                    img_ids.append(img_id)
    
    # Create submission file
    print('Creating submission file...')
    rles = []
    for mask in predictions:
        rles.append(mask2rle(mask))
    
    submission_df = pd.DataFrame({
        'ImageId': img_ids,
        'EncodedPixels': rles
    })
    
    # Save submission file
    submission_df.to_csv(args.output, index=False)
    print(f'Saved submission to {args.output}')
    
    print('Inference completed successfully!')

if __name__ == '__main__':
    main() 