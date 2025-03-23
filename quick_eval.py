import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
from tqdm import tqdm

from models import get_unet_efficientnet, get_fpn, get_deeplabv3plus
from dataset import TestDataset, get_test_loader
from utils import mask2rle, set_seed, rle2mask, get_train_val_split

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Quickly evaluate model on sample images')
    
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model')
    parser.add_argument('--model_type', type=str, default='unet', 
                        choices=['unet', 'fpn', 'deeplabv3plus'],
                        help='Model architecture')
    parser.add_argument('--encoder', type=str, default='efficientnet-b4',
                        help='Encoder used for the model')
    parser.add_argument('--data_dir', type=str, default='train/train',
                        help='Directory with validation images (using train data with masks for evaluation)')
    parser.add_argument('--output_dir', type=str, default='quick_eval',
                        help='Directory to save visualizations')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of sample images to evaluate')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for binary segmentation')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    return parser.parse_args()

def main():
    """Main evaluation function"""
    # Parse arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    set_seed(args.random_seed)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Get image IDs for evaluation (using validation split)
    _, val_ids = get_train_val_split(args.data_dir, val_split=0.2, random_state=args.random_seed)
    
    # Randomly select subset of images
    if len(val_ids) > args.num_samples:
        np.random.shuffle(val_ids)
        val_ids = val_ids[:args.num_samples]
    
    print(f'Evaluating on {len(val_ids)} images')
    
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
    
    # Create a dataset for evaluation (we will use the normal TestDataset)
    test_dataset = TestDataset(
        data_dir=os.path.join(args.data_dir, 'images'),
        img_ids=val_ids,
        img_size=(256, 256)
    )
    
    # Process each image
    dice_scores = []
    
    with torch.no_grad():
        for i, (image, img_id) in enumerate(tqdm(test_dataset)):
            # Get image and move to device
            image = image.unsqueeze(0).to(device)
            
            # Make prediction
            output = model(image)
            pred = torch.sigmoid(output).cpu().numpy()[0, 0]
            pred_binary = (pred > args.threshold).astype(np.uint8)
            
            # Load ground truth mask
            mask_path = os.path.join(args.data_dir, 'masks', f'{img_id}.png')
            gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            gt_mask = (gt_mask > 0).astype(np.uint8)
            
            # Calculate Dice score
            intersection = np.logical_and(pred_binary, gt_mask).sum()
            dice = (2.0 * intersection) / (pred_binary.sum() + gt_mask.sum() + 1e-8)
            dice_scores.append(dice)
            
            # Create visualization
            # Load original image for visualization
            vis_image = cv2.imread(os.path.join(args.data_dir, 'images', f'{img_id}.png'))
            vis_image = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
            
            # Create overlay of prediction
            pred_color = np.zeros_like(vis_image)
            pred_color[pred_binary == 1] = [255, 0, 0]  # Red for prediction
            
            # Create overlay of ground truth
            gt_color = np.zeros_like(vis_image)
            gt_color[gt_mask == 1] = [0, 255, 0]  # Green for ground truth
            
            # Blend with original image
            alpha = 0.5
            pred_blend = cv2.addWeighted(vis_image, 1-alpha, pred_color, alpha, 0)
            gt_blend = cv2.addWeighted(vis_image, 1-alpha, gt_color, alpha, 0)
            
            # Create visualization image
            viz_img = np.concatenate([
                vis_image, 
                pred_blend,
                gt_blend
            ], axis=1)
            
            # Add text with Dice score
            viz_img = cv2.putText(
                img=viz_img,
                text=f'Dice: {dice:.4f}',
                org=(10, 30),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(255, 255, 255),
                thickness=2
            )
            
            # Save visualization
            output_path = os.path.join(args.output_dir, f'{img_id}_eval.png')
            cv2.imwrite(output_path, cv2.cvtColor(viz_img, cv2.COLOR_RGB2BGR))
    
    # Calculate average Dice score
    avg_dice = np.mean(dice_scores)
    print(f'Average Dice score: {avg_dice:.4f}')
    
    # Create a summary image with all Dice scores
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(dice_scores)), dice_scores)
    plt.axhline(y=avg_dice, color='r', linestyle='-', label=f'Average: {avg_dice:.4f}')
    plt.xlabel('Image Index')
    plt.ylabel('Dice Score')
    plt.title('Dice Scores for Sample Images')
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'dice_scores.png'))
    
    print(f'Visualizations saved to {args.output_dir}')

if __name__ == '__main__':
    main() 