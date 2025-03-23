import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models import (
    get_unet_efficientnet, get_fpn, get_deeplabv3plus, get_unetplusplus, get_manet,
    get_optimizer, get_scheduler, dice_coef, dice_loss, bce_dice_loss
)
from dataset import SegmentationDataset, get_data_loaders
from utils import get_train_val_split, set_seed
from train import (
    train_epoch, validate_epoch, save_sample_predictions, 
    boundary_aware_loss, focal_loss, combo_loss, parse_args as train_parse_args
)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Resume training segmentation model from checkpoint')
    
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to the checkpoint file to resume from')
    parser.add_argument('--data_dir', type=str, default='train/train',
                        help='Directory containing the training data')
    parser.add_argument('--output_dir', type=str, default='models',
                        help='Directory to save model checkpoints')
    parser.add_argument('--model_type', type=str, default='unet', 
                        choices=['unet', 'fpn', 'deeplabv3plus', 'unetplusplus', 'manet'],
                        help='Model architecture to use')
    parser.add_argument('--encoder', type=str, default='efficientnet-b4',
                        help='Encoder network for the model')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--additional_epochs', type=int, default=20,
                        help='Number of additional epochs to train')
    parser.add_argument('--img_size', type=int, default=256,
                        help='Size of input images')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Proportion of data to use for validation')
    parser.add_argument('--advanced_loss', action='store_true',
                        help='Use advanced boundary-aware loss function')
    parser.add_argument('--scheduler_type', type=str, default='plateau',
                        choices=['plateau', 'cosine'],
                        help='Type of learning rate scheduler to use')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for optimizer')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    return parser.parse_args()

def main():
    """Main function for resuming training"""
    # Parse arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    set_seed(args.random_seed)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path)
    
    # Extract information from checkpoint
    epoch = checkpoint['epoch']
    best_dice = checkpoint.get('best_dice', 0.0)
    history = checkpoint.get('history', {
        'train_loss': [],
        'train_dice': [],
        'val_loss': [],
        'val_dice': []
    })
    
    print(f"Resuming from epoch {epoch+1}, best Dice: {best_dice:.4f}")
    
    # Split data into training and validation
    train_ids, val_ids = get_train_val_split(
        args.data_dir, val_split=args.val_split, random_state=args.random_seed
    )
    
    print(f'Training on {len(train_ids)} images, validating on {len(val_ids)} images')
    
    # Determine if using originals from the checkpoint filename
    use_originals = 'use_originals' in args.checkpoint_path
    
    # Create data loaders
    train_loader, val_loader = get_data_loaders(
        data_dir=args.data_dir,
        train_ids=train_ids,
        val_ids=val_ids,
        batch_size=args.batch_size,
        img_size=(args.img_size, args.img_size),
        num_workers=args.num_workers,
        use_originals=use_originals
    )
    
    # Determine input channels based on whether originals are used
    input_channels = 6 if use_originals else 3
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create model
    print(f'Creating {args.model_type} model with {args.encoder} encoder')
    if args.model_type == 'unet':
        model = get_unet_efficientnet(input_channels=input_channels, encoder=args.encoder)
    elif args.model_type == 'fpn':
        model = get_fpn(input_channels=input_channels, encoder=args.encoder)
    elif args.model_type == 'deeplabv3plus':
        model = get_deeplabv3plus(input_channels=input_channels, encoder=args.encoder)
    elif args.model_type == 'unetplusplus':
        model = get_unetplusplus(input_channels=input_channels, encoder=args.encoder)
    elif args.model_type == 'manet':
        model = get_manet(input_channels=input_channels, encoder=args.encoder)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Define loss function
    if args.advanced_loss:
        criterion = combo_loss  # Use the advanced loss function
        print("Using advanced combination loss (Dice + Focal + Boundary)")
    else:
        criterion = nn.BCEWithLogitsLoss()
        print("Using standard BCE with Logits loss")
    
    # Create optimizer with weight decay
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=1e-4,  # Default lr, will be overridden by optimizer state
        weight_decay=args.weight_decay
    )
    
    # Load optimizer state
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Create scheduler based on user choice
    if args.scheduler_type == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )
        print("Using ReduceLROnPlateau scheduler")
    else:  # cosine
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=1, eta_min=1e-6, verbose=True
        )
        print("Using CosineAnnealingWarmRestarts scheduler")
    
    # Extract model name from checkpoint path
    model_name = os.path.basename(args.checkpoint_path).split('_checkpoint')[0]
    
    # Define paths for saving models
    best_model_path = os.path.join(args.output_dir, f"{model_name}_best.pth")
    
    # Visualization frequency (every N epochs)
    vis_frequency = 5
    
    try:
        # Resume training
        print('Resuming training...')
        best_epoch = epoch
        total_epochs = epoch + args.additional_epochs + 1
        
        for epoch in range(epoch + 1, total_epochs):
            print(f'Epoch {epoch+1}/{total_epochs}')
            
            # Train and validate for one epoch
            train_loss, train_dice = train_epoch(
                model, train_loader, criterion, optimizer, device, args.advanced_loss
            )
            val_loss, val_dice = validate_epoch(
                model, val_loader, criterion, device, args.advanced_loss
            )
            
            # Update learning rate
            if args.scheduler_type == 'plateau':
                scheduler.step(val_dice)
            else:  # cosine
                scheduler.step()
            
            # Update history
            history['train_loss'].append(train_loss)
            history['train_dice'].append(train_dice)
            history['val_loss'].append(val_loss)
            history['val_dice'].append(val_dice)
            
            # Print epoch results
            print(f'Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}')
            print(f'Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}')
            
            # Save best model
            if val_dice > best_dice:
                best_dice = val_dice
                best_epoch = epoch
                torch.save(model.state_dict(), best_model_path)
                print(f'Saved best model to {best_model_path} with Dice: {best_dice:.4f}')
                
                # Save sample predictions whenever we save the best model
                save_sample_predictions(model, val_loader, args.output_dir, epoch+1, device)
            
            # Save checkpoint every epoch for resuming if needed
            checkpoint_path = os.path.join(args.output_dir, f"{model_name}_checkpoint.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dice': best_dice,
                'history': history
            }, checkpoint_path)
            
            # Generate sample predictions on some validation images periodically
            if (epoch + 1) % vis_frequency == 0:
                save_sample_predictions(model, val_loader, args.output_dir, epoch+1, device)
            
    except KeyboardInterrupt:
        print('Training interrupted by user')
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, f"{model_name}_final.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f'Saved final model to {final_model_path}')
    
    # Save training history
    history_df = pd.DataFrame(history)
    history_path = os.path.join(args.output_dir, f"{model_name}_history.csv")
    history_df.to_csv(history_path, index=False)
    print(f'Saved training history to {history_path}')
    
    # Plot training history
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_dice'])
    plt.plot(history['val_dice'])
    plt.title('Dice Coefficient')
    plt.ylabel('Dice')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'])
    plt.plot(history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f"{model_name}_history.png"))
    
    print(f'Training completed successfully! Best Dice: {best_dice:.4f} at epoch {best_epoch+1}')

if __name__ == '__main__':
    main() 