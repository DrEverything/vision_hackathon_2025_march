import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from datetime import datetime
import cv2
from models import get_unet_efficientnet, dice_coef
from dataset import get_data_loaders
from utils import get_train_val_split, set_seed

def parse_args():
    parser = argparse.ArgumentParser(description='Focused training for high Dice scores')
    parser.add_argument('--data_dir', type=str, default='../train/train',
                        help='Directory containing the training data')
    parser.add_argument('--output_dir', type=str, default='models',
                        help='Directory to save model checkpoints')
    parser.add_argument('--encoder', type=str, default='efficientnet-b6',
                        help='Encoder network for the model')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs to train')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility')
    return parser.parse_args()

def combined_loss(pred, target):
    # Binary Cross-Entropy Loss
    bce_loss = nn.BCEWithLogitsLoss()(pred, target)
    # Dice Loss 
    pred_sigmoid = torch.sigmoid(pred)
    dice = 1 - dice_coef(pred_sigmoid, target)
    # Combined loss
    return 0.5 * bce_loss + 0.5 * dice

def main():
    args = parse_args()
    set_seed(args.random_seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    train_ids, val_ids = get_train_val_split(args.data_dir, val_split=0.2, random_state=args.random_seed)
    print(f'Training on {len(train_ids)} images, validating on {len(val_ids)} images')
    
    train_loader, val_loader = get_data_loaders(
        data_dir=args.data_dir, train_ids=train_ids, val_ids=val_ids,
        batch_size=args.batch_size, img_size=(256, 256), num_workers=4
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    print(f'Creating UNet model with {args.encoder} encoder')
    model = get_unet_efficientnet(input_channels=3, encoder=args.encoder)
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=1e-6
    )
    
    best_dice = 0.0
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_name = f"unet_{args.encoder.replace('-', '_')}_{timestamp}"
    best_model_path = os.path.join(args.output_dir, f"{model_name}_best.pth")
    
    print('Starting training for high Dice score...')
    for epoch in range(args.epochs):
        print(f'Epoch {epoch+1}/{args.epochs}')
        
        # Train
        model.train()
        train_loss = 0
        train_dice = 0
        
        for inputs, targets in tqdm(train_loader, desc='Training'):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = combined_loss(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            train_dice += dice_coef(torch.sigmoid(outputs), targets).item() * inputs.size(0)
        
        train_loss /= len(train_loader.dataset)
        train_dice /= len(train_loader.dataset)
        
        # Validate
        model.eval()
        val_loss = 0
        val_dice = 0
        
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc='Validation'):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = combined_loss(outputs, targets)
                
                val_loss += loss.item() * inputs.size(0)
                val_dice += dice_coef(torch.sigmoid(outputs), targets).item() * inputs.size(0)
        
        val_loss /= len(val_loader.dataset)
        val_dice /= len(val_loader.dataset)
        
        scheduler.step()
        
        print(f'Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}')
        
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), best_model_path)
            print(f'Saved best model to {best_model_path} with Dice: {best_dice:.4f}')
    
    print(f'\nTraining completed! Best Dice: {best_dice:.4f}')
    print(f'\nFor top results, use:')
    print(f'python inference.py --model_path {best_model_path} --test_dir ../test/test/images --output submission.csv --model_type unet --encoder {args.encoder} --advanced_tta --postprocess')

if __name__ == '__main__':
    main() 