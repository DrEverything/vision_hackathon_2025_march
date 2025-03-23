import torch
import torch.nn as nn
import torch.optim as optim
import segmentation_models_pytorch as smp
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F

def dice_coef(y_pred, y_true, smooth=1):
    """
    Calculate Dice coefficient
    
    Args:
        y_pred: predicted binary mask
        y_true: true binary mask
        smooth: smoothing factor
        
    Returns:
        Dice coefficient
    """
    # Ensure that predictions and targets have the same shape
    if y_pred.shape != y_true.shape:
        y_pred = F.interpolate(y_pred.unsqueeze(1) if y_pred.dim() == 3 else y_pred, 
                               size=y_true.shape[2:], mode='bilinear', align_corners=False)
        y_pred = y_pred.squeeze(1) if y_pred.dim() == 4 and y_pred.shape[1] == 1 else y_pred
    
    # Make sure tensors are flattened to 1D
    y_pred_flat = y_pred.reshape(-1)
    y_true_flat = y_true.reshape(-1)
    
    intersection = (y_pred_flat * y_true_flat).sum()
    return (2. * intersection + smooth) / (y_pred_flat.sum() + y_true_flat.sum() + smooth)

def dice_loss(y_pred, y_true):
    """
    Calculate Dice loss
    
    Args:
        y_pred: predicted binary mask
        y_true: true binary mask
        
    Returns:
        Dice loss
    """
    return 1 - dice_coef(y_pred, y_true)

def bce_dice_loss(y_pred, y_true):
    """
    Combine binary cross entropy and Dice loss
    
    Args:
        y_pred: predicted binary mask
        y_true: true binary mask
        
    Returns:
        Combined BCE and Dice loss
    """
    bce = nn.BCEWithLogitsLoss()(y_pred, y_true)
    dice = dice_loss(torch.sigmoid(y_pred), y_true)
    return bce + dice

def get_unet_efficientnet(input_channels=3, encoder='efficientnet-b4', encoder_weights='imagenet'):
    """
    Create a U-Net model with EfficientNet backbone
    
    Args:
        input_channels: number of input channels
        encoder: backbone model to use (e.g. 'efficientnet-b4')
        encoder_weights: weights to use for the backbone (e.g. 'imagenet')
        
    Returns:
        PyTorch model
    """
    # Initialize using segmentation_models_pytorch
    model = smp.Unet(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=input_channels,
        classes=1
    )
    
    return model

def get_deeplabv3plus(input_channels=3, encoder='efficientnet-b4', encoder_weights='imagenet'):
    """
    Create a DeepLabV3+ model with EfficientNet backbone
    
    Args:
        input_channels: number of input channels
        encoder: backbone model to use (e.g. 'efficientnet-b4')
        encoder_weights: weights to use for the backbone (e.g. 'imagenet')
        
    Returns:
        PyTorch model
    """
    # Initialize using segmentation_models_pytorch
    model = smp.DeepLabV3Plus(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=input_channels,
        classes=1
    )
    
    return model

def get_fpn(input_channels=3, encoder='efficientnet-b4', encoder_weights='imagenet'):
    """
    Create a Feature Pyramid Network (FPN) model with EfficientNet backbone
    
    Args:
        input_channels: number of input channels
        encoder: backbone model to use (e.g. 'efficientnet-b4')
        encoder_weights: weights to use for the backbone (e.g. 'imagenet')
        
    Returns:
        PyTorch model
    """
    # Initialize using segmentation_models_pytorch
    model = smp.FPN(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=input_channels,
        classes=1
    )
    
    return model

def get_unetplusplus(input_channels=3, encoder='efficientnet-b4', encoder_weights='imagenet'):
    """
    Create a UNet++ model with specified backbone
    
    Args:
        input_channels: number of input channels
        encoder: backbone model to use (e.g. 'efficientnet-b4')
        encoder_weights: weights to use for the backbone (e.g. 'imagenet')
        
    Returns:
        PyTorch model
    """
    # Initialize using segmentation_models_pytorch
    model = smp.UnetPlusPlus(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=input_channels,
        classes=1
    )
    
    return model

def get_manet(input_channels=3, encoder='efficientnet-b4', encoder_weights='imagenet'):
    """
    Create a MAnet model with specified backbone
    
    Args:
        input_channels: number of input channels
        encoder: backbone model to use (e.g. 'efficientnet-b4')
        encoder_weights: weights to use for the backbone (e.g. 'imagenet')
        
    Returns:
        PyTorch model
    """
    # Initialize using segmentation_models_pytorch
    model = smp.MAnet(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=input_channels,
        classes=1
    )
    
    return model

def get_linknet(input_channels=3, encoder='efficientnet-b4', encoder_weights='imagenet'):
    """
    Create a LinkNet model with specified backbone
    
    Args:
        input_channels: number of input channels
        encoder: backbone model to use (e.g. 'efficientnet-b4')
        encoder_weights: weights to use for the backbone (e.g. 'imagenet')
        
    Returns:
        PyTorch model
    """
    # Initialize using segmentation_models_pytorch
    model = smp.Linknet(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=input_channels,
        classes=1
    )
    
    return model

def get_pspnet(input_channels=3, encoder='efficientnet-b4', encoder_weights='imagenet'):
    """
    Create a PSPNet model with specified backbone
    
    Args:
        input_channels: number of input channels
        encoder: backbone model to use (e.g. 'efficientnet-b4')
        encoder_weights: weights to use for the backbone (e.g. 'imagenet')
        
    Returns:
        PyTorch model
    """
    # Initialize using segmentation_models_pytorch
    model = smp.PSPNet(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=input_channels,
        classes=1
    )
    
    return model

def get_pan(input_channels=3, encoder='efficientnet-b4', encoder_weights='imagenet'):
    """
    Create a PAN model with specified backbone
    
    Args:
        input_channels: number of input channels
        encoder: backbone model to use (e.g. 'efficientnet-b4')
        encoder_weights: weights to use for the backbone (e.g. 'imagenet')
        
    Returns:
        PyTorch model
    """
    # Initialize using segmentation_models_pytorch
    model = smp.PAN(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=input_channels,
        classes=1
    )
    
    return model

def get_optimizer(model, lr=1e-4):
    """
    Get optimizer for training
    
    Args:
        model: PyTorch model
        lr: learning rate
        
    Returns:
        PyTorch optimizer
    """
    return optim.Adam(model.parameters(), lr=lr)

def get_scheduler(optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6):
    """
    Get learning rate scheduler
    
    Args:
        optimizer: PyTorch optimizer
        mode: mode for scheduler ('min' or 'max')
        factor: factor by which to reduce learning rate
        patience: number of epochs to wait for improvement
        min_lr: minimum learning rate
        
    Returns:
        PyTorch scheduler
    """
    return ReduceLROnPlateau(
        optimizer, 
        mode=mode, 
        factor=factor, 
        patience=patience, 
        verbose=True,
        min_lr=min_lr
    ) 