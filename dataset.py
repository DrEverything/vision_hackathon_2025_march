import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

class SegmentationDataset(Dataset):
    def __init__(
        self,
        data_dir,
        img_ids,
        img_size=(256, 256),
        augment=False,
        use_originals=False
    ):
        """
        Dataset for segmentation task
        
        Args:
            data_dir: directory with images, masks, and originals subdirectories
            img_ids: list of image IDs to use
            img_size: image size for model input
            augment: whether to apply data augmentation
            use_originals: whether to include original images in the input (as additional channel)
        """
        self.data_dir = data_dir
        self.img_ids = img_ids
        self.img_size = img_size
        self.augment = augment
        self.use_originals = use_originals
        
        # If using originals, filter out IDs that don't have original images
        if self.use_originals:
            valid_img_ids = []
            for img_id in self.img_ids:
                orig_path = os.path.join(self.data_dir, 'originals', f'{img_id}.png')
                if os.path.exists(orig_path):
                    valid_img_ids.append(img_id)
            
            # Update the list of image IDs
            self.img_ids = valid_img_ids
            print(f"Filtered to {len(self.img_ids)} images that have original versions")
        
        # Define base transforms (always applied)
        self.base_transform = A.Compose([
            A.Resize(height=img_size[0], width=img_size[1]),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
        
        # Define augmentations (only applied during training)
        if augment:
            self.aug_transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.1, 
                    rotate_limit=15, p=0.5
                ),
                A.OneOf([
                    A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.5),
                    A.GridDistortion(p=0.5),
                    A.OpticalDistortion(distort_limit=1, p=0.5),
                ], p=0.3),
                A.OneOf([
                    A.RandomBrightnessContrast(p=0.5),
                    A.HueSaturationValue(p=0.5),
                ], p=0.3),
            ])
    
    def __len__(self):
        """Number of samples in dataset"""
        return len(self.img_ids)
    
    def __getitem__(self, idx):
        """Get sample at position idx"""
        img_id = self.img_ids[idx]
        
        # Load image
        img_path = os.path.join(self.data_dir, 'images', f'{img_id}.png')
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask_path = os.path.join(self.data_dir, 'masks', f'{img_id}.png')
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype(np.float32)
        
        # Load original image if needed
        if self.use_originals:
            orig_path = os.path.join(self.data_dir, 'originals', f'{img_id}.png')
            orig = cv2.imread(orig_path)
            orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        
        # Apply augmentations
        if self.augment:
            if self.use_originals:
                # Augment both images with the same transformation
                augmented = self.aug_transform(image=img, image1=orig, mask=mask)
                img = augmented['image']
                orig = augmented['image1']
                mask = augmented['mask']
        
        # Apply base transforms
        if self.use_originals:
            # Transform both images
            transformed_img = self.base_transform(image=img)['image']
            transformed_orig = self.base_transform(image=orig)['image']
            
            # Concatenate manipulated and original images
            x = torch.cat([transformed_img, transformed_orig], dim=0)
        else:
            # Transform only the manipulated image
            x = self.base_transform(image=img)['image']
        
        # Transform mask
        y = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        
        return x, y


class TestDataset(Dataset):
    def __init__(
        self,
        data_dir,
        img_ids,
        img_size=(256, 256),
    ):
        """
        Dataset for test prediction
        
        Args:
            data_dir: directory with test images
            img_ids: list of image IDs to use
            img_size: image size for model input
        """
        self.data_dir = data_dir
        self.img_ids = img_ids
        self.img_size = img_size
        
        # Define transforms
        self.transform = A.Compose([
            A.Resize(height=img_size[0], width=img_size[1]),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    
    def __len__(self):
        """Number of samples in dataset"""
        return len(self.img_ids)
    
    def __getitem__(self, idx):
        """Get sample at position idx"""
        img_id = self.img_ids[idx]
        
        # Load image
        img_path = os.path.join(self.data_dir, f'{img_id}.png')
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        x = self.transform(image=img)['image']
        
        return x, img_id
        
def get_data_loaders(data_dir, train_ids, val_ids, batch_size=16, img_size=(256, 256), 
                    num_workers=4, use_originals=False):
    """
    Create PyTorch DataLoaders for training and validation
    
    Args:
        data_dir: directory with images, masks, and originals subdirectories
        train_ids: list of image IDs for training
        val_ids: list of image IDs for validation
        batch_size: batch size
        img_size: image size for model input
        num_workers: number of workers for data loading
        use_originals: whether to include original images in the input
        
    Returns:
        train_loader, val_loader: PyTorch DataLoaders
    """
    # Create datasets
    train_dataset = SegmentationDataset(
        data_dir=data_dir,
        img_ids=train_ids,
        img_size=img_size,
        augment=True,
        use_originals=use_originals
    )
    
    val_dataset = SegmentationDataset(
        data_dir=data_dir,
        img_ids=val_ids,
        img_size=img_size,
        augment=False,
        use_originals=use_originals
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

def get_test_loader(data_dir, test_ids, batch_size=32, img_size=(256, 256), num_workers=4):
    """
    Create PyTorch DataLoader for test data
    
    Args:
        data_dir: directory with test images
        test_ids: list of image IDs for testing
        batch_size: batch size
        img_size: image size for model input
        num_workers: number of workers for data loading
        
    Returns:
        test_loader: PyTorch DataLoader
    """
    # Create dataset
    test_dataset = TestDataset(
        data_dir=data_dir,
        img_ids=test_ids,
        img_size=img_size
    )
    
    # Create data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return test_loader 