import numpy as np
import cv2
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch

def mask2rle(img):
    """
    Convert mask to RLE format
    
    Args:
        img: numpy array, 1 - mask, 0 - background
        
    Returns:
        run length as string formatted
    """
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def rle2mask(mask_rle, shape=(256, 256)):
    """
    Convert RLE to mask
    
    Args:
        mask_rle: run-length as string formatted (start length)
        shape: (height, width) of array to return
        
    Returns:
        numpy array, 1 - mask, 0 - background
    """
    if mask_rle == '':
        return np.zeros(shape, dtype=np.uint8)
    
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)

def get_train_val_split(data_dir, val_split=0.2, random_state=42):
    """
    Split the dataset into training and validation sets
    
    Args:
        data_dir: directory containing the dataset
        val_split: proportion of data to use for validation
        random_state: random seed
        
    Returns:
        train_ids, val_ids: lists of image IDs for training and validation
    """
    image_dir = os.path.join(data_dir, 'images')
    image_ids = [os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.endswith('.png')]
    
    train_ids, val_ids = train_test_split(
        image_ids, test_size=val_split, random_state=random_state
    )
    
    return train_ids, val_ids

def create_submission(image_ids, masks, output_path):
    """
    Create submission file
    
    Args:
        image_ids: list of image IDs
        masks: list of predicted masks
        output_path: path to save the submission file
    """
    rles = []
    for mask in masks:
        rles.append(mask2rle(mask))
    
    df = pd.DataFrame({
        'ImageId': image_ids,
        'EncodedPixels': rles
    })
    
    df.to_csv(output_path, index=False)
    
    return df

def set_seed(seed):
    """
    Set seed for reproducibility
    
    Args:
        seed: random seed
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 