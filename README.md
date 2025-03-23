# AI-Generated Image Manipulation Detection

This project implements a solution for the binary segmentation challenge to detect AI-manipulated regions in images using PyTorch.

## Project Structure
```
.
├── README.md
├── requirements.txt
├── train.py              # UNet training script with EfficientNet-B6
├── resume_train.py       # Script to resume training from a checkpoint
├── inference.py          # Inference script for generating predictions
├── quick_eval.py         # Script for quick evaluation on sample images
├── dataset.py            # Dataset utilities
├── models.py             # Model architecture
├── utils.py              # Utility functions
├── models/               # Trained model weights
```

## How to Use

### Setup
1. Install dependencies:
```bash
pip install -r requirements.txt
```

### Training
To train the UNet model with EfficientNet-B6 backbone:
```bash
python train.py --data_dir ../train/train --output_dir models --batch_size 16 --epochs 30 --encoder efficientnet-b6
```

Additional options:
- `--random_seed`: Random seed for reproducibility (default: 42)

### Resuming Training
To resume training from a checkpoint:
```bash
python resume_train.py --checkpoint_path models/unet_efficientnet_b6_TIMESTAMP_checkpoint.pth --data_dir ../train/train --additional_epochs 20
```

### Quick Evaluation
To quickly evaluate a model on sample images and visualize results:
```bash
python quick_eval.py --model_path models/unet_efficientnet_b6_TIMESTAMP_best.pth --data_dir ../train/train --model_type unet --encoder efficientnet-b6
```

Additional options:
- `--num_samples`: Number of sample images to evaluate (default: 10)
- `--output_dir`: Directory to save visualizations (default: quick_eval)
- `--threshold`: Threshold for binary segmentation (default: 0.5)

### Inference
To generate predictions on the test set:
```bash
python inference.py --model_path models/unet_efficientnet_b6_TIMESTAMP_best.pth --test_dir ../test/test/images --output submission.csv --model_type unet --encoder efficientnet-b6 --advanced_tta
```

Additional options:
- `--batch_size`: Batch size for inference (default: 32)
- `--threshold`: Threshold for binary segmentation (default: 0.5)
- `--tta`: Use basic test-time augmentation (horizontal and vertical flips)
- `--advanced_tta`: Use advanced test-time augmentation (flips and rotations)
- `--postprocess`: Apply post-processing to clean up predictions
- `--min_size`: Minimum size for connected components in post-processing (default: 50)

## Model Architecture

The project uses a UNet architecture with an EfficientNet-B6 backbone:

- **UNet**: Classic encoder-decoder architecture with skip connections
- **EfficientNet-B6**: Powerful pre-trained encoder providing strong feature extraction
- **Input Size**: 256×256 pixels
- **Output**: Single-channel binary segmentation mask

## Training Approach

- **Loss Function**: Combined BCE and Dice Loss (50% each)
  ```python
  0.5 * bce_loss + 0.5 * dice
  ```
- **Optimizer**: AdamW with weight decay 1e-4
- **Learning Rate**: 1e-4 with Cosine Annealing Warm Restarts
- **Data Split**: 80% training, 20% validation

## Inference Pipeline

For optimal results, we use advanced test-time augmentation (TTA):
- Multiple transformations of test images (flips and rotations)
- Predictions averaged across all variations

## Dataset Information
The dataset consists of pairs of images - manipulated versions and their originals, along with binary masks indicating the manipulated regions.

- **Images**: The manipulated versions of images (in 'images' folder)
- **Masks**: Binary masks showing which areas were manipulated (in 'masks' folder)
- **Originals**: The original, unmanipulated versions (in 'originals' folder)

## Evaluation
The model performance is evaluated using the Dice coefficient, which measures the pixel-wise agreement between predicted segmentation and ground truth.

## Best Model Performance
- **Best Validation Dice**: 0.9554
- **Final Training Dice**: 0.9830 