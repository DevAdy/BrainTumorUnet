import os
import torch
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.utils

# Import your model
from model import UNET

# Inference Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512

# Paths
CHECKPOINT_PATH = "/content/BrainTumorUnet/my_checkpoint.pth.tar"  # Path to your saved checkpoint
INPUT_IMAGE_DIR = "/content/BrainTumorUnet/Data/newimagedata"  # Replace with your new image directory
OUTPUT_MASK_DIR = "/content/BrainTumorUnet/Data/newdatasaved"  # Where segmentation masks will be saved

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_MASK_DIR, exist_ok=True)

# Transforms (same as validation transforms)
inference_transforms = A.Compose([
    A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
    A.Normalize(
        mean=[0.0],
        std=[1.0],
        max_pixel_value=255.0,
    ),
    ToTensorV2(),
])

def load_image(image_path):
    """Load and preprocess a single image"""
    image = np.array(Image.open(image_path).convert("L"), dtype=np.float32)
    image = image / 255.0
    
    # Apply transforms
    transformed = inference_transforms(image=image)
    return transformed['image']

def inference(model, image_dir, output_dir):
    """Run inference on a directory of images"""
    # Set model to evaluation mode
    model.eval()
    
    # Get list of image files
    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    # Process images in batches
    with torch.no_grad():
        for i in range(0, len(image_files), BATCH_SIZE):
            batch_files = image_files[i:i+BATCH_SIZE]
            
            # Load and transform images
            batch_images = []
            for img_file in batch_files:
                img_path = os.path.join(image_dir, img_file)
                img_tensor = load_image(img_path)
                batch_images.append(img_tensor)
            
            # Convert to batch tensor
            batch_tensor = torch.stack(batch_images).to(DEVICE)
            
            # Ensure correct input dimensions
            if batch_tensor.dim() == 3:
                batch_tensor = batch_tensor.unsqueeze(1)
            
            # Run inference
            predictions = torch.sigmoid(model(batch_tensor))
            predictions = (predictions > 0.5).float()
            
            # Save predictions
            for j, pred in enumerate(predictions):
                output_filename = os.path.join(output_dir, f"segmentation_mask_{i+j}.png")
                torchvision.utils.save_image(pred, output_filename)
                print(f"Saved mask for {batch_files[j]}")

def main():
    # Initialize the model
    model = UNET(in_channels=1, out_channels=1).to(DEVICE)
    
    # Load the checkpoint
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['state_dict'])
    
    # Run inference
    inference(model, INPUT_IMAGE_DIR, OUTPUT_MASK_DIR)

if __name__ == "__main__":
    main()